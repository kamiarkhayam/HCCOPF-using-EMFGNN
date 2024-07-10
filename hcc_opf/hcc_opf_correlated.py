# -*- coding: utf-8 -*-
"""
A module for performing hybrid chance-constrained optimal power flow analysis with correlated uncertainties.
This module uses particle swarm optimization to find the optimal settings in power generation and load distribution
while considering potential outages and variable renewable outputs.

Created by: Kamiar Khayambashi
"""

import argparse
import random
import numpy as np
import pickle
import torch
import pypower.api as pp
import datetime
import os
import copy
import time
from EMF_GNN import HFGNN, MFGNN
from MF_GNN_DC import MFGNN_DC
from LF_GNN import LFGNN
from utils import *



# =============================================================================
# =============================================================================
# Required functions
# =============================================================================
# =============================================================================


def perform_power_flow_analysis(HF_model, LF_model, MF_model, active_demands, reactive_demands, particle, case, num_generators):
    """
    Simulates the power flow using the specified models and updates the case with results.

    Args:
        HF_model (torch.nn.Module): High-fidelity GNN model.
        LF_model (torch.nn.Module): Low-fidelity GNN model.
        MF_model (torch.nn.Module): Multi-fidelity GNN model.
        active_demands (list): List of active demands for each bus.
        reactive_demands (list): List of reactive demands for each bus.
        particle (list): PSO algorithm particle representing generation settings.
        case (dict): PYPOWER case.
        num_generators (int): Number of traditional generators in the system.
        args (Namespace): Command line arguments containing settings for the simulation.

    Returns:
        tuple: A tuple containing arrays of predicted voltage magnitudes and angles.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outage_adjustment = -1 if args.outage_type == 'gen_out' else 0
    num_generators += outage_adjustment
    
    # Scale demands to become p.u.
    active_demands = np.array(active_demands) / 100
    reactive_demands = np.array(reactive_demands) / 100

    # Prepare generator indices and initial generation settings
    bus_types = np.array(case['bus'][:, 1])
    gen_indices = np.where((bus_types == 2) | (bus_types == 3))[0]
    active_generations = np.zeros_like(bus_types, dtype=float)
    reactive_generations = np.zeros_like(bus_types, dtype=float)
    voltage_setpoints = np.zeros_like(bus_types, dtype=float)

    # Setup generation values from chromosome
    active_indices = np.arange(0, len(particle), 2)
    reactive_indices = np.arange(1, len(particle), 2)
    active_generations[gen_indices] = np.take(particle, active_indices) / 100
    reactive_generations[gen_indices] = np.take(particle, reactive_indices) / 100
    voltage_setpoints[gen_indices] = case["gen"][:, 5]
    
    # Create edge indices for graph neural network
    edge_start_nodes = case['branch'][:, 0]
    edge_end_nodes = case['branch'][:, 1]
    edge_index = torch.tensor(np.vstack((edge_start_nodes, edge_end_nodes)).astype(int) - 1, dtype=torch.int64).to(device)

    # Prepare input features for the GNN
    if args.analysis_type == 'dc_approx':
        
        case['gen'][:num_generators, 1] = particle[::2]
        case['gen'][:num_generators, 2] = particle[1::2]
        
        ppopt = pp.ppoption(VERBOSE=0, OUT_ALL=0)
        result_dc = pp.rundcpf(case, ppopt=ppopt)
        
        dc_bus_voltage_magnitudes = result_dc[0]['bus'][:, 7]
        dc_bus_voltage_angles = result_dc[0]['bus'][:, 8] * np.pi / 180
        
        x = torch.tensor(np.concatenate(
                (
                    active_demands.reshape(-1, 1),
                    reactive_demands.reshape(-1, 1),
                    active_generations.reshape(-1, 1),
                    reactive_generations.reshape(-1, 1),
                    voltage_setpoints.reshape(-1, 1)
                ),
                axis=1,
            )).to(device)
        x = x.to(torch.float32)  
        
        additional_x = torch.tensor(np.concatenate(
                (
                    dc_bus_voltage_magnitudes.reshape(-1, 1),
                    dc_bus_voltage_angles.reshape(-1, 1)
                ),
                axis=1,
            )).to(device)
        additional_x = additional_x.to(torch.float32)

        
    elif args.analysis_type == 'normal' and args.model_type == 'SF':
        x = torch.tensor(np.concatenate(
                (
                    active_demands.reshape(-1, 1),
                    reactive_demands.reshape(-1, 1),
                    active_generations.reshape(-1, 1),
                    reactive_generations.reshape(-1, 1),
                    voltage_setpoints.reshape(-1, 1)
                ),
                axis=1,
            )).to(device)
        x = x.to(torch.float32)  
        
    else:

        x = torch.tensor(np.concatenate(
                (
                    active_demands.reshape(-1, 1),
                    reactive_demands.reshape(-1, 1),
                    active_generations.reshape(-1, 1),
                    reactive_generations.reshape(-1, 1),
                    voltage_setpoints.reshape(-1, 1)
                ),
                axis=1,
            )).to(device)
        x = x.to(torch.float32)  
        

        LF_model.eval()

        with torch.no_grad():
            #time_in = time.time()
            LF_predicted_voltages, LF_embedding = LF_model(x, edge_index)
            
        low_fidelity_phase = LF_predicted_voltages[:, 0].view(-1, 1)
        low_fidelity_voltage = torch.ones_like(low_fidelity_phase)

        additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1).to(torch.float32) 

    
    
    model = HF_model
    if args.analysis_type == 'dc_approx' or args.model_type == 'MF':
        model = MF_model    

    model.eval()

    # Perform power flow analysis using the GNN model
    with torch.no_grad():
        #time_in = time.time()
        if args.model_type == 'SF' and args.analysis_type == 'normal':
           predicted_voltages = model(x, edge_index)
        elif args.model_type == 'MF' and args.analysis_type == 'normal':
            predicted_voltages =  model(x, edge_index, additional_x, LF_embedding)
        else:
            predicted_voltages =  model(x, edge_index, additional_x)
    

    # Extract predicted voltage magnitudes and angles
    predicted_voltages_mag = predicted_voltages[:, 0].cpu().numpy()
    predicted_voltages_angle = predicted_voltages[:, 1].cpu().numpy() * 180 / np.pi

    return predicted_voltages_mag, predicted_voltages_angle


def perform_power_flow_analysis_pp(case, active_demands, reactive_demands, particle, num_generators, args):
    """
    Performs power flow analysis using PYPOWER's power flow solver. This function adjusts the power system case based on the
    position vector provided by PSO and executes a power flow analysis.

    Args:
        case (dict): The power system case dictionary which includes grid and generator data.
        active_demands (np.array): Array of active power demands at each bus.
        reactive_demands (np.array): Array of reactive power demands at each bus.
        particle (list): Particle position in PSO, representing generation settings.
        num_generators (int): Number of generators in the system affected by the position vector.
        args (Namespace): Contains settings like the type of outage, which affect the analysis.

    Returns:
        tuple: Tuple of numpy arrays containing the magnitudes and angles of bus voltages.
    """
    # Set generator outputs based on the position vector from PSO
    case['gen'][:num_generators, 1] = particle[::2]  # Active power outputs
    case['gen'][:num_generators, 2] = particle[1::2]  # Reactive power outputs

    # Update bus loads
    case["bus"][:, 2] = active_demands  # Active demands
    case["bus"][:, 3] = reactive_demands  # Reactive demands

    if args.outage_type == 'gen_out':
        num_generators -= 1  # Adjust the number of generators if there's an outage

    # Run the power flow analysis with PYPOWER
    ppopt = pp.ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-12)
    results = pp.runpf(case, ppopt=ppopt)

    # Extract the results
    predicted_voltages_mag = results[0]['bus'][:, 7]
    predicted_voltages_angle = results[0]['bus'][:, 8]

    return predicted_voltages_mag, predicted_voltages_angle


def power_loss(v1_mag, v1_angle_deg, v2_mag, v2_angle_deg, reactance, resistance, baseMVa):
    """
    Calculates the real and reactive power losses in a transmission line based on the voltage magnitudes and angles
    at both ends of the line, and the line's electrical characteristics.

    Args:
        v1_mag (float or np.array): Voltage magnitude at the start of the line.
        v1_angle_deg (float or np.array): Voltage angle in degrees at the start of the line.
        v2_mag (float or np.array): Voltage magnitude at the end of the line.
        v2_angle_deg (float or np.array): Voltage angle in degrees at the end of the line.
        reactance (float): Reactance of the line.
        resistance (float): Resistance of the line.
        baseMVa (float): Base power of the system in MVA.

    Returns:
        tuple: Real (P_loss) and reactive (Q_loss) power losses in the line.
    """
    # Convert voltage magnitudes and angles to complex numbers
    v1 = v1_mag * np.exp(1j * np.radians(v1_angle_deg))
    v2 = v2_mag * np.exp(1j * np.radians(v2_angle_deg))

    # Calculate the voltage drop across the transmission line
    deltaV = v1 - v2

    # Calculate the impedance of the transmission line
    impedance = resistance + 1j * reactance

    # Calculate the complex current (I) flowing through the line
    current = deltaV / impedance

    # Calculate the complex power loss (S_loss) in the line
    S_loss = current * np.conjugate(current) * impedance

    # Extract real and imaginary parts of the power loss
    P_loss = S_loss.real
    Q_loss = S_loss.imag

    return P_loss, Q_loss


def check_opf_constraints(predicted_voltages_mag, predicted_voltages_angle, active_demand, reactive_demand, particle, case, num_generators):
    """
    Checks various operational constraints for an OPF problem, including power balance, voltage magnitudes,
    generator outputs, and branch flows.

    Args:
        predicted_voltages_mag (np.array): Predicted voltage magnitudes at each bus.
        predicted_voltages_angle (np.array): Predicted voltage angles at each bus.
        active_demand (np.array): Active power demand at each bus.
        reactive_demand (np.array): Reactive power demand at each bus.
        particle (list): Particle position in PSO, representing generation settings.
        case (dict): The power system case dictionary which includes all bus and branch data.
        num_generators (int): Number of generators in the system.

    Returns:
        dict: A dictionary containing boolean values for each constraint type indicating if it is violated.
        dict: A dictionary containing the actual numerical values of each constraint type.
        float: Total power imbalance in the system.
        dict: A dictionary with the distance to violation for each constraint type.
    """
    
    violations = {
        'Power Balance': None,
        'Voltage Magnitude': None,
        'Generator Real Power Output Limits': None,
        'Generator Reactive Power Output Limits': None,
        'Voltage Angle Difference': None,
        'Branch Flow': None
    }
    
    violations_bin = {
        'Power Balance': None,
        'Voltage Magnitude': None,
        'Generator Real Power Output Limits': None,
        'Generator Reactive Power Output Limits': None,
        'Voltage Angle Difference': None,
        'Branch Flow': None
    }
    
    distances = {
        'Power Balance': None,
        'Voltage Magnitude': None,
        'Generator Real Power Output Limits': None,
        'Generator Reactive Power Output Limits': None,
        'Voltage Angle Difference': None,
        'Branch Flow': None
    }
    
    active_indices = np.arange(0, len(particle), 2)
    active_generations = np.take(particle, active_indices)
    active_generations = np.hstack((active_generations, case['gen'][num_generators:, 1]))
    
    
    reactive_indices = np.arange(1, len(particle), 2)
    reactive_generations = np.take(particle, reactive_indices)
    
    # Calculate power balance violations
    line_info = np.array(case['branch'][:, :2]) - 1
    start = line_info[:, 0].astype(int)
    end = line_info[:, 1].astype(int)
    
    base_MVA = 100
    resistance = case['branch'][:, 2]
    reactance = case['branch'][:, 3]
    
    # Assuming power_loss is a function that operates element-wise on the inputs
    line_losses, q_loss = power_loss(predicted_voltages_mag[start], predicted_voltages_angle[start], predicted_voltages_mag[end], predicted_voltages_angle[end], reactance, resistance, base_MVA)
    
    line_losses = np.sum(np.array(line_losses))
    
    real_demand = np.sum(active_demand)
    #reactive_demand = np.sum(reactive_demand)
    real_generation = np.sum(active_generations)
    #reactive_generation = np.sum(reactive_generations)
    real_load = real_demand + line_losses - real_generation
    #reactive_load = reactive_demand - reactive_generation
    #if reactive_load <= 10:
    #    reactive_load = 0
    imbalance = real_load
        
    if real_load <= 0.1 * real_generation:
        real_load = 0
    else:
        real_load -= 0.1 * real_generation
    
    power_balance_violation = np.abs(real_load) 
    violations['Power Balance'] = power_balance_violation
    violations_bin['Power Balance'] = int(violations['Power Balance']>0)
    distances['Power Balance'] = imbalance - 0.1 * real_generation
    
    # Calculate voltage magnitude violations
    voltage_magnitude_violations_more = np.maximum(np.array(predicted_voltages_mag) - np.array(case['bus'][:, 11]), 0)
    voltage_magnitude_violations_less = np.minimum(np.array(predicted_voltages_mag) - np.array(case['bus'][:, 12]), 0)                                               
    
    voltage_magnitude_violations = voltage_magnitude_violations_more + voltage_magnitude_violations_less
    voltage_magnitude_violations = np.abs(voltage_magnitude_violations)
    
    case_name = 'case' + str(case['bus'].shape[0])
    non_gen_bus_idx = np.where(getattr(pp, case_name)()['bus'][:, 1] == 1)[0]

    voltage_magnitude_violation = np.max(voltage_magnitude_violations[non_gen_bus_idx])
    violations['Voltage Magnitude'] = voltage_magnitude_violation
    violations_bin['Voltage Magnitude'] =  int(violations['Voltage Magnitude']>0)
    distances['Voltage Magnitude'] = np.min(np.minimum(np.array(case['bus'][:, 11]) - np.array(predicted_voltages_mag), np.array(np.array(predicted_voltages_mag - case['bus'][:, 12]))))
    
    if args.outage_type == 'gen_out':
        num_generators = num_generators - 1 
    
    # Calculate generator real power output violations
    generator_real_min = np.array(case['gen'][:num_generators, 9])
    generator_real_max = np.array(case['gen'][:num_generators, 8])
    generator_real_output = active_generations[:num_generators]
    
    generator_real_violation = np.sum(np.maximum(generator_real_output - generator_real_max, 0)) + \
                               np.sum(np.maximum(generator_real_min - generator_real_output, 0))
    violations['Generator Real Power Output Limits'] = generator_real_violation
    violations_bin['Generator Real Power Output Limits'] =  int(violations['Generator Real Power Output Limits']>0)
    distances['Generator Real Power Output Limits'] = np.min(np.minimum((generator_real_max - generator_real_output),(generator_real_output - generator_real_min)))
    
    # Calculate generator reactive power output violations
    generator_reactive_min = np.array(case['gen'][:num_generators, 4])
    generator_reactive_max = np.array(case['gen'][:num_generators, 3])
    generator_reactive_output = reactive_generations 
    
    generator_reactive_violation = np.sum(np.maximum(generator_reactive_output - generator_reactive_max, 0)) + \
                                   np.sum(np.maximum(generator_reactive_min - generator_reactive_output, 0))
    violations['Generator Reactive Power Output Limits'] = generator_reactive_violation
    violations_bin['Generator Reactive Power Output Limits'] =  int(violations['Generator Reactive Power Output Limits']>0)
    distances['Generator Reactive Power Output Limits'] = np.min(np.minimum((generator_reactive_max - generator_reactive_output),(generator_reactive_output - generator_reactive_min)))
    
    # Calculate voltage angle difference violations
    voltage_angle_diff = np.abs(predicted_voltages_angle[line_info[:, 0].astype(int)] - predicted_voltages_angle[line_info[:, 1].astype(int)])
    
    voltage_angle_min = np.array(case['branch'][:, 11])
    voltage_angle_max = np.array(case['branch'][:, 12])
    
    # Compute violations based on min and max values
    voltage_angle_diff_violation_upper = np.sum(np.maximum(voltage_angle_diff - voltage_angle_max, 0))
    voltage_angle_diff_violation_lower = np.sum(np.maximum(voltage_angle_min - voltage_angle_diff, 0))
    
    # Total violation can be the sum of the upper and lower violations if you want to consider both
    total_violation = voltage_angle_diff_violation_upper + voltage_angle_diff_violation_lower
    violations['Voltage Angle Difference'] = total_violation
    violations_bin['Voltage Angle Difference'] =  int(violations['Voltage Angle Difference']>0)
    distances['Voltage Angle Difference'] = np.min(np.minimum((voltage_angle_max - voltage_angle_diff),(voltage_angle_diff - voltage_angle_min)))
    
    
    # Calculate branch flow violations
    angles_rad = np.radians(predicted_voltages_angle)
    complex_voltages = predicted_voltages_mag * np.exp(1j * angles_rad)
    
    from_bus_voltages = complex_voltages[line_info[:, 0].astype(int)]
    to_bus_voltages = complex_voltages[line_info[:, 1].astype(int)]
    voltage_differences = from_bus_voltages - to_bus_voltages
    
    line_impedances = case['branch'][:, 2] + 1j * case['branch'][:, 3]  # R + jX
    line_currents = voltage_differences / line_impedances
    line_currents_magnitude = np.abs(line_currents)

    # Base power of the system in MVA
    baseMVA = 100

    # Calculate apparent power flow on each line in per unit
    # Using sending end bus voltages for calculation
    apparent_power_flow_pu = from_bus_voltages * np.conj(line_currents)

    # Convert apparent power from per unit to MVA
    apparent_power_flow_mva = apparent_power_flow_pu * baseMVA

    # Magnitude of apparent power flow in MVA
    apparent_power_flow_mva_magnitude = np.abs(apparent_power_flow_mva)
    
    # Convert the MVA rating (RATE_A) to p.u.
    line_capacity = case['branch'][:, 5]
    
    # Calculate branch flow violation
    branch_flow_violation = np.sum(np.maximum(apparent_power_flow_mva_magnitude - line_capacity, 0))
    violations['Branch Flow'] = branch_flow_violation
    violations_bin['Branch Flow'] =  int(violations['Branch Flow']>0)
    distances['Branch Flow'] = np.min(line_capacity - apparent_power_flow_mva_magnitude)
    
    
    return violations, violations_bin, imbalance, distances


def compute_generation_cost(particle, imbalance, gen_cost, num_generators):
    """
    Computes the total generation cost considering the generation settings provided by the PSO position.
    The cost is calculated based on polynomial coefficients stored in gen_cost for each generator.

    Args:
        particle (list): Particle position in PSO, representing generation settings (both real and reactive powers).
        imbalance (float): The calculated power imbalance in the system.
        gen_cost (np.array): Cost coefficients for each generator. Each row corresponds to a generator and may include
                             polynomial coefficients for cost calculation.
        num_generators (int): Number of traditional generators in the system.

    Returns:
        float: The total generation cost adjusted for any imbalances.
    """
    active_generations = np.array(particle[::2])

    req_active_generations = active_generations * (np.sum(active_generations) + imbalance) / np.sum(active_generations)
    
    order = gen_cost.shape[1] - 5
    
    if args.outage_type == 'gen_out':
        num_generators -= 1
    
    generation_cost = 0
    req_generation_cost = 0
    
    if len(gen_cost == num_generators): # If only active generations incur cost
        for i in np.arange(order, -1, -1):
            generation_cost += np.sum(gen_cost[:num_generators, -i-1] * active_generations**i) 
            
        for i in np.arange(order, -1, -1):
            req_generation_cost += np.sum(gen_cost[:num_generators, -i-1] * req_active_generations**i) 
            
        if imbalance > 0:
            generation_cost += np.abs(generation_cost - req_generation_cost) * 2
        else:
            generation_cost += np.abs(generation_cost - req_generation_cost)
            
    else:
        print('Ckeck gencost!')
  
    return generation_cost


def evaluate_particle(particle, violation_costs, HF_model, LF_model, MF_model, active_demands, reactive_demands, case, active_demand_corr, reactive_demand_corr, renewable_corr, num_samples, num_generators, time_output, analysis, epsilon, last_run=False):
    """
    Evaluates the particle's position by performing power flow analysis, checking operational constraints, and
    computing generation costs. Used to guide the search in PSO optimization.

    Args:
        particle (list): Particle position in PSO, representing generation settings.
        violation_costs (np.array): Costs associated with constraint violations.
        HF_model, LF_model, MF_model (torch.nn.Module): Models used for high, low, and medium fidelity simulations.
        active_demands, reactive_demands (np.array): Demands at each bus.
        case (dict): PYPOWER case dictionary containing the grid data.
        active_demand_corr, reactive_demand_corr, renewable_corr (np.array): Correlation matrices for demand variations.
        num_samples (int): Number of samples to average over for stochastic analysis.
        num_generators (int): Number of traditional generators in the system.
        time_output (str): File path for logging time elapsed during evaluations.
        analysis (str): Type of analysis ('surrogate', 'pp', etc.) to determine the approach for evaluation.
        epsilon (float): Threshold for constraint violation probability.
        last_run (bool): Indicates if this is the final evaluation after optimization convergence.

    Returns:
        float: Total cost considering generation and violations.
        dict: Detailed information about violations and other evaluation metrics.
    """
    # For final deterministic objective function calculation without penalty at the end on analysis
    if last_run:
        total_violation_cost = 0
        
        predicted_voltages_mag, predicted_voltages_angle = perform_power_flow_analysis_pp(case, active_demands, reactive_demands, particle, num_generators)
        # Check OPF constraints and calculate violations
        violations, violations_bin, imbalance, distances = check_opf_constraints(predicted_voltages_mag, predicted_voltages_angle, active_demands, reactive_demands, particle, case, num_generators)
        
        violations = list(violations.values())
        
        cost = compute_generation_cost(particle, imbalance, case['gencost'], num_generators)
        
        all_violations = None
        violated = None
       
    # For score stochastic calculation (objective function + penalty) throughout the optimization 
    else:
        total_costs = 0
        
        all_violations_bin = []
        all_violations = []
        all_imbalance = []
        all_costs = []
        all_distances = []
        violated = []
        
        scenarios = {'real_demand_samples': [], 'reactive_demand_samples': [], 'renewable_gen_samples': []}
        
        for k in range(num_samples):
            
            active_bus_perturbation = generate_correlated_samples(active_demand_corr, size=1, mean=1, cov=0.1).flatten()
            active_demands = active_demands * active_bus_perturbation  # Scale loads
            active_demands = np.clip(active_demands, 0, np.inf)
            
            reactive_bus_perturbation = generate_correlated_samples(reactive_demand_corr, size=1, mean=1, cov=0.1).flatten()
            reactive_demands = reactive_demands * reactive_bus_perturbation
            
            scenarios['real_demand_samples'].append(active_demands)
            scenarios['reactive_demand_samples'].append(reactive_demands) 
            
            if args.uncertainty_level == 'renewable':
                renewable_gen_perturbation = generate_correlated_samples(renewable_corr, size=1, mean=1, cov=0.1).flatten()

                if args.outage_type == 'gen_out':
                    case['gen'][num_generators-1:, 1] *= renewable_gen_perturbation
                    scenarios['renewable_gen_samples'].append(case['gen'][num_generators-1:, 1])
                else:    
                    case['gen'][num_generators:, 1] *= renewable_gen_perturbation
                    scenarios['renewable_gen_samples'].append(case['gen'][num_generators:, 1])
                    
            time_in = time.time()
            if analysis == 'surrogate':
                predicted_voltages_mag, predicted_voltages_angle = perform_power_flow_analysis(HF_model, LF_model, MF_model, active_demands, reactive_demands, particle, case, num_generators)
            else:
                predicted_voltages_mag, predicted_voltages_angle = perform_power_flow_analysis_pp(case, active_demands, reactive_demands, particle, num_generators)
            
            with open(time_output, 'a') as file:
                file.write(f"Elapsed: {time.time() - time_in}\n")

            
            # Check OPF constraints and calculate violations
            violations, violations_bin, imbalance, distances = check_opf_constraints(predicted_voltages_mag, predicted_voltages_angle, active_demands, reactive_demands, particle, case, num_generators)
            
            all_violations_bin.append(list(violations_bin.values()))
            all_violations.append(list(violations.values()))
            all_imbalance.append(imbalance)
            all_distances.append(list(distances.values()))
            
            total_generation_cost = compute_generation_cost(particle, imbalance, case['gencost'], num_generators)
            all_costs.append(total_generation_cost)
            
        all_violations_bin = np.array(all_violations_bin)
        all_violations = np.array(all_violations)
        all_distances = np.array(all_distances)
        
        violated_prob = np.sum(all_violations_bin, axis=0) / num_samples
        violated = (violated_prob > epsilon).astype(int)
        
        #Hybrid approach for fixing the constraint validation errors
        distances_threshold = np.array([30, 0.03, 20, 20, 20, 10])
        
        violated_prob_proximity = violated_prob - epsilon
        
        constraints_for_hybrid = np.where(np.abs(violated_prob_proximity) <= 0.035)[0]

        if constraints_for_hybrid.shape[0] > 0 and analysis == 'surrogate':
            for idx in constraints_for_hybrid:
                active_distances = np.abs(all_distances[:, idx])
                active_threshold = distances_threshold[idx]
                hybrid_samples = np.where(active_distances < active_threshold)[0]
                
                for k in hybrid_samples:
                    active_demands = scenarios['real_demand_samples'][k]
                    
                    reactive_demands = scenarios['reactive_demand_samples'][k]
                    
                    if args.uncertainty_level == 'renewable':
                        if args.outage_type == 'gen_out':
                            case['gen'][num_generators-1:, 1] = scenarios['renewable_gen_samples'][k]
                        else:    
                            case['gen'][num_generators:, 1] = scenarios['renewable_gen_samples'][k]
                        
                    predicted_voltages_mag, predicted_voltages_angle = perform_power_flow_analysis_pp(case, active_demands, reactive_demands, particle, num_generators)
                    
                    violations, violations_bin, imbalance, _ = check_opf_constraints(predicted_voltages_mag, predicted_voltages_angle, active_demands, reactive_demands, particle, case, num_generators)
                    
                    total_generation_cost = compute_generation_cost(particle, imbalance, case['gencost'], num_generators)
                    
                    all_violations_bin[k, :] = np.array(list(violations_bin.values()))
                    all_costs[k] = total_generation_cost
            
            violated_prob = np.sum(all_violations_bin, axis=0) / num_samples
            violated = (violated_prob > epsilon).astype(int)

        mean_cost = np.mean(np.array(all_costs))
        penalty = np.dot(violation_costs, violated)
        
        cost = mean_cost + penalty

    return cost, violations

def generate_particle(num_generators, min_real_output, max_real_output, min_reactive_output, max_reactive_output, real_demand, reactive_demand, case):
    
    """
    Generates initial real and reactive power outputs for each generator to form a particle in PSO. This includes handling
    any initial conditions or constraints such as outages.

    Args:
        num_generators (int): Number of generators in the system.
        min_real_output (float): Minimum real power output limit.
        max_real_output (float): Maximum real power output limit.
        min_reactive_output (float): Minimum reactive power output limit.
        max_reactive_output (float): Maximum reactive power output limit.
        real_demand (float): Total system real power demand.
        reactive_demand (float): Total system reactive power demand.
        case (dict): The power system case dictionary which includes all bus and generator data.
        outage_type (str): Type of outage to simulate, affecting generator count.

    Returns:
        list: A list representing a particle in PSO, containing real and reactive power values for each generator.
    """
    
    # Generate random real and reactive generation values for each generator
    particle = []
    ppopt = pp.ppoption(VERBOSE=0, OUT_ALL=0)
    results = pp.rundcopf(case, ppopt=ppopt)
    
    if args.outage_type == 'gen_out':
        num_generators = num_generators - 1 
    
    for i in range(num_generators):
        real_output = np.clip(results['gen'][i, 1] * np.random.normal(1, 0.1), 0, np.inf)
        reactive_output = results['gen'][i, 2] + np.random.uniform(-100, 100)
        
        particle.append(real_output)
        particle.append(reactive_output)

    # Calculate the current sum of real and reactive generations
    current_real_sum = np.sum(particle[::2])  # Sum of even-indexed elements (real values)
    current_reactive_sum = np.sum(particle[1::2])  # Sum of odd-indexed elements (reactive values)

    # Calculate scaling factors to fit within the specified generation range
    real_scaling_factor = real_demand / current_real_sum * np.random.uniform(0.9, 1.1)
    reactive_scaling_factor = reactive_demand / current_reactive_sum * np.random.uniform(1, 1.2)

    # Repair the generation values to fit within the generation_range
    k = 0
    for i in range(0, len(particle), 2):
        generator_real_min = np.array(case['gen'][k, 9])
        generator_real_max = np.array(case['gen'][k, 8])
        
        generator_reactive_min = np.array(case['gen'][k, 4])
        generator_reactive_max = np.array(case['gen'][k, 3])
        
        particle[i] *= real_scaling_factor
        particle[i] = np.clip(particle[i], generator_real_min, generator_real_max)
        
        particle[i + 1] *= reactive_scaling_factor
        particle[i + 1] = np.clip(particle[i + 1], generator_reactive_min, generator_reactive_max)
        
        k += 1
        
    return particle

def initialize_particles(population_size, num_generators, min_real_output, max_real_output, min_reactive_output, max_reactive_output, real_demand, reactive_demand, case):
    """
    Initializes a population of particles for the PSO algorithm. Each particle is initialized with a position,
    a velocity, and a personal best position and score.

    Args:
        population_size (int): The number of particles in the swarm.
        num_generators, min_real_output, max_real_output,
        min_reactive_output, max_reactive_output (float): Parameters defining the limits for generator outputs.
        real_demand, reactive_demand (float): Total system demand for real and reactive power.
        case (dict): The power system case dictionary.

    Returns:
        list: A list of dictionaries, each representing a particle with position, velocity, best position, and score.
    """
    particles = []
    velocities = []
    for _ in range(population_size):
        particle = generate_particle(num_generators, min_real_output, max_real_output, min_reactive_output, max_reactive_output, real_demand, reactive_demand, case)
        velocity = [random.uniform(-10, 10) for _ in range(len(particle))] # Initial velocities
        particles.append({'position': particle, 'velocity': velocity, 'best_position': list(particle), 'best_score': float('inf')})
    return particles

def update_velocity(particle, global_best_position, w=0.5, c1=2, c2=1.5):
    """
    Updates the velocity of a particle based on its current velocity, the cognitive component (personal best),
    and the social component (global best).

    Args:
        particle (dict): The particle whose velocity is to be updated.
        global_best_position (list): The best position found by any particle in the swarm.
        w (float): Inertia weight factor.
        c1, c2 (float): Cognitive and social factors respectively.

    Returns:
        list: The updated velocity vector for the particle.
    """
    inertia = [w * v for v in particle['velocity']]
    cognitive = [c1 * random.random() * (pbest - p) for p, pbest in zip(particle['position'], particle['best_position'])]
    social = [c2 * random.random() * (gbest - p) for p, gbest in zip(particle['position'], global_best_position)]
    
    new_velocity = [i + c + s for i, c, s in zip(inertia, cognitive, social)]
    return new_velocity

def update_position(particle, case):
    """
    Updates the position of a particle based on its current position and velocity. Ensures the new position
    does not exceed defined generator output limits.

    Args:
        particle (dict): The particle whose position is to be updated.
        case (dict): The power system case dictionary providing generator limits.

    Returns:
        list: The updated position vector for the particle.
    """
    new_position = [p + v for p, v in zip(particle['position'], particle['velocity'])]
    
    k = 0
    for i in range(0, len(new_position), 2):
        generator_real_min = np.array(case['gen'][k, 9])
        generator_real_max = np.array(case['gen'][k, 8])
        
        generator_reactive_min = np.array(case['gen'][k, 4])
        generator_reactive_max = np.array(case['gen'][k, 3])

        new_position[i] = np.clip(new_position[i], generator_real_min, generator_real_max)
        new_position[i + 1] = np.clip(new_position[i + 1], generator_reactive_min, generator_reactive_max)
        
        k += 1
    
    return new_position


def parse_arguments():
    """
    Configures and parses command-line arguments for the PSO optimization process for OPF.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='PSO Optimization for Optimal Power Flow (OPF)')

    parser.add_argument('--case', type=str, default='case118',
                        help='IEEE test case to use (e.g., "case118" for the 118-bus test system)')
    parser.add_argument('--model_dir', type=str, default='../MFGNN/results',
                        help='Directory where the trained models are stored')
    parser.add_argument('--scenario_dir', type=str, default='scenarios',
                        help='Directory where different scenarios are stored')
    parser.add_argument('--analysis_type', type=str, default='normal',
                        help='Type of analysis to perform ("normal" for SF or MF or "dc_approx" for MFDC)')
    parser.add_argument('--model_type', type=str, default='MF',
                        help='Type of model used in the analysis ("SF" for single-fidelity, "MF" for multi-fidelity)')
    parser.add_argument('--uncertainty_level', type=str, default='renewable',
                        help='Level of uncertainty considered in the analysis ("demand" or "renewable")')
    parser.add_argument('--outage_type', type=str, default='none',
                        help='Type of outage considered in the analysis ("none", "line_out", "gen_out")')
    parser.add_argument('--output_dir', type=str, default='output/hyb',
                        help='Directory to save output files')
    parser.add_argument('--population_size', type=int, default=20,
                        help='Number of particles in the PSO population')
    parser.add_argument('--iterations', type=int, default=40,
                        help='Maximum number of iterations for the PSO algorithm')
    parser.add_argument('--num_ext_samples', type=int, default=100,
                        help='Number of external trials to evaluate for stochastic processes')
    parser.add_argument('--num_int_samples', type=int, default=150,
                        help='Number of internal MC samples to evaluate during optimization')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Threshold for chance-constrained optimizations')
    parser.add_argument('--early_stopping_threshold', type=int, default=5,
                        help='Early stopping threshold for convergence')
    parser.add_argument('--hybrid_steps', type=int, default=5,
                        help='Number of iterations to perform with actual models after surrogate-based HCC-OPF converges. Only actiive of Optimization Mode == hybrid')
    parser.add_argument('--optimization_mode', type=str, default='surrogate',
                        help='Optimization mode ("surrogate" using surrogate with hybrid constraint violation,\
                            "pp" for ACPF based CC-OPF,\
                            or "hybrid" for using surrogate in the beginning with hybrid constraint validation and a few steps of ACPF-based CC-OPF after convergence)')

    return parser.parse_args()


# =============================================================================
# =============================================================================
# Main
# =============================================================================
# =============================================================================

def main(args):
# =============================================================================
#     Required initializations before the analysis
# =============================================================================
    
    # Determine the execution device based on availability of CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Defining device at the start
    
    # Configuration parameters extracted from command line arguments
    epsilon = args.epsilon
    early_stopping_threshold = args.early_stopping_threshold
    hybrid_steps = args.hybrid_steps
    
    # Model path setup based on the type of analysis and outage condition
    # Dynamically construct paths based on the provided arguments
    if args.analysis_type == 'normal':
        if args.outage_type != 'none':
            HF_model_checkpoint = os.path.join(args.model_dir, args.outage_type, args.analysis_type, args.uncertainty_level, args.case, 'HF_best_model.pt')
            MF_model_checkpoint = os.path.join(args.model_dir, args.outage_type, args.analysis_type, args.uncertainty_level, args.case, 'MF_best_model.pt')
            LF_model_checkpoint = os.path.join(args.model_dir, args.outage_type, args.analysis_type, args.uncertainty_level, args.case, 'LF_best_model.pt')
        # Handle outage scnearios differently
        else:
            HF_model_checkpoint = os.path.join(args.model_dir, args.analysis_type, args.uncertainty_level, args.case, 'HF_best_model.pt')
            MF_model_checkpoint = os.path.join(args.model_dir, args.analysis_type, args.uncertainty_level, args.case, 'MF_best_model.pt')
            LF_model_checkpoint = os.path.join(args.model_dir, args.analysis_type, args.uncertainty_level, args.case, 'LF_best_model.pt')
     
        # Loading the pre-trained models
        # Load the Low Fidelity (LF) model checkpoint and initialize the model
        LF_checkpoint = torch.load(LF_model_checkpoint, map_location=torch.device(device))
        # Define the architecture parameters for the LF model
        input_dim = 5
        hidden_dim = 256
        output_dim = 2
        # Initialize the LF model with specific architecture settings
        LF_model = LFGNN(input_dim, hidden_dim, output_dim - 1, 'TAGConv', 2)
        # Load the trained state into the LF model
        LF_model.load_state_dict(LF_checkpoint)
        # Ensure the LF model is assigned to the correct computation device
        LF_model = LF_model.to(device)
        
        # Load the High Fidelity (HF) model checkpoint and initialize the model
        HF_checkpoint = torch.load(HF_model_checkpoint, map_location=torch.device(device))
        # Define the architecture parameters for the HF model
        input_dim = 5
        hidden_dim = 64
        output_dim = 2
        # Initialize the HF model with specific architecture settings
        HF_model = HFGNN(input_dim, hidden_dim, output_dim)
        # Load the trained state into the HF model
        HF_model.load_state_dict(HF_checkpoint)
        # Ensure the HF model is assigned to the correct computation device
        HF_model = HF_model.to(device)
        
        # Load the Medium Fidelity (MF) model checkpoint and initialize the model
        MF_checkpoint = torch.load(MF_model_checkpoint, map_location=torch.device(device))
        # Define the architecture parameters for the MF model, using augmented input dimensions
        input_dim = 5
        hidden_dim = 64
        output_dim = 2
        # Initialize the MF model with additional output dimensions considered in input
        MF_model = MFGNN(input_dim + output_dim, hidden_dim, output_dim, 256)
        # Load the trained state into the MF model
        MF_model.load_state_dict(MF_checkpoint)
        # Ensure the MF model is assigned to the correct computation device
        MF_model = MF_model.to(device)
        
        
    # DC_Approx        
    # Handling the MFDC model
    else:
        # If there's an outage, specify the model checkpoint path considering the outage type
        if args.outage_type != 'none':
            # Set High Fidelity and Low Fidelity model checkpoints to None since they are not used in DC approximation
            HF_model_checkpoint = None
            LF_model_checkpoint = None
            # Set the path for the Multi Fidelity DC model considering the specific outage type
            MF_model_checkpoint = os.path.join(args.model_dir, args.outage_type, args.analysis_type, args.uncertainty_level, args.case, 'MFDC_best_model.pt')
        else:
            # Set High Fidelity and Low Fidelity model checkpoints to None as these models are not required for DC approximation
            HF_model_checkpoint = None
            LF_model_checkpoint = None
            # Set the path for the  Multi Fidelity DC model without considering an outage
            MF_model_checkpoint = os.path.join(args.model_dir, args.analysis_type, args.uncertainty_level, args.case, 'MFDC_best_model.pt')
            
        # Set High Fidelity and Low Fidelity models to None since DC approximation does not utilize them
        HF_model = None
        LF_model = None
        
        # Load the Multi Fidelity DC model checkpoint
        MF_checkpoint = torch.load(MF_model_checkpoint, map_location=torch.device(device))
        # Define architecture parameters specific to the DC Multi Fidelity DC model
        input_dim = 7
        hidden_dim = 64
        output_dim = 2
        
        # Initialize the Medium Fidelity DC model with specified dimensions
        MF_model = MFGNN_DC(input_dim, hidden_dim, output_dim)
        # Load the trained state into the Medium Fidelity DC model
        MF_model.load_state_dict(MF_checkpoint)
        # Ensure the MF model is assigned to the correct computation device
        MF_model = MF_model.to(device)
        
    # Configuration of output filenames based on the optimization mode and analysis type
    if args.optimization_mode != 'pp':
        # Generate filenames for results, logs, and metrics with a specific suffix to distinguish them
        if args.analysis_type == 'normal': # Handling SF-GNN and EMF-GNN based HCC-OPF
            output_name = args.optimization_mode + args.model_type + '_resutls.txt'
            time_output_name = args.optimization_mode + args.model_type + '_time.txt'
            violation_output_name = args.optimization_mode + args.model_type + '_violations.txt'
            convergence_output_name = args.optimization_mode + args.model_type + '_convergence.txt'
            optimization_time_output_name = args.optimization_mode + args.model_type + '_optimization_time.txt'
        else: # Handling MFDC-GNN based HCC-OPF
            output_name = args.optimization_mode + 'MFDC_resutls.txt'
            time_output_name = args.optimization_mode + 'MFDC_time.txt'
            violation_output_name = args.optimization_mode + 'MFDC_violations.txt'
            convergence_output_name = args.optimization_mode + 'MFDC_convergence.txt'
            optimization_time_output_name = args.optimization_mode + 'MFDC_optimization_time.txt'
    else: #Handling ACPF-based CC-OPF
        output_name = args.optimization_mode + '_resutls.txt'
        time_output_name = args.optimization_mode + '_time.txt'
        violation_output_name = args.optimization_mode + '_violations.txt'
        convergence_output_name = args.optimization_mode + '_convergence.txt'
        optimization_time_output_name = args.optimization_mode + '_optimization_time.txt'
    
    #Setup directories based on the presence of an outage type
    if args.outage_type != 'none':
        # Append outage type to directory path for better segregation of results
        base_dir = os.path.join(args.output_dir, args.outage_type, args.uncertainty_level, args.case)
    else:
        # Use a general directory path when no outage is specified
        base_dir = os.path.join(args.output_dir, args.uncertainty_level, args.case)
    
    # Combine base directory path with filenames to create full paths for outputs
    output_dir = os.path.join(base_dir, output_name)
    time_output = os.path.join(base_dir, time_output_name)
    violations_dir = os.path.join(base_dir, violation_output_name)
    convergence_dir = os.path.join(base_dir, convergence_output_name)
    optimization_time_dir = os.path.join(base_dir, optimization_time_output_name)
    
    # Ensure the existence of directories for output files
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(time_output), exist_ok=True)
    os.makedirs(os.path.dirname(optimization_time_dir), exist_ok=True)
    os.makedirs(os.path.dirname(violations_dir), exist_ok=True)
    os.makedirs(os.path.dirname(convergence_dir), exist_ok=True)
    
    # Retrieve number of samples for internal and external evaluations from arguments
    num_int_samples = args.num_int_samples
    num_ext_samples = args.num_ext_samples
    
    # Retrieve population size and number of iterations for PSO from arguments
    population_size = args.population_size
    iterations = args.iterations
    
    # Determine the appropriate scenario directory based on whether there is an outage
    if args.outage_type == 'none':
        scenario_dir = os.path.join(args.scenario_dir, f'{args.case}_scenarios_corr.pkl')
    else:
        scenario_dir = os.path.join(args.scenario_dir, f'{args.case}_{args.outage_type}_scenarios_corr.pkl')
        
    # Load scenarios from the specified pickle file
    with open(scenario_dir, 'rb') as f:
        scenarios = pickle.load(f)
    
    # Open files to log various outputs and results
    with open(output_dir, 'a') as file, open(violations_dir, 'a') as violations_file, \
         open(convergence_dir, 'a') as convergence_file, open(optimization_time_dir, 'a') as optimization_time_file:
        # Log the start of analysis
        current_time = datetime.datetime.now()
        file.write(f"\n********************************* Analysis Started at {current_time} ********************************* \n")
        file.flush()
        
        violations_file.write(f"\n********************************* Analysis Started at {current_time} ********************************* \n")
        violations_file.flush()
        
        convergence_file.write(f"\n********************************* Analysis Started at {current_time} ********************************* \n")
        convergence_file.flush()
        
        optimization_time_file.write(f"\n********************************* Analysis Started at {current_time} ********************************* \n")
        optimization_time_file.flush()

        # Initialize lists to store results
        best_solutions = []
        best_positions = []
        final_violations = []
        
        # Start time for the optimization process
        opt_time_in = time.time()
        
        # Iterate over all scenarios for optimization
        for j in range(num_ext_samples):
            print("Sample ", j + 1)
            timeIn = time.time()
            
            # Load and deepcopy the original case for safe manipulation
            original_case = getattr(pp, args.case)()
            case = copy.deepcopy(original_case)  
                
            # Retrieve the scenario for the current sample
        case_scenario = scenarios[j]
        
        # Compute shortest path distances within the grid network
        distances = compute_shortest_path_distances(case)
        
        # Generate correlation matrices based on computed distances with specified decay factors
        active_corr_matrix = create_correlation_matrix(distances, decay=0.2)
        reactive_corr_matrix = create_correlation_matrix(distances, decay=0.5)

        # Get the number of generators and their output limits from the case
        num_generators = len(case['gen']) 
        min_real_output, max_real_output = 0, 500
        min_reactive_output, max_reactive_output = -100, 100

        # Define costs associated with various types of constraint violations
        violation_costs = np.array([1000.0] * 6)  # Assumed cost for each type of violation

        # Update the case with demand levels from the scenario
        active_demands = case_scenario['base_case']["bus"][:, 2]
        reactive_demands = case_scenario['base_case']["bus"][:, 3]
        case["bus"][:, 2] = active_demands
        case["bus"][:, 3] = reactive_demands

        # Sum total demands for setting up the optimization
        real_demand = np.sum(active_demands)
        reactive_demand = np.sum(reactive_demands)
        
        # Handling renewable uncertainty by updating generator settings and costs if specified
        if args.uncertainty_level == 'renewable':
            case['gen'] = case_scenario['base_case']['gen']
            case['gencost'] = case_scenario['base_case']['gencost']
            case["bus"][:, 1] = case_scenario['base_case']["bus"][:, 1]
            
            # Identify buses with newly added generators under renewable scenarios
            num_original_gens = original_case['gen'].shape[0]
            selected_buses = case_scenario['base_case']['gen'][num_original_gens:, 0].astype(int)
            
            # Compute distances specifically for the selected buses and create a corresponding correlation matrix
            selected_distances = distances[np.ix_([bus - 1 for bus in selected_buses], [bus - 1 for bus in selected_buses])]
            renewable_corr_matrix = create_correlation_matrix(selected_distances, decay=0.2)
            
# =============================================================================
#   PSO Optimization Logic
# =============================================================================
        # Initialize PSO particles with specified parameters and initial random velocities
        particles = initialize_particles(population_size, num_generators, min_real_output, max_real_output, min_reactive_output, max_reactive_output, real_demand, reactive_demand, case)
        scores = [float('inf') for _ in particles]  # Initialize scores for each particle with a high value
        
        # Determine the analysis mode based on the optimization configuration
        if args.optimization_mode != 'pp':
            analysis = 'surrogate'  # Use surrogate model evaluations
        else:
            analysis = 'pp'  # Use PyPower ACPF
                        
            # Initial evaluation of all particles
        for i, particle in enumerate(particles):
            score, _ = evaluate_chromosome(particle['position'], violation_costs, HF_model, LF_model, MF_model, active_demands, reactive_demands, case, active_corr_matrix, reactive_corr_matrix, renewable_corr_matrix, num_int_samples, num_generators, time_output, analysis, epsilon)
            scores[i] = score
            # Update particle's best known position if the new score is better
            if score < particle['best_score']:
                particle['best_position'] = particle['position']
                particle['best_score'] = score
            
            # Identify the global best position and score from initial evaluations
            global_best_position = particles[np.argmin(scores)]['position']
            global_best_score = min(scores)
            
            # Main PSO optimization loop
            convergences = []
            stop_counter = 0
            iteration = 0
            last_scores = [global_best_score]
            
            # Loop until the maximum number of iterations is reached or early stopping criteria is met
            while iteration < iterations:
                iter_time_in = time.time() # Record the start time of the iteration
                # Check for early stopping if there have been enough iterations to assess convergence
                if iteration > early_stopping_threshold:
                    recent_changes = [abs(last_scores[i] - last_scores[i - 1]) / max(last_scores[i], 1e-6) for i in range(1, 1+early_stopping_threshold)]
                    if all(change < 0.005 for change in recent_changes):
                        print(f"Stopping criteria met after {iteration} iterations.")
                        break
                # Evaluate and update each particle's score and best position    
                for i, particle in enumerate(particles):
                    score, _ = evaluate_chromosome(particle['position'], violation_costs, HF_model, LF_model, MF_model, active_demands, reactive_demands, case, active_corr_matrix, reactive_corr_matrix, renewable_corr_matrix, num_int_samples, num_generators, time_output, analysis, epsilon)
                    scores[i] = score
                    if score < particle['best_score']:
                        particle['best_position'] = particle['position']
                        particle['best_score'] = score
                    # Update global best if a new best is found
                    if score < global_best_score:
                        global_best_position = particle['position']
                        global_best_score = score
                
                # Log iteration results and update global best score convergence tracking
                iter_score, _ = evaluate_chromosome(global_best_position, violation_costs, HF_model, LF_model, MF_model, active_demands, reactive_demands, case, active_corr_matrix, reactive_corr_matrix, renewable_corr_matrix, num_int_samples, num_generators, time_output, analysis, epsilon, last_run=True)
                
                if len(convergences) > 0 and iter_score == convergences[-1]:
                    stop_counter += 1
                else:
                    stop_counter = 0
                
                convergences.append(iter_score)
                
                for particle in particles:
                    particle['velocity'] = update_velocity(particle, global_best_position)
                    particle['position'] = update_position(particle, case)
                
                last_scores.append(global_best_score)
                if len(last_scores) > 10:
                    last_scores.pop(0)
                
                iteration += 1
                print(f"Iteration {iteration} - Best Score: {global_best_score}, Best Obj. Func.: {iter_score}")
                
                optimization_time_file.write(f"Iter time: {time.time() - iter_time_in} \n")
                optimization_time_file.flush()
            
            # Handle secondary-hybrid optimization steps if specified
            # Hybrid steps with actual model
            if args.optimization_mode == 'hybrid': 
                analysis = 'pp'
                for hybrid_iteration in range(hybrid_steps):
                    for i, particle in enumerate(particles):
                        score, _ = evaluate_chromosome(particle['position'], violation_costs, HF_model, LF_model, MF_model, active_demands, reactive_demands, case, active_corr_matrix, reactive_corr_matrix, renewable_corr_matrix, num_int_samples, num_generators, time_output, analysis, epsilon)
                        scores[i] = score
                        if score < particle['best_score']:
                            particle['best_position'] = particle['position']
                            particle['best_score'] = score
                        if score < global_best_score:
                            global_best_position = particle['position']
                            global_best_score = score
                            
                    iter_score, _ = evaluate_chromosome(global_best_position, violation_costs, HF_model, LF_model, MF_model, active_demands, reactive_demands, case, active_corr_matrix, reactive_corr_matrix, renewable_corr_matrix, num_int_samples, num_generators, time_output, analysis, epsilon, last_run=True)
                    convergences.append(iter_score)
                    
                    for particle in particles:
                        particle['velocity'] = update_velocity(particle, global_best_position)
                        particle['position'] = update_position(particle, case)
                    
                    print(f"Hybrid Iteration {hybrid_iteration + 1} - Best Score: {global_best_score}, Best Obj. Func.: {iter_score}")
                    
            # Final evaluation of the best solution found in the PSO process
            global_best_score, final_violation = evaluate_chromosome(
                global_best_position, violation_costs, HF_model, LF_model, MF_model, 
                active_demands, reactive_demands, case, active_corr_matrix, 
                reactive_corr_matrix, renewable_corr_matrix, num_int_samples, 
                num_generators, time_output, analysis, epsilon, last_run=True
            )
            
            # Store the results of the final evaluation for analysis and record-keeping
            best_solutions.append(global_best_score)
            best_positions.append(global_best_position)
            final_violations.append(final_violation)
            
            # Output detailed results to the console for immediate feedback
            print(f"Sample {j + 1} - Best Score: {global_best_score}")
            print(f"Sample {j + 1} - Final Violation: {final_violation}")
            print(f"Sample {j + 1} - Final Position: {global_best_position}")
            print(f"Time elapsed: {time.time() - timeIn} \n")  # Log the time taken for processing this sample
            
            # Write results to the corresponding files for permanent record-keeping
            file.write(f"Optimal cost: {global_best_score} ")
            file.write(f"Time elapsed: {time.time() - timeIn} ")
            file.write(f"Best solution: {global_best_position} \n")
            file.flush()  # Ensure all data is written to the file system
            
            # Record any violations that occurred during the final evaluation
            violations_file.write(f"Violations: {final_violation} \n")
            violations_file.flush()  # Ensure all data is written to the file system
            
            # Log the convergence data to monitor the optimization process's progress
            convergence_file.write(f"Convergence: {convergences} \n")
            convergence_file.flush()  # Ensure all data is written to the file system
            
            # Document the time taken for the entire optimization process
            optimization_time_file.write("-------------------------------------------------------------- \n")
            optimization_time_file.write(f"Optimization time: {time.time() - opt_time_in} \n")
            optimization_time_file.write("-------------------------------------------------------------- \n \n")
            optimization_time_file.flush()  # Ensure all data is written to the file system
            
if __name__ == '__main__':
    args = parse_arguments()
    main(args)