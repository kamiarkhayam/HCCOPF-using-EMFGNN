# -*- coding: utf-8 -*-
"""
This script facilitates the generation of trial samples for HCC OPF under uncertainty using the Particle Swarm Optimization approach. 
It integrates uncertainty in load and renewable generation, allowing for simulation under different outage scenarios (line or generator outages). 
The script utilizes utilities for modifying network configurations, calculating shortest path distances, and generating correlated uncertainty samples. 

The script is structured to handle a variety of IEEE test cases, allowing for extensive customizability in terms of uncertainty levels and the number of scenarios generated. 
It uses libraries like numpy for numerical operations, pypower for power flow calculations, and scipy for statistical tasks, ensuring comprehensive and efficient power system analysis.

Created by: Kamiar Khayambashi

"""

import argparse
import random
import numpy as np
import torch
import pypower.api as pp
import scipy.stats
import os
import copy
import pickle
from utils import *



def modify_branch_capacity(case):
    """
    Modifies the transmission capacity of branches within the power system
    to simulate various loading and congestion scenarios based on 
    "An Optimal Power Flow Algorithm to Achieve Robust Operation Considering Load and Renewable Generation Uncertainties"

    Args:
        case (dict): Power system case data with branch information.

    Returns:
        dict: Updated case dictionary with modified branch capacities.
    """
    # Specified adjustments based on the system case (e.g., IEEE 14-bus or 118-bus systems)
    if case['bus'].shape[0] == 14:
        specified_capacities = np.array([200, 90, 90, 70, 50, 50, 80, 40, 40, 60, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20])
        case['branch'][:, 5] = specified_capacities
    elif case['bus'].shape[0] == 118:
        capacity = np.full(case['branch'].shape[0], 300)
        capacity[[6, 7, 8]] = [550, 500, 550]
        case['branch'][:, 5] = capacity
    
    return case


def parse_arguments():
    """
    Parses command-line arguments for configuring the simulation settings.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='PSO OPF Scenario Generation')
    parser.add_argument('--case', type=str, default='case14', help='IEEE case to work with')
    parser.add_argument('--uncertainty_level', type=str, default='renewable', help='Specify the type of uncertainty: demand or renewable generation.')
    parser.add_argument('--outage_type', type=str, default='none', help='Type of outage to simulate: none, generator, or branch.')
    parser.add_argument('--output_dir', type=str, default='scenarios', help='Directory to save the output files')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of scenario samples to generate')
    parser.add_argument('--num_inner_samples', type=int, default=200, help='Number of inner loop samples for detailed analysis')
    
    return parser.parse_args()

# =============================================================================
# Main
# =============================================================================


def main(args):
    # Initialize the number of scenario samples to generate and the number of internal samples for finer analysis
    num_samples = args.num_samples
    num_inner_samples = args.num_inner_samples
    
    # List to hold all scenario data
    scenarios = []
    
    # Main loop to generate each scenario
    for j in range(num_samples):
        print("Sample ", j + 1)  # Output current sample number
        
        # Load the power system case specified by the user
        original_case = getattr(pp, args.case)()
        # Get the number of generators from the loaded case
        num_generators = len(original_case['gencost'])
        
        # Deep copy the original case to modify without affecting the original data
        case = copy.deepcopy(original_case)
        
        # Compute the shortest path distances between all buses in the system
        distances = compute_shortest_path_distances(case)
        
        # Create correlation matrices for active and reactive power demands
        active_corr_matrix = create_correlation_matrix(distances, decay=0.2)
        reactive_corr_matrix = create_correlation_matrix(distances, decay=0.5)

        # Generate correlated uniform samples to perturb power demands
        active_uniform_samples = generate_correlated_uniform_samples(active_corr_matrix, size=1).flatten()
        reactive_uniform_samples = generate_correlated_uniform_samples(reactive_corr_matrix, size=1).flatten()

        # Apply the perturbations to the actual demands of the buses
        active_demands = case["bus"][:, 2] * active_uniform_samples
        reactive_demands = case["bus"][:, 3] * reactive_uniform_samples
        case["bus"][:, 2] = active_demands
        case["bus"][:, 3] = reactive_demands
        
        # Calculate the total real demand after perturbation
        real_demand = np.sum(active_demands)
        
        # Modify branch capacities based on predefined scenarios or policies
        case = modify_branch_capacity(case)
        
        # Estimate renewable generation as a fraction of the real demand
        renewable_gen = real_demand * np.random.uniform(0.05, 0.15)
        
        # Identify generator and non-generator buses
        gen_buses = set(case['gen'][:, 0].astype(int))
        all_buses = set(range(1, len(case['bus']) + 1))
        non_gen_buses = list(all_buses - gen_buses)
        # Calculate how many non-generator buses can have renewable resources
        num_non_gen = len(non_gen_buses) - 1
        
        # Randomly select a subset of non-generator buses to allocate renewable resources
        selected_buses = np.random.choice(non_gen_buses, int(num_non_gen / 5) + 1, replace=False)
        
        # Compute distances between selected buses to form a sub-correlation matrix
        selected_distances = distances[np.ix_([bus - 1 for bus in selected_buses], [bus - 1 for bus in selected_buses])]
        renewable_corr_matrix = create_correlation_matrix(selected_distances, decay=0.2)
        
        # Generate weights for allocating renewable resources based on the correlation among selected buses
        renewable_allocation_weights = generate_correlated_weights(renewable_corr_matrix, size=1).flatten()
        
        # Allocate renewable generation capacity proportionally based on the calculated weights
        renewable_allocation = renewable_allocation_weights * renewable_gen
        
        
        # Add new generators at selected buses with renewable allocations
        for i, bus in enumerate(selected_buses):
            bus_voltage = case['bus'][bus-1, 7]  # Fetch voltage magnitude at the bus
            # Define new generator parameters including power and voltage constraints
            new_gen = [bus, renewable_allocation[i], 0, 10, -10, bus_voltage, 100, 1, renewable_allocation[i], renewable_allocation[i]] + [0]*11
            case['gen'] = np.vstack([case['gen'], new_gen])  # Add new generator to the system

            # Update the bus type to PV to handle power injections from renewables
            case['bus'][bus-1, 1] = 2

       # Ensure generator cost data is updated to match new generators
        num_gens = len(case['gen'])
        if len(case['gencost']) < num_gens:
            # Calculate how many new gencost rows are needed and create them
            additional_rows = num_gens - len(case['gencost'])
            new_gencost_rows = np.zeros((additional_rows, case['gencost'].shape[1]))
            new_gencost_rows[:, 0] = 2  # Set polynomial cost function type
            new_gencost_rows[:, 3] = 3  # Polynomial degree
            case['gencost'] = np.vstack([case['gencost'], new_gencost_rows])
        
        # Handle outages based on the specified type, either line or generator
        if args.outage_type == 'line_out':
            remove_random_branch(case)
        elif args.outage_type == 'gen_out':
            remove_random_generator(case)
        
        # Check if the system is still connected after potential outages
        if not check_connectivity(case):
            print("Disconnected buses detected. Skipping this case.")
            continue  # Skip to next sample if the system is disconnected
        
        
        # Prepare the base scenario before any perturbations
        base_case = copy.deepcopy(case)

        # Dictionary to store samples related to this base case
        scenario_samples = {
            'base_case': base_case, 
            'real_demand_samples': [], 
            'reactive_demand_samples': [], 
            'renewable_gen_samples': []
        }
        
        # Generate additional inner samples by perturbing demands and renewable outputs
        for k in range(num_inner_samples):
            perturbed_case = copy.deepcopy(base_case)
            
            # Generate perturbations for demands and renewable generation
            active_bus_perturbation = generate_correlated_samples(active_corr_matrix, size=1, mean=1, cov=0.1).flatten()
            reactive_bus_perturbation = generate_correlated_samples(reactive_corr_matrix, size=1, mean=1, cov=0.1).flatten()
            renewable_gen_perturbation = generate_correlated_samples(renewable_corr_matrix, size=1, mean=1, cov=0.1).flatten()

            # Apply perturbations and clip to ensure no negative demands
            perturbed_active_demands = np.clip(active_demands * active_bus_perturbation, 0, np.inf)
            perturbed_reactive_demands = reactive_demands * reactive_bus_perturbation
            
            # Update perturbed demands in the case data
            perturbed_case["bus"][:, 2] = perturbed_active_demands
            perturbed_case["bus"][:, 3] = perturbed_reactive_demands

            # Handle renewable generation perturbations based on outage type
            if args.outage_type == 'gen_out':
                perturbed_case['gen'][num_generators-1:, 1] *= renewable_gen_perturbation
                scenario_samples['renewable_gen_samples'].append(perturbed_case['gen'][num_generators-1:, 1])
            else:
                perturbed_case['gen'][num_generators:, 1] *= renewable_gen_perturbation
                scenario_samples['renewable_gen_samples'].append(perturbed_case['gen'][num_generators:, 1])

            scenario_samples['real_demand_samples'].append(perturbed_active_demands)
            scenario_samples['reactive_demand_samples'].append(perturbed_reactive_demands)
        
        scenarios.append(scenario_samples)  # Append the detailed scenario data to the main list
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define file path for saving scenarios
    if args.outage_type == 'none':
        output_path = os.path.join(args.output_dir, f'{args.case}_scenarios_corr.pkl')
    else:
        output_path = os.path.join(args.output_dir, f'{args.case}_{args.outage_type}_scenarios_corr.pkl')

    # Save all scenarios to a pickle file for later use
    with open(output_path, 'wb') as f:
        pickle.dump(scenarios, f)

    print(f'Scenarios saved to {output_path}')  # Notify user of save location
     
if __name__ == '__main__':
    args = parse_arguments()
    main(args)