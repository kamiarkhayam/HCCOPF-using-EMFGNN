# -*- coding: utf-8 -*-
"""
This script is designed to generate training datasets for surrogate models in power system analysis. It accounts for varying levels of uncertainty in load demand and renewable energy integration, simulating realistic power system behaviors under different conditions, including N-1 contingency scenarios.
The script supports a variety of IEEE test cases and performs both DC and AC power flow analyses. It integrates functionalities for modifying generation capacities, perturbing loads, and incorporating renewable energy sources, thereby creating comprehensive datasets for training advanced machine learning models.

The script uses libraries such as numpy for numerical operations, pypower for executing power flow analyses, and torch_geometric for handling graph-based data structures, ensuring robust and efficient simulation capabilities.

Created by: Kamiar Khayambashi
"""

import numpy as np
import pypower.api as pp
import torch
from torch_geometric.data import Data
import pickle
import argparse
import os
import time
from utils import *

def generate_and_save_training_data(num_samples, save_path, seed, case_name, power_flow_type, uncertainty_level, analysis_type):
    
    """
    Generates training data for power system analysis by modifying generation capacities, perturbing loads,
    and performing power flow calculations. The function saves the processed data for training machine learning models.
    
    Args:
        num_samples (int): Number of samples to generate.
        save_path (str): Directory path to save the generated data.
        seed (int): Seed for random number generation to ensure reproducibility.
        case_name (str): Name of the power system case to be used.
        power_flow_type (str): Type of power flow analysis to perform ('dc' or 'ac').
        uncertainty_level (str): Level of uncertainty to introduce in the data ('low', 'medium', 'high', or 'renewable').
        analysis_type (str): Specifies the type of analysis to be performed.
    
    """
    def modify_generations(case):
        """
        Modifies the generation capacity of power systems by integrating renewable energy sources
        and adjusting existing generation capacities accordingly.
        
        Args:
            case (dict): The power system case dictionary.

        Returns:
            dict: The modified power system case with updated generation capacities.
        """
        # Calculate the total existing generation
        total_gen = np.sum(case['gen'][:, 1])  # Sum of real power output from existing generators

        # Determine additional renewable generation (10% to 20% of total generation)
        renewable_gen = total_gen * 0.1

        # Proportionally reduce the generation of existing generators by the total renewable generation added
        reduction_factor = renewable_gen / total_gen
        case['gen'][:, 1] *= 0.9

        # Identify the buses that are not currently generator buses
        gen_buses = set(case['gen'][:, 0].astype(int))
        all_buses = set(range(1, len(case['bus']) + 1))
        non_gen_buses = list(all_buses - gen_buses)
        num_non_gen = len(non_gen_buses)
        
        # Choose int(num_non_gen) + 1 buses randomly from non-generator buses to convert into renewable generators
        selected_buses = np.random.choice(non_gen_buses, int(num_non_gen / 5) + 1, replace=False)

        # Allocate renewable generation across the selected buses
        renewable_allocation = np.random.dirichlet(np.ones(len(selected_buses))) * renewable_gen

        # Create new generator entries for the selected buses
        for i, bus in enumerate(selected_buses):
            bus_voltage = case['bus'][bus-1, 7]  # Voltage magnitude at the bus
            new_gen = [bus, renewable_allocation[i], 0, 10, -10, bus_voltage, 100, 1, 5, 0] + [0]*15  # New generator parameters
            case['gen'] = np.vstack([case['gen'], new_gen])

            # Change the bus type of the selected bus to 2 (PV bus)
            case['bus'][bus-1, 1] = 2

        # Update 'gencost' to match the new number of generators
        num_gens = len(case['gen'])
        if len(case['gencost']) < num_gens:
            additional_rows = num_gens - len(case['gencost'])
            new_gencost_rows = np.zeros((additional_rows, case['gencost'].shape[1]))
            new_gencost_rows[:, 0] = 2  # Set model to polynomial
            new_gencost_rows[:, 3] = 3  # Set N to 3
            case['gencost'] = np.vstack([case['gencost'], new_gencost_rows])

        return case
    
    def perturb_loads(case, uncertainty_level):
        """
        Perturbs the load values of the power system case based on a specified uncertainty level.
        
        Args:
            case (dict): The power system case dictionary.
            uncertainty_level (str): The level of uncertainty to introduce in load values.
        """
        num_buses = len(case["bus"])
        bus_perturbation_p = np.random.uniform(0.6, 1.4, size=num_buses)
        active_demands = case["bus"][:, 2]
        case["bus"][:, 2] = active_demands.astype(np.float64) * bus_perturbation_p  # active demand

        bus_perturbation_q = np.random.uniform(0.6, 1.4, size=num_buses)
        reactive_demands = case["bus"][:, 3]
        case["bus"][:, 3] = reactive_demands.astype(np.float64) * bus_perturbation_q  # reactive demand


        ppopt = pp.ppoption(VERBOSE=0, OUT_ALL=0)
        results_dcopf = pp.rundcopf(case, ppopt)
        
        num_gens = len(case['gen'])
        
        case["gen"][:, 1] = results_dcopf['gen'][:, 1] * np.random.normal(1, 0.7, num_gens)
        case["gen"][:, 2] = results_dcopf['gen'][:, 2] + np.random.uniform(-100, 100, num_gens)
        
        if uncertainty_level == 'renewable':
            case = modify_generations(case)

    def run_power_flow(case, power_flow_type):
        """
       Runs power flow analysis on the power system case using either DC or AC methods.
       
       Args:
           case (dict): The power system case dictionary.
           power_flow_type (str): Specifies whether to run a DC or AC power flow analysis.

       Returns:
           dict: Results of the power flow analysis.
       """
        ppopt = pp.ppoption(VERBOSE=0, OUT_ALL=0)
        try:
            if power_flow_type == "dc":
                results = pp.rundcpf(case, ppopt)
            elif power_flow_type == "ac":
                results = pp.runpf(case, ppopt)
            else:
                raise ValueError("Invalid power flow type. Please specify 'dc' or 'ac'.")

            return results
        except Exception as e:
            print(f"Error running power flow analysis: {e}")
            return None
        

    # Set the random seed
    np.random.seed(seed)

    # Generate data for the specified number of samples
    successful_samples = 0  # Counter for successful power flow analysis
    data_list = []
    total_time = 0
    while successful_samples < num_samples:
        print(f"Generating Sample {successful_samples + 1}/{num_samples}")

        # Create a sample power system network
        if case_name == "random":
            case_name_i = np.random.choice(CASE_NAMES, p=CASE_WEIGHTS)
        else:
            case_name_i = case_name
        case = getattr(pp, case_name_i)()  # Fetch the specified case using the case_name input
        baseMVA = case["baseMVA"]
        # Randomly perturb load settings
        perturb_loads(case, uncertainty_level)
        
        if args.analysis_type == 'line_out':
            remove_random_branch(case)
        
        if args.analysis_type == 'gen_out':
            remove_random_generator(case)
        
        
        if not check_connectivity(case):
            print("Disconnected buses detected. Skipping this case.")
            continue

        start_time = time.time()
        # Run power flow analysis
        results = run_power_flow(case, power_flow_type)
        end_time = time.time()

        run_time = end_time - start_time

        if results and (np.sum(np.isnan(results[0]['bus'][:, 7])) == 0 and np.sum(np.isnan(results[0]['bus'][:, 8])) == 0):
            total_time += run_time
            bus_active_output = [0] * len(case["bus"])
            bus_reactive_output = [0] * len(case["bus"])
            bus_initial_voltage = [1] * len(case["bus"])

            gen_ids = results[0]['gen'][:, 0]
            gen_active_output = results[0]['gen'][:, 1] / baseMVA
            gen_reactive_output = results[0]['gen'][:, 2] / baseMVA
            gen_voltage_setpoint = results[0]['gen'][:, 5]

            # Convert gen_ids to integers
            gen_ids = gen_ids.astype(int)
            for i, index in enumerate(gen_ids):
                idx = np.where(case["bus"][:, 0] == index)[0][0]
                bus_active_output[idx] = gen_active_output[i]
                bus_reactive_output[idx] = gen_reactive_output[i]
                bus_initial_voltage[idx] = gen_voltage_setpoint[i]

            # Extract voltage magnitudes and active power flows as NumPy arrays
            bus_types = results[0]['bus'][:, 1]
            bus_voltage_magnitudes = results[0]['bus'][:, 7]
            bus_voltage_angles = results[0]['bus'][:, 8] * np.pi / 180
            bus_susceptances = results[0]['bus'][:, 5] / baseMVA
            bus_conductances = results[0]['bus'][:, 4] / baseMVA
            bus_active_demands = results[0]["bus"][:, 2] / baseMVA
            bus_reactive_demands = results[0]["bus"][:, 3] / baseMVA
            bus_active_outputs = bus_active_output
            bus_reactive_outputs = bus_reactive_output
            bus_initial_voltages = bus_initial_voltage

            # Save node IDs and edge start/end nodes
            edge_start_nodes = results[0]['branch'][:, 0]
            edge_end_nodes = results[0]['branch'][:, 1]
            edge_resistance = results[0]['branch'][:, 2]
            edge_reactance = results[0]['branch'][:, 3]
            edge_susceptance = results[0]['branch'][:, 4]
            edge_transformation_ratio = results[0]['branch'][:, 8]
            edge_shift_angle = results[0]['branch'][:, 9] * np.pi / 180
            edge_flow_active = results[0]['branch'][:, 13]
            edge_flow_reactive = results[0]['branch'][:, 14]
            edge_loss = np.abs(np.abs(results[0]['branch'][:, 13]) - np.abs(results[0]['branch'][:, 15]))
            
            # Convert arrays to use them in graph structures
            bus_types = np.array(bus_types)
            bus_active_demands = np.array(bus_active_demands)
            bus_reactive_demands = np.array(bus_reactive_demands)
            bus_active_outputs = np.array(bus_active_outputs)
            bus_reactive_outputs = np.array(bus_reactive_outputs)
            bus_susceptances = np.array(bus_susceptances)
            bus_conductances = np.array(bus_conductances)
            bus_initial_voltages = np.array(bus_initial_voltages)
            bus_voltage_magnitudes = np.array(bus_voltage_magnitudes)
            bus_voltage_angles = np.array(bus_voltage_angles)
            edge_resistance = np.array(edge_resistance)
            edge_reactance = np.array(edge_reactance)
            edge_susceptance = np.array(edge_susceptance)
            edge_transformation_ratio = np.array(edge_transformation_ratio)
            edge_shift_angle = np.array(edge_shift_angle)
            edge_start_nodes = np.array(edge_start_nodes) - 1
            edge_end_nodes = np.array(edge_end_nodes) - 1

            
            # Prepare the edge indices for graph structure data
            edge_index = np.concatenate((edge_start_nodes.reshape(1, -1), edge_end_nodes.reshape(1, -1)),
                                        axis=0)
            
            # Prepare node and edge features depending on the analysis type
            if analysis_type == 'dc_approx':
                result_dc = run_power_flow(case, 'dc')
                
                dc_bus_voltage_angles = result_dc[0]['bus'][:, 8] * np.pi / 180
                
                node_features = np.concatenate(
                        (
                            bus_active_demands.reshape(-1, 1),
                            bus_reactive_demands.reshape(-1, 1),
                            bus_active_outputs.reshape(-1, 1),
                            bus_reactive_outputs.reshape(-1, 1),
                            bus_initial_voltages.reshape(-1, 1),
                            dc_bus_voltage_angles.reshape(-1, 1)
                        ),
                        axis=1,
                    )
            else:
                node_features = np.concatenate(
                        (
                            bus_active_demands.reshape(-1, 1),
                            bus_reactive_demands.reshape(-1, 1),
                            bus_active_outputs.reshape(-1, 1),
                            bus_reactive_outputs.reshape(-1, 1),
                            bus_initial_voltages.reshape(-1, 1),
                        ),
                        axis=1,
                    )
            
            # Prepare the features for graph structure data                
            edge_features = np.concatenate((edge_resistance.reshape(-1, 1), edge_reactance.reshape(-1, 1), edge_susceptance.reshape(-1, 1), edge_transformation_ratio.reshape(-1, 1), edge_shift_angle.reshape(-1, 1), edge_flow_active.reshape(-1, 1), edge_flow_reactive.reshape(-1, 1), edge_loss.reshape(-1, 1)), axis=1)
            node_targets = np.concatenate((bus_voltage_magnitudes.reshape(-1, 1), bus_voltage_angles.reshape(-1, 1), bus_reactive_outputs.reshape(-1, 1)), axis=1)
            additional_features =np.concatenate((bus_reactive_outputs.reshape(-1, 1), bus_conductances.reshape(-1, 1), bus_susceptances.reshape(-1, 1), bus_types.reshape(-1, 1)), axis=1)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.int64)
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(node_targets, dtype=torch.float)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            x_add = torch.tensor(additional_features, dtype=torch.float)
            
            # Create a Data object
            data = Data(x=x, y=y, edge_index=edge_index_tensor, edge_attr=edge_attr, x_add=x_add)
            
            # Append the Data object to the list
            data_list.append(data)
            successful_samples += 1  # Increment successful samples counter

    # Create a folder based on the case name
    if args.outage_type == 'none':
        folder = f"data/{analysis_type}/{uncertainty_level}/{case_name}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    else:
        folder = f"data/{args.outage_type}/{analysis_type}/{uncertainty_level}/{case_name}"
        if not os.path.exists(folder):
            os.makedirs(folder)

    file_name = f"{case_name}_{save_path}_{power_flow_type}.pkl"
    folder_save_path = os.path.join(folder, file_name)

    with open(folder_save_path, "wb") as file:
        # Dump the data_list into the file
        pickle.dump(data_list, file)

    print(f"Successfully generated and saved {num_samples} training samples to {file_name}.\n")
    print(f"Average Run Time: {total_time/num_samples}")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Generate and save training data for power system analysis.")

    # Add arguments to the parser
    parser.add_argument("--num_samples", type=int, default=20000,
                        help="Number of power system networks to generate and save.")
    parser.add_argument("--save_path", type=str, default='test_data',
                        help="File path to save the generated data.")
    parser.add_argument("--seed", type=int, default=231, help="Random seed for reproducibility.")
    parser.add_argument("--case_name", type=str, default="case14", help="Name of the power system case to use.")
    parser.add_argument("--power_flow_type", type=str, default="dc", choices=["ac", "dc"],
                        help="Type of power flow analysis. Choose 'ac' for AC power flow or 'dc' for DC power flow.")
    
    parser.add_argument("--uncertainty_level", type=str, default='renewable', choices=['demand', 'renewable'],
                        help="If uncertainty is only in demand or demand and renewables.")
    parser.add_argument("--analysis_type", type=str, default='normal', choices=['normal', 'dc_approx'],
                        help="If dc approximation are saved as node feaures.")
    parser.add_argument("--outage_type", type=str, default='none', choices=['none', 'line_out', 'gen_out'],
                        help="If there is an outage in the case for checkin N-1 constraints.")
    

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    generate_and_save_training_data(args.num_samples, args.save_path, args.seed, args.case_name, args.power_flow_type, args.uncertainty_level, args.analysis_type)
