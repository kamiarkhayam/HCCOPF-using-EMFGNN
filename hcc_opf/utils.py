# -*- coding: utf-8 -*-
"""
Utilities for Hybrid Chance-Constrained Optimal Power Flow

This module provides essential utilities for performing simulations and analyses of HCC_OPF. 
It supports operations such as the removal of specific power system components (branches and generators), 
the calculation of shortest path distances between buses, and the generation of correlated samples based 
on network topology. These functionalities are crucial for studying the effects of random outages and 
the propagation of disturbances within the network.

The module uses libraries such as numpy for numerical operations, scipy for statistical functions, and 
networkx for graph-based computations, ensuring efficient and robust processing.

Created by: Kamiar Khayambashi
"""

import numpy as np
import scipy.stats
import networkx as nx

# Line indices that will cause islanding in power systems
EXCLUDE_LIST_BY_MAX_VALUE = {
    9: [0, 3, 6], #Case 9
    20: [13, 16], #Case 14
    41: [12, 15, 33], #Case 30
    186: [6, 8, 180, 179, 112, 117, 132, 133, 175, 176] #Case 118
}

def choose_random_line(max_value):
    """
    Selects a random line index to be removed from the power grid model,
    excluding specific lines based on the `max_value`.

    Args:
        max_value (int): The upper limit of line indices, typically the number of lines in the grid.

    Returns:
        int: A random line index that is not in the exclude list for the given `max_value`.
    """
    exclude_list = EXCLUDE_LIST_BY_MAX_VALUE.get(max_value, [])
    possible_values = set(range(max_value)) - set(exclude_list)
    return np.random.choice(list(possible_values))

def remove_random_branch(case):
    """
    Removes a random branch from the power system case.

    Args:
        case (dict): The power system case dictionary which includes all branches.
    """
    num_lines = len(case["branch"])
    if num_lines > 1:
        random_branch_idx = choose_random_line(num_lines)
        case["branch"] = np.delete(case["branch"], random_branch_idx, axis=0)
        
def remove_random_generator(case):
    """
    Removes a random generator from the power system case, avoiding the main generator.

    Args:
        case (dict): The power system case dictionary which includes all generators.
    """
    main_gen_idx = 25 if len(case['bus']) == 118 else 0
    num_gens = len(case['gen'])
    gen_to_cut = np.random.randint(0, num_gens - 1)
    if gen_to_cut >= main_gen_idx:
        gen_to_cut += 1
    gen_to_cut_index = int(case['gen'][gen_to_cut, 0] - 1)
    case['gen'] = np.delete(case['gen'], gen_to_cut, axis=0)
    case['gencost'] = np.delete(case['gencost'], gen_to_cut, axis=0)
    case['bus'][gen_to_cut_index, 1] = 1  # Mark bus as PQ bus after removing generator

def compute_shortest_path_distances(case):
    """
    Computes the shortest path distances between all pairs of buses in the power system network.

    Args:
        case (dict): Contains the system's bus and branch information, where each branch defines a connection between buses.

    Returns:
        np.array: A matrix of shortest path distances between all pairs of buses.
    """
    graph = nx.Graph()
    for branch in case['branch']:
        graph.add_edge(int(branch[0]) - 1, int(branch[1]) - 1, weight=1)  # weight is set to 1
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph))
    num_buses = len(case['bus'])
    distances = np.zeros((num_buses, num_buses))
    for i in range(num_buses):
        for j in range(num_buses):
            distances[i, j] = lengths[i][j]  # No need to add 1, as indices are adjusted to 0-based
    return distances

def create_correlation_matrix(distances, decay=0.1):
    """
    Creates a correlation matrix from distance matrix using an exponential decay function.

    Args:
        distances (np.array): Matrix of distances between points/nodes.
        decay (float): Decay factor controlling how correlation decreases with distance.

    Returns:
        np.array: Correlation matrix derived from the distances.
    """
    return np.exp(-decay * distances)

def generate_correlated_uniform_samples(correlation_matrix, size):
    """
    Generates uniformly distributed correlated samples based on a correlation matrix.

    Args:
        correlation_matrix (np.array): Correlation matrix defining dependencies between variables.
        size (int): Number of samples to generate.

    Returns:
        np.array: Uniformly distributed correlated samples.
    """
    normal_samples = np.random.multivariate_normal(np.zeros(correlation_matrix.shape[0]), correlation_matrix, size=size)
    uniform_samples = scipy.stats.norm.cdf(normal_samples)
    return uniform_samples
        
def generate_correlated_weights(correlation_matrix, size):
    """
    Generates normalized, non-negative correlated weights from a correlation matrix.

    Args:
        correlation_matrix (np.array): Correlation matrix.
        size (int): Number of sample sets to generate.

    Returns:
        np.array: Normalized weights, each set summing to 1.
    """
    normal_samples = np.random.multivariate_normal(np.zeros(correlation_matrix.shape[0]), correlation_matrix, size=size)
    weights = np.abs(normal_samples)  # Ensure non-negative weights
    normalized_weights = weights / weights.sum()  # Normalize to sum to 1
    return normalized_weights

def generate_correlated_samples(correlation_matrix, size, mean=1, cov=0.1):
    """
    Generates correlated samples for a variable with specified mean and standard deviation derived from a coefficient of variation.

    Args:
        correlation_matrix (np.array): Correlation matrix.
        size (int): Number of sample sets to generate.
        mean (float): Desired mean value of the samples.
        cov (float): Coefficient of variation, defining the standard deviation as a fraction of the mean.

    Returns:
        np.array: Correlated samples with the specified mean and variation.
    """
    std_dev = cov * mean
    normal_samples = np.random.multivariate_normal(np.zeros(correlation_matrix.shape[0]), correlation_matrix, size=size)
    scaled_samples = mean + std_dev * normal_samples
    return scaled_samples

def check_connectivity(case):
    """
    Checks if the power system remains fully connected (i.e., no islands) after
    modifications such as branch or generator removals.

    Args:
        case (dict): The modified power system case dictionary.

    Returns:
        bool: True if the system is still connected, False otherwise.
    """
    G = nx.Graph()
    for line in case["branch"]:
        G.add_edge(int(line[0]), int(line[1]))
    return nx.is_connected(G)