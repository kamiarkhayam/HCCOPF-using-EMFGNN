# -*- coding: utf-8 -*-
"""
This script implements and trains both Enhanced Multi-Fidelity Graph Neural Networks (EMF-GNN) and Single-Fidelity Graph Neural Networks (SF-GNN) for power system analysis. 
It includes the generation and saving of training data, model definitions, and training and evaluation routines.

The script supports a variety of power system cases, providing extensive customization in terms of data dimensions, training epochs, batch sizes, and learning rates. 
It leverages PyTorch and PyTorch Geometric for neural network operations, and NetworkX for visualization of prediction errors.

This script is adapted and modified from 
"Multi-fidelity Graph Neural Networks for Efficient Power Flow Analysis under High-Dimensional 
Demand and Renewable Generation Uncertainty".

Created by: Kamiar Khayambashi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import TAGConv, GraphConv, GATConv, EdgeConv, SAGEConv, SGConv, APPNP, ChebConv, AGNNConv, GCNConv, GINConv
from torch.nn import L1Loss
from lf_gnn import LFGNN
from torch.optim.lr_scheduler import StepLR
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import logging
import os
import time  # Add the time module for measuring execution time
import random


def set_random_seeds(seed):
    """
    Sets the random seed for all necessary libraries to ensure reproducibility.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set a specific random seed to ensure reproducible results
seed = 321
set_random_seeds(seed)


class HFGNN(nn.Module):
    """
    Defines a Single-Fidelity Graph Neural Network (SF-GNN) model using Graph Convolutional layers.
    This model is structured to process input features through multiple layers of graph convolutions,
    applying non-linear activation functions to capture complex patterns in data.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HFGNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)  # First convolutional layer
        self.conv2 = GraphConv(hidden_dim, hidden_dim)  # Second convolutional layer
        self.conv3 = GraphConv(hidden_dim, output_dim)  # Output layer

    def forward(self, x, edge_index):
        """
        Forward pass of the neural network.
        Args:
            x (Tensor): Input features.
            edge_index (LongTensor): Graph structure in COO format.
        Returns:
            Tensor: The output of the network after processing.
        """
        x = F.relu(self.conv1(x, edge_index))  # Apply ReLU activation function after first convolution
        x = F.relu(self.conv2(x, edge_index))  # Apply ReLU activation function after second convolution
        x = self.conv3(x, edge_index)  # Apply final convolution to produce output features
        return x

class MFGNN(nn.Module):
    """
    Multi-Fidelity Graph Neural Network (MFGNN) that incorporates additional low-fidelity embeddings into the model.
    This model extends the capabilities of standard GNNs by merging high-fidelity and low-fidelity information,
    aimed at improving prediction accuracy and robustness.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, embedding_dim):
        super(MFGNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)  # First convolutional layer
        # Second convolutional layer that also incorporates the embedding dimension
        self.conv2 = GraphConv(hidden_dim + embedding_dim, hidden_dim + embedding_dim)
        self.conv3 = GraphConv(hidden_dim + embedding_dim, output_dim)  # Output layer

    def forward(self, x, edge_index, additional_x, lf_embedding):
        """
        Forward pass of the MFGNN.
        Args:
            x (Tensor): High-fidelity features.
            edge_index (LongTensor): Graph structure in COO format.
            additional_x (Tensor): Additional features to concatenate.
            lf_embedding (Tensor): Low-fidelity embedding to enhance the model's input.
        Returns:
            Tensor: The output of the network after processing.
        """
        x = torch.cat([x, additional_x], dim=1)  # Concatenate additional features
        x = F.relu(self.conv1(x, edge_index))  # Apply ReLU activation function after first convolution
        x = torch.cat([x, lf_embedding], dim=1)  # Concatenate low-fidelity embedding
        x = F.relu(self.conv2(x, edge_index))  # Apply ReLU activation function after second convolution
        x = self.conv3(x, edge_index)  # Apply final convolution to produce output features
        return x + additional_x  # Return output with a residual connection

def load_data(file_path, device):
    """
    Loads data from a pickle file and moves it to the specified computing device.
    Args:
        file_path (str): Path to the data file.
        device (torch.device): Device to which the data should be moved (e.g., CPU or GPU).
    Returns:
        list: List of data objects loaded onto the specified device.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return [d.to(device) for d in data]

def train_low_fidelity_model(model, optimizer, loss_function, train_data, val_data, num_epochs, batch_size, device, save_path):
    """
    Trains a low-fidelity model over a specified number of epochs and evaluates it using validation data.
    Tracks and logs training and validation loss and mean absolute error (MAE) for each epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        loss_function (callable): The loss function used for training.
        train_data (DataLoader): DataLoader containing the training data.
        val_data (DataLoader): DataLoader containing the validation data.
        num_epochs (int): Total number of epochs to train.
        batch_size (int): The size of each batch of data.
        device (torch.device): The device on which to train the model.
        save_path (str): Path to save the best model state.

    Returns:
        tuple: Returns lists of training losses, validation losses, training MAE losses, and validation MAE losses.
    """
    train_losses = []  # List to store loss per epoch for training data.
    val_losses = []  # List to store loss per epoch for validation data.
    train_mae_losses = []  # List to store MAE per epoch for training data.
    val_mae_losses = []  # List to store MAE per epoch for validation data.
    best_val_loss = float('inf')  # Initialize best validation loss for model saving.

    mae_loss = L1Loss()  # Initialize the Mean Absolute Error loss function.

    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing the epoch.
        total_loss = 0.0  # Accumulate total loss for this epoch.
        total_mae_loss = 0.0  # Accumulate total MAE for this epoch.
        num_samples = 0  # Counter for total number of samples processed.

        model.train()  # Set the model to training mode.
        for batch in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            predictions, _ = model(x, edge_index)

            loss = loss_function(predictions[:, 0], batch.y[:, 1].to(device))  # Calculate loss.
            mae_loss_val = mae_loss(predictions[:, 0], batch.y[:, 1].to(device))  # Calculate MAE.

            optimizer.zero_grad()  # Clear gradients.
            loss.backward()  # Backpropagate the error.
            optimizer.step()  # Update model parameters.

            total_loss += loss.item() * batch.num_graphs  # Aggregate loss scaled by batch size.
            total_mae_loss += mae_loss_val.item() * batch.num_graphs  # Aggregate MAE scaled by batch size.
            num_samples += batch.num_graphs  # Count total samples processed.

        avg_loss = total_loss / num_samples  # Average loss for this epoch.
        avg_mae_loss = total_mae_loss / num_samples  # Average MAE for this epoch.
        train_losses.append(avg_loss)  # Append average loss to list.
        train_mae_losses.append(avg_mae_loss)  # Append average MAE to list.

        # Evaluate model on validation data.
        validation_loss, validation_mae_loss, validation_time = evaluate_low_fidelity(model, val_data, batch_size, device)
        val_losses.append(validation_loss)  # Append validation loss.
        val_mae_losses.append(validation_mae_loss)  # Append validation MAE.

        end_time = time.time()  # End timing the epoch.
        training_time = (end_time - start_time) / num_samples  # Calculate time per sample.

        # Log training and validation results.
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train MSE Loss: {avg_loss:.4e}, Validation MSE Loss: {validation_loss:.4e}, Train MAE Loss: {avg_mae_loss:.4e}, Validation MAE Loss: {validation_mae_loss:.4e}, Training Time: {training_time:.5e} seconds, Inference Time: {validation_time:.5e} seconds")

        # Save the model if the validation loss is the best so far.
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, train_mae_losses, val_mae_losses

def evaluate_low_fidelity(model, data, batch_size, device):
    """
    Evaluates the low-fidelity model on given data using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data (DataLoader): DataLoader containing the data for evaluation.
        batch_size (int): Batch size to use for evaluation.
        device (torch.device): Device on which to evaluate the model.
    
    Returns:
        tuple: Average loss, average MAE, and average evaluation time per sample.
    """
    model.eval()  # Set the model to evaluation mode.
    mse_loss = nn.MSELoss()  # Initialize MSE loss function.
    mae_loss = L1Loss()  # Initialize MAE loss function.
    total_loss = 0.0  # Total accumulated loss.
    total_mae_loss = 0.0  # Total accumulated MAE.
    num_samples = 0  # Counter for total samples processed.
    start_time = time.time()  # Start timing the evaluation.

    with torch.no_grad():  # Disable gradient computation.
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            predictions, _ = model(x, edge_index)

            loss = mse_loss(predictions[:, 0], batch.y[:, 1].to(device))  # Calculate MSE loss.
            mae_loss_val = mae_loss(predictions[:, 0], batch.y[:, 1].to(device))  # Calculate MAE.

            total_loss += loss.item() * batch.num_graphs  # Aggregate loss.
            total_mae_loss += mae_loss_val.item() * batch.num_graphs  # Aggregate MAE.
            num_samples += batch.num_graphs  # Count total samples processed.

    end_time = time.time()  # End timing the evaluation.
    epoch_time = end_time - start_time  # Calculate total time taken.

    return total_loss / num_samples, total_mae_loss / num_samples, epoch_time / num_samples

def train_high_fidelity_model(model, optimizer, loss_function, train_data, val_data, num_epochs, batch_size, device, save_path):
    """
    Trains a high-fidelity GNN model for the specified number of epochs, evaluating it on validation data, and tracks both
    MSE and MAE losses for voltage and phase predictions.

    Args:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_function (callable): Loss function to measure the error.
        train_data (DataLoader): DataLoader containing training data.
        val_data (DataLoader): DataLoader containing validation data.
        num_epochs (int): Total number of epochs to train the model.
        batch_size (int): Number of samples in each batch.
        device (torch.device): Device to perform computation on.
        save_path (str): Path to save the best model based on validation loss.

    Returns:
        tuple: Lists containing train and validation losses and MAEs for voltage and phase, and overall metrics.
    """
    # Initialize lists to store loss and MAE metrics for detailed tracking
    train_losses_v, train_losses_d, train_losses = [], [], []
    train_mae_losses_v, train_mae_losses_d, train_mae_losses = [], [], []
    val_losses_v, val_losses_d, val_losses = [], [], []
    val_mae_losses_v, val_mae_losses_d, val_mae_losses = [], [], []
    best_val_loss = float('inf')  # Initialize best validation loss for early stopping

    mae_loss = L1Loss()  # Instantiate Mean Absolute Error loss function

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing the epoch
        # Initialize accumulators for losses and MAEs
        total_loss_v, total_loss_d, total_loss = 0.0, 0.0, 0.0
        total_mae_loss_v, total_mae_loss_d, total_mae_loss = 0.0, 0.0, 0.0
        num_samples = 0  # Counter for samples seen

        model.train()  # Set model to training mode
        for batch in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            predictions = model(x, edge_index)

            # Compute losses for voltage and phase separately and their MAEs
            loss_v = loss_function(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = loss_function(predictions[:, 1], batch.y[:, 1].to(device))
            mae_loss_v = mae_loss(predictions[:, 0], batch.y[:, 0].to(device))
            mae_loss_d = mae_loss(predictions[:, 1], batch.y[:, 1].to(device))

            # Aggregate total losses and MAEs
            loss = loss_v + loss_d
            mae_loss_total = mae_loss_v + mae_loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total losses and MAEs for voltage and phase
            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            total_mae_loss_v += mae_loss_v.item() * batch.num_graphs
            total_mae_loss_d += mae_loss_d.item() * batch.num_graphs
            total_mae_loss += mae_loss_total.item() * batch.num_graphs
            num_samples += batch.num_graphs  # Update sample count

        # Compute average losses and MAEs for voltage, phase, and overall
        avg_loss_v, avg_loss_d, avg_loss = total_loss_v / num_samples, total_loss_d / num_samples, total_loss / num_samples
        avg_mae_loss_v, avg_mae_loss_d, avg_mae_loss = total_mae_loss_v / num_samples, total_mae_loss_d / num_samples, total_mae_loss / num_samples

        # Log training progress
        validation_loss_v, validation_loss_d, validation_loss, validation_mae_loss_v, validation_mae_loss_d, validation_mae_loss, validation_time = evaluate_high_fidelity(model, val_data, batch_size, device)
        # Update lists for losses and MAEs
        train_losses_v.append(avg_loss_v)
        train_losses_d.append(avg_loss_d)
        train_losses.append(avg_loss)
        train_mae_losses_v.append(avg_mae_loss_v)
        train_mae_losses_d.append(avg_mae_loss_d)
        train_mae_losses.append(avg_mae_loss)
        val_losses_v.append(validation_loss_v)
        val_losses_d.append(validation_loss_d)
        val_losses.append(validation_loss)
        val_mae_losses_v.append(validation_mae_loss_v)
        val_mae_losses_d.append(validation_mae_loss_d)
        val_mae_losses.append(validation_mae_loss)

        end_time = time.time()  # End time for epoch
        training_time = (end_time - start_time) / num_samples  # Compute training time per sample
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train MSE Loss: {avg_loss:.4e}, Validation MSE Loss: {validation_loss:.4e}, Train MAE Loss: {avg_mae_loss:.4e}, Validation MAE Loss: {validation_mae_loss:.4e}, Training Time: {training_time:.5e} seconds, Inference Time: {validation_time:.5e} seconds")

        # Save model if validation loss is the best seen so far
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

    return train_losses_v, val_losses_v, train_losses_d, val_losses_d, train_losses, val_losses, train_mae_losses_v, val_mae_losses_v, train_mae_losses_d, val_mae_losses_d, train_mae_losses, val_mae_losses

def evaluate_high_fidelity(model, data, batch_size, device):
    """
    Evaluates the high-fidelity model on given data using MSE and MAE for both voltage and phase predictions.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data (DataLoader): DataLoader containing the data for evaluation.
        batch_size (int): Batch size to use for evaluation.
        device (torch.device): Device on which to evaluate the model.
    
    Returns:
        tuple: Average loss and MAE values for voltage and phase, and the average evaluation time per sample.
    """
    model.eval()  # Set the model to evaluation mode
    mse_loss_f = nn.MSELoss()  # Initialize MSE loss function
    mae_loss_f = L1Loss()  # Initialize MAE loss function
    # Initialize accumulators for losses and MAEs
    total_loss_v, total_loss_d, total_loss = 0.0, 0.0, 0.0
    total_mae_loss_v, total_mae_loss_d, total_mae_loss = 0.0, 0.0, 0.0
    num_samples = 0  # Counter for samples seen

    start_time = time.time()  # Start timing the evaluation
    with torch.no_grad():  # Disable gradient computation
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            predictions = model(x, edge_index)

            # Compute MSE and MAE for voltage and phase predictions
            loss_v = mse_loss_f(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = mse_loss_f(predictions[:, 1], batch.y[:, 1].to(device))
            mae_loss_v = mae_loss_f(predictions[:, 0], batch.y[:, 0].to(device))
            mae_loss_d = mae_loss_f(predictions[:, 1], batch.y[:, 1].to(device))

            # Aggregate total losses and MAEs
            loss = loss_v + loss_d
            mae_loss_total = mae_loss_v + mae_loss_d

            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            total_mae_loss_v += mae_loss_v.item() * batch.num_graphs
            total_mae_loss_d += mae_loss_d.item() * batch.num_graphs
            total_mae_loss += mae_loss_total.item() * batch.num_graphs

            num_samples += batch.num_graphs  # Update sample count

    end_time = time.time()  # End time for evaluation
    epoch_time = end_time - start_time  # Compute total time taken

    # Calculate and return average losses, MAEs, and evaluation time per sample
    return total_loss_v / num_samples, total_loss_d / num_samples, total_loss / num_samples, total_mae_loss_v / num_samples, total_mae_loss_d / num_samples, total_mae_loss / num_samples, epoch_time / num_samples


def train_multi_fidelity_model(model, optimizer, loss_function, train_data, val_data, num_epochs, batch_size, low_fidelity_model, device, save_path):
    """
    Trains a multi-fidelity GNN model using both high and low-fidelity predictions. This function handles the training process
    over multiple epochs and evaluates the model on a validation set. It also logs the training and validation loss and MAE.

    Args:
        model (torch.nn.Module): The high-fidelity GNN model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer to use for adjusting model weights.
        loss_function (callable): Function to calculate the loss between predictions and true values.
        train_data (DataLoader): DataLoader providing the training data.
        val_data (DataLoader): DataLoader providing the validation data.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples in each batch.
        low_fidelity_model (torch.nn.Module): Pre-trained low-fidelity model for generating additional features.
        device (torch.device): Device on which to perform computations.
        save_path (str): Path where the best model weights are saved.

    Returns:
        tuple: Lists containing training and validation losses and MAEs, separated by voltage and phase.
    """
    # Initialize lists to store losses and MAEs for detailed tracking
    train_losses_v, train_losses_d, train_losses = [], [], []
    train_mae_losses_v, train_mae_losses_d, train_mae_losses = [], [], []
    val_losses_v, val_losses_d, val_losses = [], [], []
    val_mae_losses_v, val_mae_losses_d, val_mae_losses = [], [], []
    best_val_loss = float('inf')  # For tracking the best validation loss for model saving

    mae_loss = L1Loss()  # Instantiate Mean Absolute Error loss function for evaluation

    # Training loop over specified number of epochs
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing the epoch
        # Initialize accumulators for losses and MAEs
        total_loss_v, total_loss_d, total_loss = 0.0, 0.0, 0.0
        total_mae_loss_v, total_mae_loss_d, total_mae_loss = 0.0, 0.0, 0.0
        num_samples = 0  # Counter for samples processed

        model.train()  # Set model to training mode
        for batch in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            # Generate low-fidelity predictions and embeddings without gradient tracking
            with torch.no_grad():
                low_fidelity_predictions, low_fidelity_embedding = low_fidelity_model(x, edge_index)

            # Extract phase predictions and create an input tensor with high and low fidelity data
            low_fidelity_phase = low_fidelity_predictions[:, 0].view(-1, 1)
            low_fidelity_voltage = torch.ones_like(low_fidelity_phase)
            additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1)

            # Make predictions using the high-fidelity model
            predictions = model(x, edge_index, additional_x, low_fidelity_embedding)

            # Compute losses for voltage and phase predictions
            loss_v = loss_function(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = loss_function(predictions[:, 1], batch.y[:, 1].to(device))
            mae_loss_v = mae_loss(predictions[:, 0], batch.y[:, 0].to(device))
            mae_loss_d = mae_loss(predictions[:, 1], batch.y[:, 1].to(device))

            # Aggregate the total losses and MAEs
            loss = loss_v + loss_d
            mae_loss_total = mae_loss_v + mae_loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the total losses and MAEs for the batch
            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            total_mae_loss_v += mae_loss_v.item() * batch.num_graphs
            total_mae_loss_d += mae_loss_d.item() * batch.num_graphs
            total_mae_loss += mae_loss_total.item() * batch.num_graphs
            num_samples += batch.num_graphs  # Update sample count

        # Calculate average losses and MAEs for voltage and phase
        avg_loss_v = total_loss_v / num_samples
        avg_loss_d = total_loss_d / num_samples
        avg_loss = total_loss / num_samples
        avg_mae_loss_v = total_mae_loss_v / num_samples
        avg_mae_loss_d = total_mae_loss_d / num_samples
        avg_mae_loss = total_mae_loss / num_samples

        # Update lists for tracking losses and MAEs over epochs
        train_losses_v.append(avg_loss_v)
        train_losses_d.append(avg_loss_d)
        train_losses.append(avg_loss)
        train_mae_losses_v.append(avg_mae_loss_v)
        train_mae_losses_d.append(avg_mae_loss_d)
        train_mae_losses.append(avg_mae_loss)

        # Evaluate the model on the validation set
        validation_loss_v, validation_loss_d, validation_loss, validation_mae_loss_v, validation_mae_loss_d, validation_mae_loss, validation_time = evaluate_multi_fidelity(model, val_data, batch_size, low_fidelity_model, device)
        val_losses_v.append(validation_loss_v)
        val_losses_d.append(validation_loss_d)
        val_losses.append(validation_loss)
        val_mae_losses_v.append(validation_mae_loss_v)
        val_mae_losses_d.append(validation_mae_loss_d)
        val_mae_losses.append(validation_mae_loss)

        end_time = time.time()  # End time for the epoch
        training_time = (end_time - start_time) / num_samples  # Calculate training time per sample
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train MSE Loss: {avg_loss:.4e}, Validation MSE Loss: {validation_loss:.4e}, Train MAE Loss: {avg_mae_loss:.4e}, Validation MAE Loss: {validation_mae_loss:.4e}, Training Time: {training_time:.5e} seconds, Inference Time: {validation_time:.5e} seconds")

        # Save the model if the current validation loss is the lowest observed
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

    return train_losses_v, val_losses_v, train_losses_d, val_losses_d, train_losses, val_losses, train_mae_losses_v, val_mae_losses_v, train_mae_losses_d, val_mae_losses_d, train_mae_losses, val_mae_losses

def evaluate_multi_fidelity(model, data, batch_size, low_fidelity_model, device):
    """
    Evaluates the multi-fidelity model using both high and low fidelity data sources.
    Calculates and returns the mean squared error (MSE), mean absolute error (MAE), and
    computational time for the evaluation.

    Args:
        model (torch.nn.Module): The high-fidelity GNN model to be evaluated.
        data (DataLoader): DataLoader providing the validation or test data.
        batch_size (int): Number of samples in each batch.
        low_fidelity_model (torch.nn.Module): The low-fidelity model used for generating additional features.
        device (torch.device): Device on which to perform computations.

    Returns:
        tuple: Average MSE, MAE, and evaluation time per sample across all batches for voltage and phase predictions.
    """
    model.eval()  # Set the model to evaluation mode
    mse_loss_f = nn.MSELoss()  # Loss function for calculating the mean squared error
    mae_loss_f = L1Loss()  # Loss function for calculating the mean absolute error
    # Initialize counters for losses and sample numbers
    total_loss_v, total_loss_d, total_loss = 0.0, 0.0, 0.0
    total_mae_loss_v, total_mae_loss_d, total_mae_loss = 0.0, 0.0, 0.0
    num_samples = 0
    start_time = time.time()  # Start timer for evaluation

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            # Get predictions from the low-fidelity model
            low_fidelity_predictions, low_fidelity_embedding = low_fidelity_model(x, edge_index)
            # Prepare additional inputs derived from low-fidelity predictions
            low_fidelity_phase = low_fidelity_predictions[:, 0].view(-1, 1)
            low_fidelity_voltage = torch.ones_like(low_fidelity_phase)
            additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1)
            # Get predictions from the high-fidelity model
            predictions = model(x, edge_index, additional_x, low_fidelity_embedding)
            # Calculate losses for voltage and phase predictions
            loss_v = mse_loss_f(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = mse_loss_f(predictions[:, 1], batch.y[:, 1].to(device))
            mae_loss_v = mae_loss_f(predictions[:, 0], batch.y[:, 0].to(device))
            mae_loss_d = mae_loss_f(predictions[:, 1], batch.y[:, 1].to(device))
            # Aggregate losses and MAE
            loss = loss_v + loss_d
            mae_loss_total = mae_loss_v + mae_loss_d
            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            total_mae_loss_v += mae_loss_v.item() * batch.num_graphs
            total_mae_loss_d += mae_loss_d.item() * batch.num_graphs
            total_mae_loss += mae_loss_total.item() * batch.num_graphs
            num_samples += batch.num_graphs  # Update sample count

    end_time = time.time()
    epoch_time = end_time - start_time  # Calculate evaluation time

    # Return average losses and MAE, and evaluation time per sample
    return total_loss_v / num_samples, total_loss_d / num_samples, total_loss / num_samples, total_mae_loss_v / num_samples, total_mae_loss_d / num_samples, total_mae_loss / num_samples, epoch_time / num_samples


def write_results(output_dir, case, N_MF_TRAIN, results):
    """
    Writes a summary of training and evaluation results for both multi-fidelity and high-fidelity models to a text file.

    Args:
        output_dir (str): Directory where to save the results file.
        case (str): Power system case identifier.
        N_MF_TRAIN (int): Number of training samples used for the multi-fidelity model.
        results (dict): Contains performance metrics such as MSE and MAE for both training and evaluation phases.

    The function appends results to a 'results_summary.txt' file located in the specified output directory. It logs detailed 
    performance metrics for both the multi-fidelity and high-fidelity training sessions, providing a structured record of 
    model performance across various test cases and training configurations.
    """
    
    # Define the path for the results summary file
    output_file = os.path.join(output_dir, 'results_summary.txt')
    
    # Open the results summary file in append mode
    with open(output_file, 'a') as f:
        # Write case and training sample size information
        f.write(f'Case: {case}, N_MF_TRAIN: {N_MF_TRAIN}\n')
        
        # Write training and evaluation results for multi-fidelity model
        f.write(f"Train MSE Loss - Magnitude (Multi-Fidelity): {results['train_mse_loss_v_multi_fidelity']:.5e}\n")
        f.write(f"Train MSE Loss - Phase (Multi-Fidelity): {results['train_mse_loss_d_multi_fidelity']:.5e}\n")
        f.write(f"Train MSE Loss (Multi-Fidelity): {results['train_mse_loss_multi_fidelity']:.5e}\n")
        f.write(f"Train MAE Loss - Magnitude (Multi-Fidelity): {results['train_mae_loss_v_multi_fidelity']:.5e}\n")
        f.write(f"Train MAE Loss - Phase (Multi-Fidelity): {results['train_mae_loss_d_multi_fidelity']:.5e}\n")
        f.write(f"Train MAE Loss (Multi-Fidelity): {results['train_mae_loss_multi_fidelity']:.5e}\n")
        f.write(f"Eval MSE Loss - Magnitude (Multi-Fidelity): {results['val_mse_loss_v_multi_fidelity']:.5e}\n")
        f.write(f"Eval MSE Loss - Phase (Multi-Fidelity): {results['val_mse_loss_d_multi_fidelity']:.5e}\n")
        f.write(f"Eval MSE Loss (Multi-Fidelity): {results['val_mse_loss_multi_fidelity']:.5e}\n")
        f.write(f"Eval MAE Loss - Magnitude (Multi-Fidelity): {results['val_mae_loss_v_multi_fidelity']:.5e}\n")
        f.write(f"Eval MAE Loss - Phase (Multi-Fidelity): {results['val_mae_loss_d_multi_fidelity']:.5e}\n")
        f.write(f"Eval MAE Loss (Multi-Fidelity): {results['val_mae_loss_multi_fidelity']:.5e}\n")
        
        # Write training and evaluation results for high-fidelity model
        f.write(f"Train MSE Loss - Magnitude (HF Only): {results['train_mse_loss_v_hf_only']:.5e}\n")
        f.write(f"Train MSE Loss - Phase (HF Only): {results['train_mse_loss_d_hf_only']:.5e}\n")
        f.write(f"Train MSE Loss (HF Only): {results['train_mse_loss_hf_only']:.5e}\n")
        f.write(f"Train MAE Loss - Magnitude (HF Only): {results['train_mae_loss_v_hf_only']:.5e}\n")
        f.write(f"Train MAE Loss - Phase (HF Only): {results['train_mae_loss_d_hf_only']:.5e}\n")
        f.write(f"Train MAE Loss (HF Only): {results['train_mae_loss_hf_only']:.5e}\n")
        f.write(f"Eval MSE Loss - Magnitude (HF Only): {results['val_mse_loss_v_hf_only']:.5e}\n")
        f.write(f"Eval MSE Loss - Phase (HF Only): {results['val_mse_loss_d_hf_only']:.5e}\n")
        f.write(f"Eval MSE Loss (HF Only): {results['val_mse_loss_hf_only']:.5e}\n")
        f.write(f"Eval MAE Loss - Magnitude (HF Only): {results['val_mae_loss_v_hf_only']:.5e}\n")
        f.write(f"Eval MAE Loss - Phase (HF Only): {results['val_mae_loss_d_hf_only']:.5e}\n")
        f.write(f"Eval MAE Loss (HF Only): {results['val_mae_loss_hf_only']:.5e}\n")
        
        # Add a new line for separation between different cases
        f.write("\n")
        
def parse_args():
    """
    Parses command-line arguments for setting up the training configuration.
    
    Returns:
        argparse.Namespace: The namespace containing all the command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Multi-Fidelity GNN for Power System Analysis')
    parser.add_argument('--case', type=str, default='case14', help='Name of the power system case (e.g., case118, case14, etc.)')
    parser.add_argument('--input-dir', type=str, default='./data/normal/renewable/case14', help='Input directory where data files are located')
    parser.add_argument('--output-dir', type=str, default= 'results_MF/normal/renewable', help='Parent directory to save the case folder')
    parser.add_argument('--input-dim', type=int, default=5, help='Input dimension of the data')
    parser.add_argument('--hidden-dim-low-fidelity', type=int, default=256, help='Hidden dimension for GNN layers')
    parser.add_argument('--hidden-dim-high-fidelity', type=int, default=64, help='Hidden dimension for GNN layers')
    parser.add_argument('--output-dim', type=int, default=2, help='Output dimension (e.g., 1 for regression or number of classes for classification)')
    parser.add_argument('--learning-rate-low-fidelity', type=float, default=0.0025927123368856184, help='Learning rate for the Low-Fidelity GNN model')
    parser.add_argument('--learning-rate-high-fidelity', type=float, default=0.001, help='Learning rate for the High-Fidelity GNN model')
    parser.add_argument('--weight-decay-low-fidelity', type=float, default=0.00011296845021643265, help='weight decay for the High-Fidelity GNN model')
    parser.add_argument('--num-epochs-low-fidelity', type=int, default=400, help='Number of epochs for training Low-Fidelity GNN')
    parser.add_argument('--num-epochs-high-fidelity', type=int, default=400, help='Number of epochs for training High-Fidelity GNN')
    parser.add_argument('--batch-size-low-fidelity', type=int, default=128, help='Batch size for training')
    parser.add_argument('--batch-size-high-fidelity', type=int, default=64, help='Batch size for training')
    parser.add_argument('--log-file', type=str, default='training_log.txt', help='File to store the training log')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    return args


def configure_logging(log_file):
    """
    Configures logging to file.

    Args:
        log_file (str): Path to the log file where logs should be written.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clear_log_file(log_file):
    """
   Clears the content of the specified log file. This is typically used before starting a new training session 
   to ensure that logs from previous sessions do not accumulate.

   Args:
       log_file (str): Path to the log file to be cleared.
   """
    with open(log_file, 'w'):
        pass


def main(args, N_MF_TRAIN):
    # Calculate the number of training, testing, and validation samples for multi-fidelity, high-fidelity, and low-fidelity data
    N_HF_TRAIN = int(1.2 * N_MF_TRAIN)
    N_LF_TRAIN = int(2 * N_MF_TRAIN)

    N_MF_TEST = int(0.5 * N_MF_TRAIN)
    N_HF_TEST = int(0.5 * N_MF_TRAIN)
    N_LF_TEST = int(0.5 * N_LF_TRAIN)

    N_MF_VAL = int(0.25 * N_MF_TRAIN)
    N_HF_VAL = int(0.25 * N_MF_TRAIN)
    N_LF_VAL = int(0.25 * N_LF_TRAIN)

    # args.output_dir = f'./Sensitivity/results_{N_MF_TRAIN}'
    # Create the case folder with the specified folder name
    case_folder = os.path.join(args.output_dir, args.case)
    os.makedirs(case_folder, exist_ok=True)
    # Clear existing log file or create a new one
    log_file_path = os.path.join(case_folder, f'training_log_embed.txt')
    clear_log_file(log_file_path)

    # Configure logging to write to the log file
    configure_logging(log_file_path)

    logging.info(f"***********************RESULTS FOR NUMBER OF HIGH-FIDELITY DATA = {N_MF_TRAIN}***********************")
    
    # Load the data
    train_data_DC = load_data(os.path.join(args.input_dir, f'{args.case}_train_data_dc.pkl'), args.device)
    validation_data_DC = load_data(os.path.join(args.input_dir, f'{args.case}_val_data_dc.pkl'), args.device)
    test_data_dc = load_data(os.path.join(args.input_dir, f'{args.case}_test_data_dc.pkl'), args.device)
    train_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_train_data_ac.pkl'), args.device)
    validation_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_val_data_ac.pkl'), args.device)
    test_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_test_data_ac.pkl'), args.device)
    
    # Randomly sample data for use in training, validation, and testing
    train_data_ac_mf = random.sample(train_data_AC, N_MF_TRAIN)
    train_data_ac = random.sample(train_data_AC, N_HF_TRAIN)
    train_data_dc = random.sample(train_data_DC, N_LF_TRAIN)

    validation_data_ac = random.sample(validation_data_AC, N_HF_VAL)
    validation_data_dc = random.sample(validation_data_DC, N_LF_VAL)
    test_data_ac = random.sample(test_data_AC, N_HF_TEST)


    # Create Low-Fidelity and High-Fidelity models
    low_fidelity_model = LFGNN(args.input_dim, args.hidden_dim_low_fidelity, args.output_dim - 1, 'TAGConv', 2).to(args.device)
    high_fidelity_model = MFGNN(args.input_dim + args.output_dim, args.hidden_dim_high_fidelity, args.output_dim, args.hidden_dim_low_fidelity).to(args.device)

    # Define optimizers and loss function
    low_fidelity_optimizer = optim.Adam(low_fidelity_model.parameters(), lr=args.learning_rate_low_fidelity, weight_decay=args.weight_decay_low_fidelity)
    high_fidelity_optimizer = optim.Adam(high_fidelity_model.parameters(), lr=args.learning_rate_high_fidelity)
    loss_function = nn.MSELoss()

    # Training the Low-Fidelity GNN model
    logging.info("Training Low-Fidelity GNN...")
    low_fidelity_model_save_path = os.path.join(case_folder, 'LF_best_model.pt')
    train_losses_low_fidelity, val_losses_low_fidelity, train_mae_losses_low_fidelity, val_mae_losses_low_fidelity = train_low_fidelity_model(
        low_fidelity_model, low_fidelity_optimizer, loss_function, train_data_dc, validation_data_dc, args.num_epochs_low_fidelity, args.batch_size_low_fidelity, args.device, low_fidelity_model_save_path)
    
    # Load the best Low-Fidelity model based on validation losses
    low_fidelity_model.load_state_dict(torch.load(low_fidelity_model_save_path))


    # Training the High-Fidelity GNN model using Low-Fidelity predictions
    logging.info("\nTraining Multi-Fidelity GNN...")
    multi_fidelity_model_save_path = os.path.join(case_folder, 'MF_best_model.pt')
    train_losses_v_high_fidelity, val_losses_v_high_fidelity, train_losses_d_high_fidelity, val_losses_d_high_fidelity, train_losses_high_fidelity, val_losses_high_fidelity, train_mae_losses_v_high_fidelity, val_mae_losses_v_high_fidelity, train_mae_losses_d_high_fidelity, val_mae_losses_d_high_fidelity, train_mae_losses_high_fidelity, val_mae_losses_high_fidelity = train_multi_fidelity_model(
        high_fidelity_model, high_fidelity_optimizer, loss_function, train_data_ac_mf, validation_data_ac, args.num_epochs_high_fidelity, args.batch_size_high_fidelity, low_fidelity_model, args.device, multi_fidelity_model_save_path)

    # Load the best High-Fidelity model based on validation losses
    high_fidelity_model.load_state_dict(torch.load(multi_fidelity_model_save_path))

    # Evaluation of the High-Fidelity model on the test set
    logging.info("\nEvaluating Multi-Fidelity GNN on Test Set...")
    test_loss_v_multi_fidelity, test_loss_d_multi_fidelity, test_loss_multi_fidelity, test_mae_loss_v_multi_fidelity, test_mae_loss_d_multi_fidelity, test_mae_loss_multi_fidelity, test_time_multi_fidelity = evaluate_multi_fidelity(high_fidelity_model, test_data_ac, args.batch_size_high_fidelity, low_fidelity_model, args.device)
    logging.info(f"Test MSE Loss - Magnitude: {test_loss_v_multi_fidelity:.5e}")
    logging.info(f"Test MSE Loss - Phase: {test_loss_d_multi_fidelity:.5e}")
    logging.info(f"Test MSE Loss: {test_loss_multi_fidelity:.5e}")
    logging.info(f"Test MAE Loss - Magnitude: {test_mae_loss_v_multi_fidelity:.5e}")
    logging.info(f"Test MAE Loss - Phase: {test_mae_loss_d_multi_fidelity:.5e}")
    logging.info(f"Test MAE Loss: {test_mae_loss_multi_fidelity:.5e}")
    logging.info(f"Test Inference Time: {test_time_multi_fidelity:.5e}")

    # Train the High-Fidelity GNN model using HF data only
    high_fidelity_model_only_hf = HFGNN(args.input_dim, args.hidden_dim_high_fidelity, args.output_dim).to(args.device)
    high_fidelity_optimizer_only_hf = optim.Adam(high_fidelity_model_only_hf.parameters(), lr=args.learning_rate_high_fidelity)

    # Training loop for High-Fidelity GNN using HF data only
    logging.info("\nTraining High-Fidelity GNN (HF data only)...")
    high_fidelity_model_only_hf_save_path = os.path.join(case_folder, 'HF_best_model.pt')
    train_losses_v_high_fidelity_only_hf, val_losses_v_high_fidelity_only_hf, train_losses_d_high_fidelity_only_hf, val_losses_d_high_fidelity_only_hf, train_losses_high_fidelity_only_hf, val_losses_high_fidelity_only_hf, train_mae_losses_v_high_fidelity_only_hf, val_mae_losses_v_high_fidelity_only_hf, train_mae_losses_d_high_fidelity_only_hf, val_mae_losses_d_high_fidelity_only_hf, train_mae_losses_high_fidelity_only_hf, val_mae_losses_high_fidelity_only_hf = train_high_fidelity_model(
        high_fidelity_model_only_hf, high_fidelity_optimizer_only_hf, loss_function, train_data_ac, validation_data_ac, args.num_epochs_high_fidelity, args.batch_size_high_fidelity, args.device, high_fidelity_model_only_hf_save_path)

    # Load the best High-Fidelity model based on validation losses
    high_fidelity_model_only_hf.load_state_dict(torch.load(high_fidelity_model_only_hf_save_path))

    # Evaluation of High-Fidelity GNN model using HF data only on the test set
    logging.info("\nEvaluating High-Fidelity GNN (HF data only) on Test Set...")
    test_loss_v_hf_only, test_loss_d_hf_only, test_loss_hf_only, test_mae_loss_v_hf_only, test_mae_loss_d_hf_only, test_mae_loss_hf_only, test_time_hf_only = evaluate_high_fidelity(high_fidelity_model_only_hf, test_data_ac, args.batch_size_high_fidelity, args.device)
    logging.info(f"Test MSE Loss - Magnitude (HF data only): {test_loss_v_hf_only:.5e}")
    logging.info(f"Test MSE Loss- Phase (HF data only): {test_loss_d_hf_only:.5e}")
    logging.info(f"Test MSE Loss (HF data only): {test_loss_hf_only:.5e}")
    logging.info(f"Test MAE Loss - Magnitude (HF data only): {test_mae_loss_v_hf_only:.5e}")
    logging.info(f"Test MAE Loss - Phase (HF data only): {test_mae_loss_d_hf_only:.5e}")
    logging.info(f"Test MAE Loss (HF data only): {test_mae_loss_hf_only:.5e}")
    logging.info(f"Test Inference Time: {test_time_hf_only:.5e}")


    # Compare the losses
    logging.info("\nComparison of Test MSE Loss - Magnitude:")
    logging.info(f"Multi-Fidelity Approach: {test_loss_v_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_loss_v_hf_only:.5e}")

    logging.info("\nComparison of Test MSE Loss - Phase:")
    logging.info(f"Multi-Fidelity Approach: {test_loss_d_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_loss_d_hf_only:.5e}")

    logging.info("\nComparison of Test MSE Loss:")
    logging.info(f"Multi-Fidelity Approach: {test_loss_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_loss_hf_only:.5e}")

    logging.info("\nComparison of Test MAE Loss - Magnitude:")
    logging.info(f"Multi-Fidelity Approach: {test_mae_loss_v_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_mae_loss_v_hf_only:.5e}")

    logging.info("\nComparison of Test MAE Loss - Phase:")
    logging.info(f"Multi-Fidelity Approach: {test_mae_loss_d_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_mae_loss_d_hf_only:.5e}")

    logging.info("\nComparison of Test MAE Loss:")
    logging.info(f"Multi-Fidelity Approach: {test_mae_loss_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_mae_loss_hf_only:.5e}")
    
    # Compile results into a dictionary
    results = {
    'train_mse_loss_v_multi_fidelity': train_losses_v_high_fidelity[-1],
    'train_mse_loss_d_multi_fidelity': train_losses_d_high_fidelity[-1],
    'train_mse_loss_multi_fidelity': train_losses_high_fidelity[-1],
    'train_mae_loss_v_multi_fidelity': train_mae_losses_v_high_fidelity[-1],
    'train_mae_loss_d_multi_fidelity': train_mae_losses_d_high_fidelity[-1],
    'train_mae_loss_multi_fidelity': train_mae_losses_high_fidelity[-1],
    'val_mse_loss_v_multi_fidelity': val_losses_v_high_fidelity[-1],
    'val_mse_loss_d_multi_fidelity': val_losses_d_high_fidelity[-1],
    'val_mse_loss_multi_fidelity': val_losses_high_fidelity[-1],
    'val_mae_loss_v_multi_fidelity': val_mae_losses_v_high_fidelity[-1],
    'val_mae_loss_d_multi_fidelity': val_mae_losses_d_high_fidelity[-1],
    'val_mae_loss_multi_fidelity': val_mae_losses_high_fidelity[-1],
    'train_mse_loss_v_hf_only': train_losses_v_high_fidelity_only_hf[-1],
    'train_mse_loss_d_hf_only': train_losses_d_high_fidelity_only_hf[-1],
    'train_mse_loss_hf_only': train_losses_high_fidelity_only_hf[-1],
    'train_mae_loss_v_hf_only': train_mae_losses_v_high_fidelity_only_hf[-1],
    'train_mae_loss_d_hf_only': train_mae_losses_d_high_fidelity_only_hf[-1],
    'train_mae_loss_hf_only': train_mae_losses_high_fidelity_only_hf[-1],
    'val_mse_loss_v_hf_only': val_losses_v_high_fidelity_only_hf[-1],
    'val_mse_loss_d_hf_only': val_losses_d_high_fidelity_only_hf[-1],
    'val_mse_loss_hf_only': val_losses_high_fidelity_only_hf[-1],
    'val_mae_loss_v_hf_only': val_mae_losses_v_high_fidelity_only_hf[-1],
    'val_mae_loss_d_hf_only': val_mae_losses_d_high_fidelity_only_hf[-1],
    'val_mae_loss_hf_only': val_mae_losses_high_fidelity_only_hf[-1],
    }
    
    # Write the results to a file
    write_results(args.output_dir, args.case, N_MF_TRAIN, results)

    # Font settings
    label_font = {'fontname': 'Times New Roman', 'size': 18}
    tick_font = {'fontname': 'Times New Roman', 'size': 18}
    legend_font = {'family': 'serif', 'size': 15}
    
    # Plotting for Comparison of Loss Curves
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 24}
    plt.rc('font', **font)
    
    # Plot MSE losses for magnitude
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_v_high_fidelity_only_hf, label='SF-GNN Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_v_high_fidelity_only_hf, label='SF-GNN Validation Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_v_high_fidelity, label='EMF-GNN Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_v_high_fidelity, label='EMF-GNN Validation Loss')
    plt.xlabel('Epoch', **label_font)
    plt.ylabel('Loss', **label_font)
    plt.yscale('log')
    plt.xticks(fontsize=tick_font['size'], fontname=tick_font['fontname'])
    plt.yticks(fontsize=tick_font['size'], fontname=tick_font['fontname'])
    plt.legend(frameon=False, prop=legend_font)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_losses_magnitude.png'), dpi=700)
    plt.close()
    
    # Plot MSE losses for phase
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_d_high_fidelity_only_hf, label='SF-GNN Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_d_high_fidelity_only_hf, label='SF-GNN Validation Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_d_high_fidelity, label='EMF-GNN Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_d_high_fidelity, label='EMF-GNN Validation Loss')
    plt.xlabel('Epoch', **label_font)
    plt.ylabel('Loss', **label_font)
    plt.yscale('log')
    plt.xticks(fontsize=tick_font['size'], fontname=tick_font['fontname'])
    plt.yticks(fontsize=tick_font['size'], fontname=tick_font['fontname'])
    plt.legend(frameon=False, prop=legend_font)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_losses_phase.png'), dpi=300)
    plt.close()
    
    # Plot total MSE losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_high_fidelity_only_hf, label='SF-GNN Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_high_fidelity_only_hf, label='SF-GNN Validation Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_high_fidelity, label='EMG-GNN Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_high_fidelity, label='EMG-GNN Validation Loss')
    plt.xlabel('Epoch', **label_font)
    plt.ylabel('Loss', **label_font)
    plt.yscale('log')
    plt.xticks(fontsize=tick_font['size'], fontname=tick_font['fontname'])
    plt.yticks(fontsize=tick_font['size'], fontname=tick_font['fontname'])
    plt.legend(frameon=False, prop=legend_font)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_mse_losses.png'), dpi=300)
    plt.close()
    
    # Save training and evaluation MSE losses for magnitude
    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_v_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_v_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_d_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_d_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_high_fidelity, f)

    # Save training and evaluation MSE losses for phase for HF data only
    output_data_path = os.path.join(case_folder, 'hf_train_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_v_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_eval_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_v_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_train_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_d_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_eval_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_d_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_train_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_eval_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_high_fidelity_only_hf, f)


    # Visualization of the power system graph with node colors representing the prediction errors
    selected_sample_index = np.random.randint(len(test_data_ac))
    selected_sample = test_data_ac[selected_sample_index]

    x, edge_index = selected_sample.x.to(args.device), selected_sample.edge_index.to(args.device)
    with torch.no_grad():
        low_fidelity_predictions, lf_embedding = low_fidelity_model(x, edge_index)

    low_fidelity_phase = low_fidelity_predictions[:, 0].view(-1, 1)
    low_fidelity_voltage = torch.ones_like(low_fidelity_phase)

    additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1)
    high_fidelity_model.eval()
    with torch.no_grad():
        predicted_output = high_fidelity_model(x, edge_index, additional_x, lf_embedding)

    true_output = selected_sample.y.cpu().numpy()
    errors_v = np.abs(predicted_output[:, 0].cpu().numpy() - true_output[:, 0])
    errors_d = np.abs(predicted_output[:, 1].cpu().numpy() - true_output[:, 1])
    
    # Create a networkx graph for visualization
    graph = nx.Graph()
    for node_idx in range(selected_sample.num_nodes):
        node_value = selected_sample.x[node_idx].cpu().numpy()
        graph.add_node(node_idx, value=node_value)

    for edge_idx in range(selected_sample.edge_index.shape[1]):
        src, tgt = selected_sample.edge_index[:, edge_idx].cpu().numpy()
        graph.add_edge(src, tgt)
    
    # Plot the graph with node colors representing the voltage magnitude errors
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    node_colors = [errors_v[node_idx] for node_idx in graph.nodes()]
    edges = nx.draw_networkx_edges(graph, pos, edge_color='gray')
    nodes = nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap='rainbow', node_size=250, alpha=0.9)
    plt.colorbar(nodes, label='Absolute Error')
    labels = nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')
    plt.axis('off')
    plt.title("Voltage Magnitude")
    plt.savefig(os.path.join(case_folder, f"network_error_magnitude_{selected_sample_index}.png"), dpi=700)
    # Close the plot to release resources
    plt.close()
    
    # Plot the graph with node colors representing the phase angle errors
    plt.figure(figsize=(12, 8))
    node_colors = [errors_d[node_idx] for node_idx in graph.nodes()]
    edges = nx.draw_networkx_edges(graph, pos, edge_color='gray')
    nodes = nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap='rainbow', node_size=250, alpha=0.9)
    plt.colorbar(nodes, label='Absolute Error')
    labels = nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')
    plt.axis('off')
    plt.title("Phase Angle")
    plt.savefig(os.path.join(case_folder, f"network_error_phase_{selected_sample_index}.png"), dpi=700)
    # Close the plot to release resources
    plt.close()



if __name__ == "__main__":
    args = parse_args()
    #for N_MF_TRAIN in range(16000, 3999, -2000):
    main(args, N_MF_TRAIN =8000)
