"""
Implementation of the Multi-Fidelity Graph Neural Network using DC Power Flow (MFDC-GNN), adapted and modified from 
"Multi-fidelity Graph Neural Networks for Efficient Power Flow Analysis under High-Dimensional 
Demand and Renewable Generation Uncertainty". 

The script leverages PyTorch and PyTorch Geometric for model construction and operation, ensuring efficient data handling and computation on graphical data structures.

Created by: Kamiar Khayambashi
"""



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import TAGConv, GraphConv, GATConv, EdgeConv, SAGEConv, SGConv, APPNP, ChebConv, AGNNConv, GCNConv, GINConv
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import logging
import os
import time  # Add the time module for measuring execution time
import random
from torch.nn import L1Loss

def set_random_seeds(seed):
    """
    Sets the seed for randomness to ensure reproducibility across runs.
    Args:
        seed (int): The seed value to use for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
class MFGNN_DC(nn.Module):
    """
    Defines a multi-fidelity GNN model specifically with direct current (DC) approximations input.
    This class implements a sequence of GraphConv layers for processing graph data.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MFGNN_DC, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, additional_x):
        """
        Forward pass through the network.
        Args:
            x (Tensor): Input features.
            edge_index (LongTensor): Graph connectivity in COO format.
            additional_x (Tensor): Low-Fidelity Model (DCPF) input.
        Returns:
            Tensor: Output of the network.
        """
        x = torch.cat([x, additional_x], dim=1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x + additional_x # Residual Connection

def load_data(file_path, device):
    """
    Loads data from a pickle file and moves it to the specified device.
    Args:
        file_path (str): Path to the data file.
        device (torch.device): The device tensors will be transferred to.
    Returns:
        list: Data loaded from the file and moved to the specified device.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return [d.to(device) for d in data]


def train_high_fidelity_model(model, optimizer, loss_function, mae_function, train_data, val_data, num_epochs, batch_size, device, save_path):
    """
    Trains a high-fidelity GNN model using the provided data and parameters.
    Args:
        model (nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        loss_function (callable): Loss function used for training.
        mae_function (callable): Function to calculate mean absolute error.
        train_data (list): Training data.
        val_data (list): Validation data.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size used for training.
        device (torch.device): Device to perform training on.
        save_path (str): Path where the best model state should be saved.
    Returns:
        tuple: Training and validation losses and MAEs over epochs.
    """
    train_losses_v = []
    train_losses_d = []
    train_losses = []
    train_mae_v = []
    train_mae_d = []
    train_mae = []
    val_losses_v = []
    val_losses_d = []
    val_losses = []
    val_mae_v = []
    val_mae_d = []
    val_mae = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()  # Record the start time of each epoch
        total_loss_v = 0.0
        total_loss_d = 0.0
        total_loss = 0.0
        total_mae_v = 0.0
        total_mae_d = 0.0
        total_mae = 0.0
        num_samples = 0

        model.train()
        for batch in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            low_fidelity_predictions = x[:, -1].to(device)

            low_fidelity_phase = low_fidelity_predictions.view(-1, 1)
            low_fidelity_voltage = torch.ones_like(low_fidelity_phase)

            additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1)

            predictions = model(x[:, :-1], edge_index, additional_x)
            
            loss_v = loss_function(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = loss_function(predictions[:, 1], batch.y[:, 1].to(device))
            loss = loss_v + loss_d
            
            mae_v = mae_function(predictions[:, 0], batch.y[:, 0].to(device))
            mae_d = mae_function(predictions[:, 1], batch.y[:, 1].to(device))
            mae = mae_v + mae_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            total_mae_v += mae_v.item() * batch.num_graphs
            total_mae_d += mae_d.item() * batch.num_graphs
            total_mae += mae.item() * batch.num_graphs
            num_samples += batch.num_graphs

        avg_loss_v = total_loss_v / num_samples
        avg_loss_d = total_loss_d / num_samples
        avg_loss = total_loss / num_samples
        avg_mae_v = total_mae_v / num_samples
        avg_mae_d = total_mae_d / num_samples
        avg_mae = total_mae / num_samples
        train_losses_v.append(avg_loss_v)
        train_losses_d.append(avg_loss_d)
        train_losses.append(avg_loss)
        train_mae_v.append(avg_mae_v)
        train_mae_d.append(avg_mae_d)
        train_mae.append(avg_mae)

        validation_loss_v, validation_loss_d, validation_loss, validation_mae_v, validation_mae_d, validation_mae, validation_time = evaluate_high_fidelity(model, val_data, batch_size, device)
        val_losses_v.append(validation_loss_v)
        val_losses_d.append(validation_loss_d)
        val_losses.append(validation_loss)
        val_mae_v.append(validation_mae_v)
        val_mae_d.append(validation_mae_d)
        val_mae.append(validation_mae)

        end_time = time.time()  # Record the end time of each epoch
        training_time = (end_time - start_time) / num_samples  # Calculate the time taken for this epoch
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train MSE Loss: {avg_loss:.4e}, Train MAE Loss: {avg_mae:.4e}, Validation MSE Loss: {validation_loss:.4e}, Validation MAE Loss: {validation_mae:.4e}, Training Time: {training_time:.5e} seconds, Inference Time: {validation_time:.5e} seconds")
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train MSE Loss: {avg_loss:.4e}, Train MAE Loss: {avg_mae:.4e}, Validation MSE Loss: {validation_loss:.4e}, Validation MAE Loss: {validation_mae:.4e}, Training Time: {training_time:.5e} seconds, Inference Time: {validation_time:.5e} seconds")

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

    return train_losses_v, val_losses_v, train_losses_d, val_losses_d, train_losses, val_losses, train_mae_v, val_mae_v, train_mae_d, val_mae_d, train_mae, val_mae

        
def evaluate_high_fidelity(model, data, batch_size, device):
    """
    Evaluates the high-fidelity GNN model on a dataset.
    Args:
        model (nn.Module): The neural network model to evaluate.
        data (list): Dataset for evaluation.
        batch_size (int): Batch size for evaluation.
        device (torch.device): Device to perform evaluation on.
    Returns:
        tuple: Average loss and MAE values for voltage and demand, and the average epoch time.
    """
    model.eval()
    mse_loss_f = nn.MSELoss()
    mae_loss_f = L1Loss()
    total_loss_v = 0.0
    total_loss_d = 0.0
    total_loss = 0.0
    total_mae_v = 0.0
    total_mae_d = 0.0
    total_mae = 0.0
    num_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            low_fidelity_predictions = x[:, -1].to(device)

            low_fidelity_phase = low_fidelity_predictions.view(-1, 1)
            low_fidelity_voltage = torch.ones_like(low_fidelity_phase)

            additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1)

            predictions = model(x[:, :-1], edge_index, additional_x)

            loss_v = mse_loss_f(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = mse_loss_f(predictions[:, 1], batch.y[:, 1].to(device))
            loss = loss_v + loss_d

            mae_v = mae_loss_f(predictions[:, 0], batch.y[:, 0].to(device))
            mae_d = mae_loss_f(predictions[:, 1], batch.y[:, 1].to(device))
            mae = mae_v + mae_d

            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs

            total_mae_v += mae_v.item() * batch.num_graphs
            total_mae_d += mae_d.item() * batch.num_graphs
            total_mae += mae.item() * batch.num_graphs

            num_samples += batch.num_graphs

    end_time = time.time()
    epoch_time = end_time - start_time

    return total_loss_v / num_samples, total_loss_d / num_samples, total_loss / num_samples, total_mae_v / num_samples, total_mae_d / num_samples, total_mae / num_samples, epoch_time / num_samples



def parse_args():
    """
    Parses command-line arguments for setting up the training configuration.
    
    Returns:
        argparse.Namespace: The namespace containing all the command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Multi-Fidelity GNN for Power System Analysis')
    parser.add_argument('--case', type=str, default='case14', help='Name of the power system case (e.g., case118, case14, etc.)')
    parser.add_argument('--input-dir', type=str, default='./data/gen_out/dc_approx/renewable/case14', help='Input directory where data files are located')
    parser.add_argument('--output-dir', type=str, default= 'results_f/gen_out/dc_approx/renewable', help='Parent directory to save the case folder')
    parser.add_argument('--input-dim', type=int, default=7, help='Input dimension of the data')
    parser.add_argument('--hidden-dim-high-fidelity', type=int, default=64, help='Hidden dimension for GNN layers')
    parser.add_argument('--output-dim', type=int, default=2, help='Output dimension (e.g., 1 for regression or number of classes for classification)')
    parser.add_argument('--learning-rate-high-fidelity', type=float, default=0.001, help='Learning rate for the High-Fidelity GNN model')
    parser.add_argument('--num-epochs-high-fidelity', type=int, default=400, help='Number of epochs for training High-Fidelity GNN')
    parser.add_argument('--batch-size-high-fidelity', type=int, default=64, help='Batch size for training')
    parser.add_argument('--log-file', type=str, default='training_log_withoutphy.txt', help='File to store the training log')
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
    Clears the content of the specified log file. This is typically used before starting a new training session to ensure that logs from previous sessions do not accumulate.

    Args:
        log_file (str): Path to the log file to be cleared.
    """
    with open(log_file, 'w'):
        pass
    
# =============================================================================
# Main
# =============================================================================
def main(args, N_MF_TRAIN):
    """
    Main function to train, validate, and test the Multi-Fidelity GNN model for power system analysis.
    
    Args:
        args (Namespace): Command line arguments parsed by argparse.
        N_MF_TRAIN (int): Number of high-fidelity training samples to be used in the model training process.
    """
    # Setup training, testing, and validation data sizes
    N_MF_TRAIN = int(N_MF_TRAIN)
    N_MF_TEST = int(0.5 * N_MF_TRAIN)
    N_MF_VAL = int(0.25 * N_MF_TRAIN)
    
    # Create the case folder with the specified folder name
    case_folder = os.path.join(args.output_dir, args.case)
    os.makedirs(case_folder, exist_ok=True)
    # Clear existing log file or create a new one
    log_file_path = os.path.join(case_folder, f'training_log.txt')
    clear_log_file(log_file_path)

    # Configure logging to write to the log file
    configure_logging(log_file_path)
    
    # Load training, validation, and testing data
    train_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_train_data_ac.pkl'), args.device)
    validation_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_val_data_ac.pkl'), args.device)
    test_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_test_data_ac.pkl'), args.device)
    
    # Randomly sample data for use in training, validation, and testing
    train_data_ac = random.sample(train_data_AC, N_MF_TRAIN)
    validation_data_ac = random.sample(validation_data_AC, N_MF_VAL)
    test_data_ac = random.sample(test_data_AC, N_MF_TEST)
    
    # Define loss functions
    loss_function = nn.MSELoss()
    mae_function = L1Loss()

    # Train the High-Fidelity GNN model using HF data only
    multi_fidelity_model_dc = MFGNN_DC(args.input_dim, args.hidden_dim_high_fidelity, args.output_dim).to(args.device)
    multi_fidelity_optimizer_dc = optim.Adam(multi_fidelity_model_dc.parameters(), lr=args.learning_rate_high_fidelity)

    # Training loop for High-Fidelity GNN using HF data only
    logging.info("\nTraining Multi-Fidelity-DC GNN...")
    multi_fidelity_model_dc_save_path = os.path.join(case_folder, 'MFDC_best_model.pt')
    train_losses_v_multi_fidelity_dc, val_losses_v_multi_fidelity_dc, train_losses_d_multi_fidelity_dc, val_losses_d_multi_fidelity_dc, train_losses_multi_fidelity_dc, val_losses_multi_fidelity_dc, train_mae_v_multi_fidelity_dc, val_mae_v_multi_fidelity_dc, train_mae_d_multi_fidelity_dc, val_mae_d_multi_fidelity_dc, train_mae_multi_fidelity_dc, val_mae_multi_fidelity_dc = train_high_fidelity_model(
        multi_fidelity_model_dc, multi_fidelity_optimizer_dc, loss_function, mae_function, train_data_ac, validation_data_ac, args.num_epochs_high_fidelity, args.batch_size_high_fidelity, args.device, multi_fidelity_model_dc_save_path)

    # Load the best High-Fidelity model based on validation losses
    multi_fidelity_model_dc.load_state_dict(torch.load(multi_fidelity_model_dc_save_path))

    # Evaluation of High-Fidelity GNN model using HF data only on the test set
    logging.info("\nEvaluating Multi-Fidelity-DC GNN on Test Set...")
    test_loss_v_mf_dc, test_loss_d_mf_dc, test_loss_mf_dc, test_mae_v_mf_dc, test_mae_d_mf_dc, test_mae_mf_dc, test_time_mf_dc = evaluate_high_fidelity(
        multi_fidelity_model_dc, test_data_ac, args.batch_size_high_fidelity, args.device)
    logging.info(f"Test MSE Loss - Magnitude: {test_loss_v_mf_dc:.5e}")
    logging.info(f"Test MSE Loss - Phase: {test_loss_d_mf_dc:.5e}")
    logging.info(f"Test MSE Loss: {test_loss_mf_dc:.5e}")
    logging.info(f"Test MAE Loss - Magnitude: {test_mae_v_mf_dc:.5e}")
    logging.info(f"Test MAE Loss - Phase: {test_mae_d_mf_dc:.5e}")
    logging.info(f"Test MAE Loss: {test_mae_mf_dc:.5e}")
    logging.info(f"Test Inference Time: {test_time_mf_dc:.5e}")

    # Plot and save loss graphs for training and validation
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 24}
    plt.rc('font', **font)
    plt.figure(figsize=(12, 7.4))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_v_multi_fidelity_dc, label='Multi-Fidelity-DC Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_v_multi_fidelity_dc, label='Multi-Fidelity-DC Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_losses_magnitude.png'), dpi=700)
    plt.close()

    plt.figure(figsize=(12, 7.4))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_d_multi_fidelity_dc, label='Multi-Fidelity-DC Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_d_multi_fidelity_dc, label='Multi-Fidelity-DC Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_losses_phase.png'), dpi=700)
    plt.close()

    plt.figure(figsize=(12, 7.4))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_multi_fidelity_dc, label='Multi-Fidelity-DC Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_multi_fidelity_dc, label='Multi-Fidelity-DC Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_mse_losses.png'), dpi=700)
    plt.close()

    # Save training and validation losses to pickle files for later analysis
    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_v_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_v_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_d_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_d_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mae_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_mae_v_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mae_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_mae_v_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mae_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_mae_d_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mae_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_mae_d_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mae_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_mae_multi_fidelity_dc, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mae_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_mae_multi_fidelity_dc, f)

    # Visualization of the power system graph with node colors representing the prediction errors
    selected_sample_index = np.random.randint(len(test_data_ac))
    selected_sample = test_data_ac[selected_sample_index]

    # Prepare the data and model for visualization
    x, edge_index = selected_sample.x.to(args.device), selected_sample.edge_index.to(args.device)
    with torch.no_grad():
        multi_fidelity_dc_predictions = multi_fidelity_model_dc(x, edge_index)

    multi_fidelity_phase = multi_fidelity_dc_predictions[:, 1].view(-1, 1)
    multi_fidelity_voltage = multi_fidelity_dc_predictions[:, 0].view(-1, 1)
    
    # Calculate absolute errors in predictions
    true_output = selected_sample.y.cpu().numpy()
    errors_v = np.abs(multi_fidelity_voltage.cpu().numpy() - true_output[:, 0].reshape((-1, 1)))
    errors_d = np.abs(multi_fidelity_phase.cpu().numpy() - true_output[:, 1].reshape((-1, 1)))
    
    # Visualize the network with nodes colored by prediction errors
    graph = nx.Graph()
    for node_idx in range(selected_sample.num_nodes):
        node_value = selected_sample.x[node_idx].cpu().numpy()
        graph.add_node(node_idx, value=node_value)

    for edge_idx in range(selected_sample.edge_index.shape[1]):
        src, tgt = selected_sample.edge_index[:, edge_idx].cpu().numpy()
        graph.add_edge(src, tgt)

    # Visualization for voltage magnitude errors    
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
    plt.close()

    # Repeat visualization for phase angle errors
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
    # for N_MF_TRAIN in range(2000, 20001, 2000):
    main(args, N_MF_TRAIN =8000)
