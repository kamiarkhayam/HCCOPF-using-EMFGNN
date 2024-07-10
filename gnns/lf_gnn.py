# -*- coding: utf-8 -*-
"""
Implementation of the Low-Fidelity Graph Neural Network (LFGNN), adapted and modified from 
"Multi-fidelity Graph Neural Networks for Efficient Power Flow Analysis under High-Dimensional 
Demand and Renewable Generation Uncertainty". 

This model leverages various GNN layers from the PyTorch Geometric library, enabling efficient 
graph processing for applications where high computational efficiency and manageable fidelity are prioritized.

Created by: Kamiar Khayambashi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, GraphConv, GATConv, EdgeConv, SAGEConv, SGConv, APPNP, ChebConv, AGNNConv, GCNConv



class LFGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gnn_type, num_gnn_layers):
        super(LFGNN, self).__init__()

        # Encoder layers
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, hidden_dim)

        # Initialize GNN layers based on the specified type
        if gnn_type == 'TAGConv':
            self.gnn_layers = nn.ModuleList([
                TAGConv(hidden_dim, hidden_dim, k=2) for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'GraphConv':
            self.gnn_layers = nn.ModuleList([
                GraphConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'GATConv':
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=3) for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'EdgeConv':
            self.gnn_layers = nn.ModuleList([
                EdgeConv(nn=nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU()), aggr='mean') for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'GraphSAGE':
            self.gnn_layers = nn.ModuleList([
                SAGEConv(hidden_dim, hidden_dim, aggr='mean') for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'SGConv':
            self.gnn_layers = nn.ModuleList([
                SGConv(hidden_dim, hidden_dim, K=2) for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'APPNP':
            self.gnn_layers = nn.ModuleList([
                APPNP(K=3, alpha=0.5) for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'ChebConv':
            self.gnn_layers = nn.ModuleList([
                ChebConv(hidden_dim, hidden_dim, K=2) for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'AGNNConv':
            self.gnn_layers = nn.ModuleList([
                AGNNConv(requires_grad=False) for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'GCNConv':
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
            ])

        # Decoder for the task
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index):
        # Encoder
        x_encoded = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x_encoded))

        # GNN layers
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))

        # Decoder for the task
        x_decoded = self.decoder(x)

        return x_decoded, x