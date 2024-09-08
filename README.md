# Hybrid Chance-Constrained Optimal Power Flow Using Multi-Fidelity Graph Neural Networks

## Overview
This GitHub repository contains the source code and datasets used in the research paper, "Hybrid Chance-Constrained Optimal Power Flow under Load and Renewable Generation Uncertainty using Multi-Fidelity Graph Neural Networks." The paper introduces a novel Hybrid Chance-Constrained Optimal Power Flow (HCC-OPF) methodology that incorporates an Enhanced Multi-Fidelity Graph Neural Network (EMF-GNN) as a surrogate for power flow calculations. This approach significantly reduces the computational burden of power flow analyses under uncertain conditions while maintaining high accuracy and reliability, especially in handling N-1 security contingencies.

### Key Contributions:
- **Development of EMF-GNN**: A Graph Neural Network that integrates low-fidelity data with high-fidelity simulations to optimize the accuracy and computational efficiency of power flow analyses.
- **Hybrid CC-OPF Framework**: A framework that utilizes GNNs to solve the OPF problem under uncertainty efficiently. It selectively incorporates a deterministic power flow solver to ensure the reliability of the solutions near critical thresholds.


The repository is structured to support researchers and practitioners interested in advanced power system optimization and graph neural network applications.

## Repository Structure
- `gnns/`: Contains surrogate GNN models for power flow analysis.
  - `emf_gnn.py`: Contains classes and functions for training Enhanced Multi-Fidelity GNNs and Single Fidelity GNNs.
  - `lf_gnn.py`: Low-Fidelity GNN models for initial approximations.
  - `mfdc_gnn.py`: Multi-Fidelity GNN using DC power flow input.
  - `data_generation.py`: Contains necessary codes for generating low fidelity and high fidelity data for surrogate model training.
- `hcc_opf/`: Scripts for surrogate-based and normal Chance-Constrained Optimal Power Flow (CC-OPF).
  - `hcc_opf_correlated.py`: Runs HCC-OPF using surrogate models with options for hybrid approaches under uncertainty. It also runs CC-OPF using surrogate models or AC power flow solvers. It can manage contingencies as well.
  - `trial_generation.py`: Utilities for generating trial scenarios, handling correlated uncertainties and system contingencies.
  - `utils.py`: Helper functions for data handling and simulation operations.

## Getting Started
To set up and run the simulations effectively, follow these steps:

1. Ensure Python 3.x is installed on your system.
2. Clone this repository using: `git clone <repository-url>`.
3. Install required Python packages: `pip install -r requirements.txt`.

### Data Preparation
Before running the simulations, you need to prepare the necessary datasets:
1. Generate both low-fidelity and high-fidelity data for your test cases using the script in `gnns/data_generation.py`. This is crucial for training the GNN models.

### Model Training
2. Train the Single Fidelity (SF), Enhanced Multi-Fidelity (EMF), and Multi-Fidelity DC (MFDC) GNN models using the corresponding scripts in the `gnns/` directory:
   - `python gnns/emf_gnn.py` for LF, EMF, and SF models.
   - `python gnns/mfdc_gnn.py` for MFDC models.
   Save the trained models as they will be used in the subsequent analysis.

### Scenario Generation
3. Use `trial_generation.py` in the `hcc_opf/` directory to generate scenarios with different distributions for load, renewable energy, and system contingencies. This step is vital for testing the robustness of the models under various conditions.

### Analysis
4. Adjust the script `hcc_opf/hcc_opf_correlated.py` based on the trained model structures and directories. Specify the model and scenario settings to use.
5. Run HCC-OPF or CC-OPF under uncertainty with or without system contingencies using your chosen surrogate model. Execute the script with:
   - `python hcc_opf/hcc_opf_correlated.py`

## Requirements
- Python 3.x
- PyTorch
- PyTorch Geometric
- NetworkX
- Numpy
- SciPy
- PyPower

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Citation
If you use this code or the framework in your research, please cite our papers:

> Khayambashi, K., Hasnat, M. A., & Alemazkoor, N. (2024). HYBRID CHANCE-CONSTRAINED OPTIMAL POWER FLOW UNDER LOAD AND RENEWABLE GENERATION UNCERTAINTY USING ENHANCED MULTI-FIDELITY GRAPH NEURAL NETWORKS. Journal of Machine Learning for Modeling and Computing, 5(4).

> Taghizadeh, M., Khayambashi, K., Hasnat, M. A., & Alemazkoor, N. (2024). Multi-fidelity graph neural networks for efficient power flow analysis under high-dimensional demand and renewable generation uncertainty. Electric Power Systems Research, 237, 111014.
