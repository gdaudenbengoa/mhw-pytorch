# ST-MH Forecast: Spatio-Temporal Marine Heatwave Forecasting

Personal project to learn deep learning applied to oceanography, focused on the modeling and forecasting of Marine Heatwaves (MHW) using PyTorch.

## Objectives

- Master spatio-temporal deep learning architectures with PyTorch.
- Work with oceanographic data and generate synthetic Sea Surface Temperature (SST) tensors.
- Implement structured data pipelines (`Dataset` and `DataLoader`).
- Practice Git and GitHub, maintaining a professional workflow.
- Rigorously document the learning process.

## Project Structure & Modules

- `notebooks/`: Step-by-step Jupyter notebooks executed and managed in **VS Code**.
  - `01_oceanic_data_tensors.ipynb`: Synthetic SST generation, sliding window temporal slicing, and PyTorch pipeline (`Dataset` & `DataLoader`).
  - `02_mhw_architecture_baseline.ipynb`: Baseline Convolutional Neural Network (`MHWBaselineCNN`) using 2D spatial convolutions (`nn.Conv2d`) and ReLU activations.
  - `03_mhw_training_loop.ipynb`: End-to-end model training workflow, loss function configuration (`nn.MSELoss`), optimizer setup (`torch.optim.Adam`), and backpropagation loop.
  - `04_mhw_evaluation_metrics.ipynb`: Performance validation and domain-specific MHW metrics.
- `src/`: Modular Python scripts for datasets, models, and training routines.

## Author
Gonzalo Daudén