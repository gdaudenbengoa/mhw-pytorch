# uHEAT: Spatio-Temporal Marine Heatwave Forecasting

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

`uHEAT` is a personal deep learning project aimed at mastering spatio-temporal modeling and forecasting for Marine Heatwaves (MHW) using PyTorch.

## Project Structure
- `notebooks/`: Step-by-step Jupyter notebooks for data engineering, model design, and training.
  - `01_oceanic_data_tensors.ipynb`: Synthetic SST generation, sliding window temporal slicing, and PyTorch `Dataset`/`DataLoader` pipeline.
- `src/`: Modular Python scripts for datasets, models, and training routines.
- `requirements.txt`: Minimal dependencies required for the project.

## Getting Started

1. Clone the repository:
   ```bash
   git clone git@github.com:gdaudenbengoa/uheat-mhw-pytorch.git

   ## Author
Gonzalo Daudén