# Learning to Learn Graph Topologies

This is the official code of L2G, Unrolling and Recurrent Unrolling in Learning to Learn Graph Topologies.



## Requirement 

The code has been tested under:

- Python == 3.6.0

- PyTorch >= 1.4.0 ｜ CUDA == 10.1



## Overview

A quick summary of different folders:

- `src/models.py` contains the source code for the proposed L2G and Unrolling.

- `src/baselines.py` contains the source code for the iterative algorithm PDS and ADMM.

- `src/utils.py` contains utility functions. 

- `src/utils_data.py` contains the code for generating synthetic data and graphs.

- `data/` is a folder for datasets.

- `log/` contains training logs.

- `saved_model/` is a folder to store trained models.

- `saved_results/` is a folder to store testing results.

- `data_simulation.py` contains a code snippet of generating synthetic data and graphs. 

- `main_L2G.py` includes the code for training, validating and testing  L2G. 

- `main_Unrolling.py`  includes the code for training, validating and testing Unrolling.

  

## Examples

As there is a requirement on  the maximum file size for submissions, we cannot upload all the experimental results and dataset. However, we include all the source code and some of the results as below.

- Training and testing L2G on scale-free networks, run:

  ```bash
  export PYTHONPATH=$PATHONPATH:'pwd' &&
  python data_simulation.py &&
  python main_L2G.py --graph_type='BA' --n_epochs=100
  ```
  One can find a running log of training and validation loss per epoch at `logs/L2G_BA_m20_x20.log`. The trained model and test results are automatically saved in `saved_model/L2G_BA20_unroll20.pt` and `saved_results/L2G_BA20_unroll20.pt`.

- Training and testing Unrolling (ablation study) on scale-free networks, run:

  ```bash
  export PYTHONPATH=$PATHONPATH:'pwd' &&
  python data_simulation.py &&
  python main_Unrolling.py --graph_type='BA' --n_epochs=100
  ```

- In `L2G_WS_m50_x20.ipynb`, we show a step-by-step example of training and testing L2G on small-world graphs.

For all the above examples, the results are saved in `saved_results/` and the trained models are saved in `saved_model/` .