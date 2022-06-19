# Quiver-Feature
Quiver-Feature is a high performance component for distributed feature collections for GNN training on extreme large graphs. It has **5-10x throughput performance** compared with current solutions in different GNN systems, such as DGL, PyG, GraphLearn.

# Install

## Pip Install
1. Install the Quiver pip package [from here](https://github.com/quiver-team/torch-quiver).

2. Install Quiver-Feature pip package

        pip install quiver-feature 

## Install From Source

        git clone git@github.com:quiver-team/quiver-feature.
        pip install .


We have tested Quiver with the following setup:

        OS: Ubuntu 18.04, Ubuntu 20.04

        CUDA: 10.2, 11.1

        GPU: P100, V100, Titan X, A6000

## Test Install

1. You can download Quiver-Feature's examples to test installation:

        git clone git@github.com:quiver-team/quiver-feature.git
        cd quiver-feature/examples/reddit

2. Start your E2E training using command `python3 distribute_training.py -server_world_size [x] -server_rank [x] -device_per_node [x]`. Here `server_world_size` stands for how many machines you are using. `device_per_node` stands for how many training process you are starting on a single machine. `server_rank` stands for server's rank in server world. Remember to set `MASTER_IP` correctly in `config.py` when you start multi-machine training so that all servers can communicate with each other.

# Core Ideas

