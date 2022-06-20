[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.org/project/quiver-feature/

<p align="center">
  <img height="150" src="https://github.com/quiver-team/torch-quiver/blob/main/docs/multi_medias/imgs/quiver-logo-min.png" />
</p>

--------------------------------------------------------------------------------

Quiver-Feature is a high performance component for **distributed feature collection** for **training GNN models on extreme large graphs**. 

1. It has neat ideas about **feature data placement** and **data access method** across devices and machines. 

2. **It has great performance**, **5-10x throughput performance** compared with current feature collection solutions in existing GNN systems, such as DGL, PyG. 

3. **It is very easy to use**, so you can easily integrate Quiver-Feature as a component in your GNN training pipeline to speedup your time-consuming feature collection step when training GNN models on extreme large graphs.

--------------------------------------------------------------------------------

`DistTensorPGAS` is the key component Quiver-Feature provides. It places graph feature across devices(CPU DRAM, GPU HBM) and machines, trying to take full advantage of the multi-tier GPU-centric storage layers. During training, `DistTensorPGAS` uses **UVA** for local data access and **RDMA read** for remote data access, achieving E2E zero-copy and CPU/kernel bypass.


# Install

## Install From Source
1. Install the Quiver pip package [from here](https://github.com/quiver-team/torch-quiver).

2. Install Quiver-Feature from source

        git clone git@github.com:quiver-team/quiver-feature.
        pip install .

## Pip Install

Comming soon....

We have tested Quiver with the following setup:

        OS: Ubuntu 18.04, Ubuntu 20.04

        CUDA: 10.2, 11.1

        GPU: P100, V100, Titan X, A6000

## Test Install

1. You can download Quiver-Feature's examples to test installation:

        git clone git@github.com:quiver-team/quiver-feature.git
        cd quiver-feature/examples/reddit
        python3 distribute_training.py -server_world_size [your_value] -server_rank [your_value] -device_per_node [your_value]

2.Here `server_world_size` stands for how many machines you are using. `device_per_node` stands for how many training process you are starting on a single machine. `server_rank` stands for server's rank in server world. Remember to set `MASTER_IP` correctly in `config.py` when you start multi-machine training so that all servers can communicate with each other. You will see logs like this if your test is successful.

        Starting Server With: [TensorEndPoint(server_rank=0, ip='155.198.152.xx', port=3344, range=Range(start=0, end=116482)), TensorEndPoint(server_rank=1, ip='155.198.152.xx', port=3344, range=Range(start=116482, end=232964))]
        Waiting All Servers To Start
        Registering Buffer, Please Wait...
        Buffer Registeration Done! Ready To Receive Connections, Start Your Clients Now
       

# Core Ideas

## Data Placement

## Zero-Copy Data Access