# Quiver-Feature
Quiver-Feature is a high performance component for distributed feature collections. It is based on quiver.Feature and using RDMA for cross-machine data access. 

Compared with [TensorPipe](https://github.com/pytorch/tensorpipe), Quiver-Feature has 


# Install

## Pip Install
1. Install the Quiver pip package [from here](https://github.com/quiver-team/torch-quiver).
2. Install Quiver-Feature pip package

        pip install quiver-feature 

We have tested Quiver with the following setup:

OS: Ubuntu 18.04, Ubuntu 20.04
CUDA: 10.2, 11.1
GPU: P100, V100, Titan X, A6000

## Test Install

1. You can download Quiver-Feature's examples to test installation:

        git clone git@github.com:quiver-team/quiver-feature.git
        cd quiver-feature/tests/python

2. Change `config.py` and set `MASTER_IP` as your machine's IP address.

3. Start your feature server, and wait for data registeration to complete.

    
        python3 test_DistTensorServer.py
       
    Wait for logs like:

        Buffer Registeration Done! Ready To Receive Connections Start Your Clients Now

4. Start your feature client:
        
        python3 test_DistTensorClient.py

    A successful run should contain the following line:


        Before Collect, Check RegisteredTensor Shape  torch.Size([xxx, xxx])
        Using TLB Optimization
        Begin To Check Result...
        Result Check Passed!, Throughput = xxxxx MB/s


## Install from Source

To build Quiver-Feature python package from source:

        git@github.com:quiver-team/quiver-feature.git
        pip install .

If you want to develop using our C++ APIs or you want to run C++ tests:

        git@github.com:quiver-team/quiver-feature.git
        sh build.sh



## Quick Start

