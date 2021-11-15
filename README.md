## About

**DistMILE: A Distributed Multi-Level Framework for Scalable Graph Embedding**
Yuntian He, Saket Gurukar, Pouya Kousha, Hari Subramoni, Dhabaleswar K. Panda, Srinivasan Parthasarathy
Published in HiPC 2021

***Abstract***:  DistMILE is a Distributed MultI-Level Embedding framework, which leverages a novel shared-memory parallel algorithm for graph coarsening and a distributed training paradigm for embedding refinement. With the advantage of high-performance computing techniques, DistMILE can smoothly scale different base embedding methods over large networks.

Citation information and link to be added.

## Required packages

- Horovod
- TensorFlow
- MVAPICH2-GDR
- mpi4py
- numpy
- scipy
- scikit-learn
- networkx
- gensim (For DeepWalk)
- theano (For NetMF)

## How to Run

Run the main program using the below command:
> mpirun_rsh -np ${NUM_GPU} ${GPUS} ${MPI_ENV} python main.py --data ${DATA} --basic-embed ${EMBED} --batch-size ${BATCH_SIZE} --coarsen-m ${COARSEN_M} --coarse-threshold ${COARSE_THRESHOLD} --workers ${NUM_THREADS} --num-threads ${NUM_THREADS} --coarse-parallel

Arguments and variables:
- `-np ${NUM_GPU}`: Number of machines for use.
- `${GPUS}`: Hostnames of the machines.
- `${MPI_ENV}`: Environmental variables for MPI.  Please use `"MV2_USE_CUDA=1 MV2_SUPPORT_DL=1 MV2_ENABLE_AFFINITY=0 MV2_HOMOGENEOUS_CLUSTER=1"`.
- `--data`: Dataset, located in `dataset/{$DATA}`.
- `--basic-embed`: Base embedding method, located in `base_embed_methods`.
- `--batch-size`: Batch size for distributed learning. Default is `100000`.
- `--coarsen-m`: Coarsen Depth.
- `--coarse-threshold`: Threshold for coarsening a graph in parallel. Denoted as $n_{c}$ in the paper. Default is `10000`.
- `--workers`: # threads for base embedding.
- `--num-threads`: # threads for coarsening and refinement. All threads are used by default.
- `--coarse-parallel`: Leverage the parallel version of graph coarsening.
- `--large`: Only necessary input is sent to TensorFlow model. Not used by default. 

