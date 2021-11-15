#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from coarsen import match_and_create_coarse_graph
from defs import Control
from embed import multilevel_embed
from eval_embed import eval_multilabel_clf
from refine_model import GCN, GraphSage
from utils import read_graph, setup_custom_logger
import importlib
import logging
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import horovod.tensorflow as hvd
import pymp

import time

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', required=False, default='PPI',
                        help='Input graph file')
    parser.add_argument('--format', required=False, default='metis', choices=['metis', 'edgelist'],
                        help='Format of the input graph file (metis/edgelist)')
    parser.add_argument('--store-embed', action='store_true',
                        help='Store the embeddings.')
    parser.add_argument('--no-eval', action='store_true',
                        help='Evaluate the embeddings.')
    parser.add_argument('--embed-dim', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--basic-embed', required=False, default='netmf',
                        choices=['deepwalk', 'grarep', 'netmf', 'sdne'],
                        help='The basic embedding method. If you added a new embedding method, please add its name to choices')
    parser.add_argument('--refine-type', required=False, default='MD-gs',
                        choices=['MD-gcn', 'MD-dumb', 'MD-gs'],
                        help='The method for refining embeddings.')
    parser.add_argument('--sdne-epoch', default=5, type=int,
                        help='epoch for training SDNE.')
    # parser.add_argument('--coarsen-level', default=2, type=int,
    #                     help='MAX number of levels of coarsening.')
    # parser.add_argument('--coarsen-to', default=500, type=int,
    #                     help='MAX number of nodes in the coarest graph.')
    parser.add_argument('--coarsen-m', default=2, type=int,
                        help='MAX number of levels of coarsening.')
    # parser.add_argument('--batch-per-node', default=1, type=int,
    #                     help='Number of mini batches for training the refinement model')
    parser.add_argument('--batch-size', default=1e5, type=int,
                        help='Batch size for applying the refinement model')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of workers.')
    parser.add_argument('--double-base', action='store_true',
                        help='Use double base for training')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='Learning rate of the refinement model')
    parser.add_argument('--self-weight', default=0.05, type=float,
                        help='Self-loop weight for GCN model.')  # usually in the range [0, 1]
    parser.add_argument('--large', action='store_true',
                        help='Feed only used embeddings to TF network for large graphs.')
    parser.add_argument('--sync-neigh', action='store_true',
                        help='Separately generate neighbors and exchange instead of broadcast all neighbors.')
    parser.add_argument('--only-coarse', action='store_true',
                        help='Only coarse the graph for saving time in experiments.')
    parser.add_argument('--mpi-buff', default=2e9, type=int,
                        help='Buffer size for MPI communication.')
    parser.add_argument('--num-threads', default=pymp.config.num_threads[0], type=int,
                        help='Number of threads used for coarsening')
    parser.add_argument('--coarse-parallel', action='store_true',
                        help='Coarse the graph in parallel.')
    parser.add_argument('--coarse-threshold', default=10000, type=int,
                        help='Minimum graph size for parallelized coarsening.')
    parser.add_argument('--unique-threshold', default=1000000, type=int,
                        help='Minimum graph size for parallelized np.unique in coarsening.')
    parser.add_argument('--refine-threshold', default=2000, type=int,
                        help='Minimum graph size for multi-threaded computation for refinement.')
    parser.add_argument('--min-hash', default=16, type=int,
                        help='Number of min-hashes used for SEM.')
    # Consider increasing self-weight a little bit if coarsen-level is high.
    args = parser.parse_args()
    return args


def set_control_params(ctrl, args, graph):
    ctrl.refine_model.double_base = args.double_base
    ctrl.refine_model.learning_rate = args.learning_rate
    ctrl.refine_model.self_weight = args.self_weight
    ctrl.refine_model.large = args.large
    ctrl.refine_model.sync_neigh = args.sync_neigh
    # ctrl.refine_model.gs_batch_per_node = args.batch_per_node
    ctrl.refine_model.batch_size = args.batch_size
    ctrl.refine_model.refine_threshold = args.refine_threshold
    ctrl.refine_model.gs_mpi_buff = args.mpi_buff
    if args.data == "yelp":
        ctrl.refine_model.gs_sample_neighbrs_num = 10

    # ctrl.coarsen_level = args.coarsen_level
    ctrl.coarsen_k = args.coarsen_m
    ctrl.coarsen_to = max(1, graph.node_num // (2 ** args.coarsen_m))  # rough estimation.
    ctrl.embed_dim = args.embed_dim
    ctrl.basic_embed = args.basic_embed
    ctrl.refine_type = args.refine_type
    ctrl.epoch = args.sdne_epoch
    ctrl.data = args.data
    ctrl.workers = args.workers
    ctrl.max_node_wgt = int((5.0 * graph.node_num) / ctrl.coarsen_to)
    ctrl.coarse_parallel = args.coarse_parallel
    ctrl.coarse_threshold = args.coarse_threshold
    ctrl.unique_threshold = args.unique_threshold
    ctrl.num_threads = min(args.num_threads, pymp.config.num_threads[0])
    ctrl.min_hash = args.min_hash
    ctrl.only_coarse = args.only_coarse

def set_logger(ctrl, args):
    ctrl.logger = setup_custom_logger('MILE')

    if ctrl.debug_mode:
        ctrl.logger.setLevel(logging.DEBUG)
    else:
        ctrl.logger.setLevel(logging.INFO)
    ctrl.logger.info(args)

def read_data(ctrl, args):
    prefix = "./dataset/" + args.data
    ctrl.edgelist = prefix + ".edgelist"
    if args.format == "metis":
        input_graph_path = prefix + ".metis"
        graph, mapping = read_graph(ctrl, input_graph_path, metis=True)
    else:
        input_graph_path = prefix + ".edgelist"
        graph, mapping = read_graph(ctrl, input_graph_path, edgelist=True)

    return input_graph_path, graph, mapping


def select_base_embed(ctrl):
    mod_path = "base_embed_methods." + ctrl.basic_embed
    embed_mod = importlib.import_module(mod_path)
    embed_func = getattr(embed_mod, ctrl.basic_embed)
    return embed_func


def select_refine_model(ctrl):
    refine_model = None
    if ctrl.refine_type == 'MD-gcn':
        refine_model = GCN
    elif ctrl.refine_type == 'MD-gs':
        refine_model = GraphSage
    elif ctrl.refine_type == 'MD-dumb':
        refine_model = GCN
        ctrl.refine_model.untrained_model = True
    return refine_model

# def select_match_model(ctrl):
#     match_model = None
#     if ctrl.coarse_parallel:
#         match_model = generate_HEM_parallel
#     else:
#         match_model = generate_hybrid_matching
#     return match_model

def evaluate_embeddings(input_graph_path, mapping, embeddings):
    truth_mat = np.loadtxt(input_graph_path + ".truth").astype(int)  # truth before remapping
    idx_arr = truth_mat[:, 0].reshape(-1)  # this is the original index
    raw_truth = truth_mat[:, 1:]  # multi-class result
    if mapping is not None:
        idx_arr = [mapping.old2new[idx] for idx in idx_arr]
    if args.format == "metis":
        idx_arr = [idx - 1 for idx in idx_arr]  # -1 due to METIS (starts from 1)
    embeddings = embeddings[idx_arr, :]  # in the case of yelp, only a fraction of data contains label.
    truth = raw_truth
    res = eval_multilabel_clf(ctrl, embeddings, truth)
    print(res)


def store_embeddings(input_graph_path, mapping, embeddings, node_num):
    prefix = "./dataset/" + args.data
    is_metis = (args.format == "metis")
    idx_arr = range(node_num)
    if mapping is not None:
        idx_arr = [mapping.new2old[idx] for idx in idx_arr]
    if is_metis:
        idx_arr = [idx + 1 for idx in idx_arr]  # METIS starts from 1.
    out_file = open(prefix + ".embeddings", "w")
    for i, node_idx in enumerate(idx_arr):
        print >> out_file, str(node_idx) + " " + " ".join(["%.6f" % val for val in embeddings[i]])
    out_file.close()


if __name__ == "__main__":
    # Initialize Horovod
    comm = MPI.COMM_WORLD
    name = MPI.Get_processor_name()
    hvd.init(comm)
    rank = hvd.rank()
    procs = hvd.size()
    
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Initialize
    if rank == 0:
        ctrl = Control()
        args = parse_args()

        # Read input graph
        input_graph_path, graph, mapping = read_data(ctrl, args)
        set_control_params(ctrl, args, graph)

        # Coarsen method
        match_method = match_and_create_coarse_graph # select_match_model(ctrl)

        # Base embedding
        basic_embed = select_base_embed(ctrl)

        # Refinement model
        refine_model = select_refine_model(ctrl)
    else:
        ctrl, graph, match_method, basic_embed, refine_model = None, None, None, None, None

    ctrl, graph, match_method, basic_embed, refine_model = comm.bcast([ctrl, graph, match_method, basic_embed, refine_model], root=0)
    print("Rank %d / %d: %s starts"%(rank, procs, name))
    if rank == 0:
        set_logger(ctrl, args)

    # Generate embeddings
    embeddings = multilevel_embed(comm, ctrl, graph, hvd, rank, procs, match_method=match_method, basic_embed=basic_embed,
                                  refine_model=refine_model)

    if rank == 0 and not ctrl.only_coarse:
        # Evaluate embeddings
        if not args.no_eval:
            evaluate_embeddings(input_graph_path, mapping, embeddings)

        # Store embeddings
        if args.store_embed:
            store_embeddings(input_graph_path, mapping, embeddings, graph.node_num)
