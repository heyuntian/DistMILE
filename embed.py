import time
import tensorflow as tf
from mpi4py import MPI
import horovod.tensorflow as hvd
# from coarsen import match_and_create_coarse_graph # ,create_coarse_graph
from utils import normalized, graph_to_adj, Timer
from mputils import balancedNodePartition, smart_bcast
# from graph import Subgraph
import numpy as np

import os


def print_coarsen_info(ctrl, g):
    cnt = 0
    while g is not None:
        ctrl.logger.info("Level " + str(cnt) + " --- # nodes: " + str(g.node_num))
        g = g.coarser
        cnt += 1


def multilevel_embed(comm, ctrl, graph, hvd, rank, procs, match_method, basic_embed, refine_model):
    '''This method defines the multilevel embedding method.'''

    if rank == 0:
        start = time.time()
        timer = Timer(logger=ctrl.logger, ident=0)

    ori_graph_size = graph.node_num
    flag_need_coarsen = ctrl.coarsen_to < ori_graph_size
    # Step-1: Graph Coarsening.
    # if ctrl.coarsen_level > 0:
    if flag_need_coarsen:
        if rank == 0:
            original_graph = graph
            # coarsen_level = ctrl.coarsen_level
            # if ctrl.refine_model.double_base:  # if it is double-base, it will need to do one more layer of coarsening
            #     coarsen_level += 1
            timer_coarse_level = Timer(logger=ctrl.logger, ident=1)  # timer for each coarsen level
            # for i in range(coarsen_level):
            #     # get a balance node distribution for parallel
            #     nodePartition = balancedNodePartition(graph.adj_idx, ctrl.num_threads)

            #     # # node matching
            #     # match, coarse_graph_size = match_method(ctrl, graph, nodePartition)
            #     # timer_coarse_level.printIntervalTime('node matching')

            #     # # create a coarser graph
            #     # coarse_graph = create_coarse_graph(ctrl, graph, match, coarse_graph_size)
            #     # graph = coarse_graph
            #     # timer_coarse_level.restart(title='create_coarse_graph', name='coarsen_level = %d'%(i))

            #     coarse_graph = match_method(ctrl, graph, nodePartition)
            #     graph = coarse_graph
            #     timer_coarse_level.printIntervalTime('coarsen_level = %d'%(i))

            #     if graph.node_num <= ctrl.embed_dim:
            #         ctrl.logger.error("Error: coarsened graph contains less than embed_dim nodes.")
            #         exit(0)

            flag_double_base = ctrl.refine_model.double_base
            coarsen_level = 0
            while (graph.node_num > ctrl.coarsen_to or flag_double_base):
                if graph.node_num <= ctrl.coarsen_to:
                    flag_double_base = False
                nodePartition = balancedNodePartition(graph.adj_idx, ctrl.num_threads)
                coarse_graph = match_method(ctrl, graph, nodePartition)
                graph = coarse_graph
                timer_coarse_level.printIntervalTime('coarsen_level = %d'%(coarsen_level))
                coarsen_level += 1

                if graph.node_num <= ctrl.embed_dim:
                    ctrl.logger.error("Error: coarsened graph contains less than embed_dim nodes.")
                    exit(0)

            ctrl.coarsen_level = coarsen_level
            
            if ctrl.debug_mode and graph.node_num < 1e3:
                assert np.allclose(graph_to_adj(graph).A, graph.A.A), "Coarser graph is not consistent with Adj matrix"
            print_coarsen_info(ctrl, original_graph)

            timer.printIntervalTime(name='graph coarsening')
            # assert graph.node_num == 0, "terminate for saving time (we only need coarsening)"
        # else:
        #     # Deprecated. distributed version.
        #     if rank == 0:
        #         timer_coarse = Timer(ident=1)
        #         original_graph = graph

        #     for cr in range(ctrl.coarsen_level):
        #         #  get the initial subgraph
        #         subgraph = Subgraph(graph, procs, rank, comm)
        #         if rank == 0:
        #             timer_coarse.printIntervalTime(name='get subgraph at level %d'%(cr))
        #         #  match
        #         cnvtxs, match = parallel_matching(ctrl, subgraph, procs, rank, comm, cr)
        #         if rank == 0:
        #             timer_coarse.printIntervalTime(name='match at level %d'%(cr))
        #         #  combine match
        #         match = comm.gather(match, root=0)
        #         if rank == 0:
        #             match = np.concatenate(match)
        #             match, coarse_graph_size = getGroupsFromMatch(graph, match)
        #             timer_coarse.printIntervalTime(name='get groups at level %d'%(cr))
        #             #  build coarsen-graph
        #             coarse_graph = create_coarse_graph(ctrl, graph, match, coarse_graph_size)
        #             graph = coarse_graph
        #             timer_coarse.printIntervalTime(name='create coarse graph at level %d'%(cr))
        #         # bcast the graph
        #         graph = comm.bcast(graph, root=0)
        #         if rank == 0:
        #             timer_coarse.printIntervalTime(name='bcast graph at level %d'%(cr))

        #     if rank == 0:
        #         timer.printIntervalTime(name='graph coarsening')
        #         print_coarsen_info(ctrl, original_graph)

        #     comm.Barrier()


    # Step-2 : Base Embedding
    if rank == 0:
        if ctrl.refine_model.double_base:
            graph = graph.finer
        graph_size = graph.node_num
        embedding = basic_embed(ctrl, graph)
        embedding = normalized(embedding, per_feature=False)
        timer.printIntervalTime(name='graph embedding')
    else:
        embedding, graph_size = None, None

    if ctrl.only_coarse:
        return None

    # embedding, graph_size = comm.bcast([embedding, graph_size], root=0)  # broadcast embedding from rank 0
    embedding = smart_bcast(comm, rank, embedding, ctrl.refine_model.gs_mpi_buff, root=0)
    graph_size = comm.bcast(graph_size, root=0)
    
    # if ctrl.coarsen_level > 0:
    if flag_need_coarsen:
        # Step - 3: Embeddings Refinement.
        if rank == 0:
            timer.printIntervalTime(name='broadcast embedding')
            if ctrl.refine_model.double_base:
                coarse_embed = basic_embed(ctrl, graph.coarser)
                coarse_embed = normalized(coarse_embed, per_feature=False)
            else:
                coarse_embed = None
        else:
            coarse_embed = None

        coarse_embed = comm.bcast(coarse_embed, root=0)
        if rank == 0:
            timer.printIntervalTime(name='broadcast coarse_embed (None)')

        # Reset
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        #     tf.config.set_logical_device_configuration(gpu,
        #                                                [tf.config.LogicalDeviceConfiguration(memory_limit=30591)])

        # Pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto(intra_op_parallelism_threads=ctrl.num_threads)
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        iterations = int((graph_size - 1) / procs / ctrl.refine_model.batch_size) + 1
        hooks = [hvd.BroadcastGlobalVariablesHook(0), tf.train.StopAtStepHook(last_step=ctrl.refine_model.epoch * iterations + 50)]
        model = refine_model(comm, ctrl, hvd, rank, procs)
        print('----------rank %d built the model----------'%(rank))

        if rank == 0:
            timer.printIntervalTime('building model')
            timer_phase3 = Timer(logger=ctrl.logger, ident=1)

        with tf.train.MonitoredTrainingSession(config=config, hooks=hooks) as session:
            
            model.train_model(session, coarse_graph=graph.coarser, fine_graph=graph, coarse_embed=coarse_embed,
                              fine_embed=embedding)  # refinement model training

            if rank == 0:
                timer_phase3.printIntervalTime('training refinement model')
                flag_finer = graph.finer is not None
            else:
                flag_finer = None
            flag_finer = comm.bcast(flag_finer, root=0)

            while flag_finer:  # apply the refinement model.
                embedding = model.refine_embedding(session, coarse_graph=graph, fine_graph=graph.finer, coarse_embed=embedding)
                if rank == 0:
                    graph = graph.finer
                    flag_finer = graph.finer is not None
                flag_finer = comm.bcast(flag_finer, root=0)

            if rank == 0:
                timer_phase3.printIntervalTime('refinement')

    if rank == 0:
        end = time.time()
        timer.restart(title='refinement training and applying', name='main program')
        ctrl.embed_time += end - start

    # comm.barrier()
    return embedding
