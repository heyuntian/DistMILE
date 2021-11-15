from collections import defaultdict
from graph import Graph
# import cupy as cp
# import cupyx.scipy.sparse as cpx
# import tensorflow as tf
import numpy as np
from utils import cmap2C, Timer
import time
import pymp
from mputils import Barrier, bNP_degree

# import threading


def normalized_adj_wgt(ctrl, graph, nodePartition):
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    degree = graph.degree
    if ctrl.coarse_parallel:
        norm_wgt = pymp.shared.array(adj_wgt.shape, dtype='float32')
        num_threads = ctrl.num_threads
        with pymp.Parallel(num_threads) as p:
            # for i in p.range(graph.node_num):
            tid = p.thread_num
            # node_st, node_ed = nodePartition[tid], nodePartition[tid + 1]
            # for i in range(node_st, node_ed):  # Option 1: Thread %tid gets N/num_threads * [tid, tid+1)
            for i in range(tid, graph.node_num, num_threads):
                deg_i = degree[i]
                for j in range(adj_idx[i], adj_idx[i + 1]):
                    neigh = graph.adj_list[j]
                    norm_wgt[j] = adj_wgt[neigh] / np.sqrt(deg_i * degree[neigh])
    else:
        norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
        for i in range(graph.node_num):
            deg_i = degree[i]
            for j in range(adj_idx[i], adj_idx[i + 1]):
                neigh = graph.adj_list[j]
                norm_wgt[j] = adj_wgt[neigh] / np.sqrt(deg_i * degree[neigh])
    return norm_wgt

# def threading_adj_wgt(graph, st, ed, norm_wgt):
#     adj_wgt = graph.adj_wgt
#     adj_idx = graph.adj_idx
#     degree = graph.degree
#     for i in range(st, ed):
#         deg_i = degree[i]
#         for j in range(adj_idx[i], adj_idx[i + 1]):
#             neigh = graph.adj_list[j]
#             norm_wgt[j] = adj_wgt[neigh] / np.sqrt(deg_i * degree[neigh])
#     return

# class tfMatMul:

#     def __init__(self):
#         self.model_dict = dict()
#         # self.build_graph()

#     # def build_graph(self):
#     #     # x = tf.compat.v1.sparse.placeholder(tf.float32)
#     #     # y = tf.compat.v1.sparse.placeholder(tf.float32)
#     #     # mul = tf.compat.v1.sparse_matmul(x, y, a_is_sparse=True, b_is_sparse=True, name=None)

#     #     x = tf.compat.v1.sparse_placeholder(tf.int32)
#     #     y = tf.compat.v1.sparse_placeholder(tf.int32)
#     #     mul = tf.compat.v1.matmul(tf.sparse.to_dense(x, 0), tf.sparse.to_dense(y, 0), a_is_sparse=True, b_is_sparse=True, name=None)
#     #     self.model_dict['x'] = x
#     #     self.model_dict['y'] = y
#     #     self.model_dict['mul'] = mul

#     def matmul(self, x_cpu, y_cpu, a_is_sparse=True, b_is_sparse=True, name=None):

#         # ini_st = time.time()
#         # x_ts = self.convert_sparse_matrix_to_sparse_tensor(x_cpu)
#         # y_ts = self.convert_sparse_matrix_to_sparse_tensor(y_cpu)
#         # ini_ed = time.time()
#         # print("initial time: %.3f"%(ini_ed - ini_st))
#         t = None
#         with tf.compat.v1.Session() as sess:
#             # t = sess.run(self.model_dict['mul'], feed_dict={
#             #     self.model_dict['x']: x_ts,
#             #     self.model_dict['y']: y_ts
#             #     })
#             init = tf.global_variables_initializer()
#             sess.run(init)
#             # t = sess.run(self.model_dict['mul'], feed_dict={
#             #     self.model_dict['x']: self.convert_sparse_matrix_to_sparse_tensor(x_cpu),
#             #     self.model_dict['y']: self.convert_sparse_matrix_to_sparse_tensor(y_cpu)
#             #     })

#             x = self.convert_sparse_matrix_to_sparse_tensor(x_cpu) if a_is_sparse else x_cpu
#             y = self.convert_sparse_matrix_to_sparse_tensor(y_cpu) if b_is_sparse else y_cpu
#             mul = tf.linalg.matmul(tf.sparse.to_dense(x, 0) if a_is_sparse else x,
#                                 tf.sparse.to_dense(y, 0) if b_is_sparse else y, 
#                                 a_is_sparse=a_is_sparse, b_is_sparse=b_is_sparse, name=name)
#             t = sess.run(mul)
#         return t

#     def convert_sparse_matrix_to_sparse_tensor(self, X):
#         print('----------\n')
#         print(X.shape)
#         coo = X.tocoo()
#         indices = np.mat([coo.row, coo.col]).transpose()
#         return tf.sparse.reorder(tf.compat.v1.SparseTensor(indices, coo.data, coo.shape))

# cls_matmul = tfMatMul()

# def convert_sparse_matrix_to_sparse_tensor(X):
#     coo = X.tocoo()
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return tf.SparseTensor(indices, coo.data, coo.shape)

# def tf_Sparse_MatMul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=True, b_is_sparse=True, name=None):
#     ini_st = time.time()
#     a_ts = convert_sparse_matrix_to_sparse_tensor(a)
#     b_ts = convert_sparse_matrix_to_sparse_tensor(b)
#     ini_ed = time.time()
#     print("initial time: %.3f"%(ini_ed - ini_st))
#     dot_tf = tf.sparse_matmul(a_tf, b_tf, 
#                     transpose_a=transpose_a, 
#                     transpose_b=transpose_b, 
#                     a_is_sparse=a_is_sparse, 
#                     b_is_sparse=b_is_sparse, 
#                     name=name)
#     result = None
#     with tf.Session() as sess:
#         result = sess.run(dot_tf, feed_dict={a_tf})
    
#     return result

def match_and_create_coarse_graph(ctrl, graph, nodePartition):
    '''Generate matchings using the hybrid method. It changes the cmap in graph object, 
    return groups array and coarse_graph_size.'''
    timer = Timer(logger=ctrl.logger, ident=2)
    node_num = graph.node_num
    edge_num = graph.edge_num
    adj_list = graph.adj_list  # big array for neighbors.
    adj_idx = graph.adj_idx  # beginning idx of neighbors.
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node

    # compute norm_adj_wgt
    norm_adj_wgt = normalized_adj_wgt(ctrl, graph, nodePartition)
    timer.printIntervalTime('norm_adj_wgt')

    # initialize before matching
    enable_parallel = ctrl.coarse_parallel and graph.node_num > ctrl.coarse_threshold
    max_node_wgt = ctrl.max_node_wgt
    groups = []  # a list of groups, each group corresponding to one coarse node.
    if enable_parallel:
        matched = pymp.shared.array((node_num,), dtype='int32')
        matched[:] = -1
        MATCHED_SEM = 0x40000000L  # --> INFTY
    else:
        matched = [False] * node_num
    timer.printIntervalTime('initialize before SEM')

    # SEM: structural equivalence matching.
    jaccard_idx_preprocess(ctrl, graph, matched, groups, matched_sem=MATCHED_SEM if enable_parallel else True, nodePartition=nodePartition)
    len_SEM_groups = len(groups)
    ctrl.logger.info("# groups have perfect jaccard idx (1.0): %d" % len_SEM_groups)
    timer.printIntervalTime('SEM')

    # Heavy-edge Match and create the coarser graph
    coarse_graph_size = 0
    if not enable_parallel:
        # attempt matching
        degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]
        sorted_idx = np.argsort(degree)
        for idx in sorted_idx:
            if matched[idx]:
                continue
            max_idx = idx
            max_wgt = -1
            for j in range(adj_idx[idx], adj_idx[idx + 1]):
                neigh = adj_list[j]
                if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                    continue
                curr_wgt = norm_adj_wgt[j]
                if ((not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
                    max_idx = neigh
                    max_wgt = curr_wgt
            # it might happen that max_idx is idx, which means cannot find a match for the node. 
            matched[idx] = matched[max_idx] = True
            if idx == max_idx:
                groups.append([idx])
            else:
                groups.append([idx, max_idx])
        timer.printIntervalTime('attempt matching (serial)')

        # create cmap
        cmap = graph.cmap = np.zeros(node_num, dtype=np.int32) - 1
        coarse_graph_size = 0
        for idx in range(len(groups)):
            for ele in groups[idx]:
                cmap[ele] = coarse_graph_size
            coarse_graph_size += 1
        timer.printIntervalTime('create cmap (serial)')

        # initialize a coarser graph
        coarse_graph = Graph(coarse_graph_size, edge_num)
        coarse_graph.finer = graph
        graph.coarser = coarse_graph
        timer.printIntervalTime('initialize graph')

        # data reference
        coarse_adj_list = coarse_graph.adj_list
        coarse_adj_idx = coarse_graph.adj_idx
        coarse_adj_wgt = coarse_graph.adj_wgt
        coarse_node_wgt = coarse_graph.node_wgt
        coarse_degree = coarse_graph.degree
        timer.printIntervalTime('data reference')

        # add edges to the coarser graph
        coarse_adj_idx[0] = 0
        nedges = 0  # number of edges in the coarse graph
        for idx in range(len(groups)):  # idx in the graph
            coarse_node_idx = idx
            neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
            group = groups[idx]
            for i in range(len(group)):
                merged_node = group[i]
                if (i == 0):
                    coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
                else:
                    coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

                istart = adj_idx[merged_node]
                iend = adj_idx[merged_node + 1]
                for j in range(istart, iend):
                    k = cmap[adj_list[
                        j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                    if k not in neigh_dict:  # add new neigh
                        coarse_adj_list[nedges] = k
                        coarse_adj_wgt[nedges] = adj_wgt[j]
                        neigh_dict[k] = nedges
                        nedges += 1
                    else:  # increase weight to the existing neigh
                        coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                    # add weights to the degree. For now, we retain the loop. 
                    coarse_degree[coarse_node_idx] += adj_wgt[j]

            coarse_node_idx += 1
            coarse_adj_idx[coarse_node_idx] = nedges
        timer.printIntervalTime('add edges (serial)')
    else:
        # initialize
        num_threads = ctrl.num_threads # min(16, ctrl.num_threads)
        num_groups = pymp.shared.array((num_threads,), dtype='int32')  # NUMBER OF GROUPS ASSIGNED TO EACH THREAD
        num_nodes_per_thread = pymp.shared.array((num_threads,), dtype='int32')
        # unmatched_nodes = pymp.shared.array((node_num,), dtype='bool')
        barrier1 = Barrier(num_threads)
        barrier2 = Barrier(num_threads)
        barrier3 = Barrier(num_threads)
        barrier4 = Barrier(num_threads)
        # barrier5 = Barrier(num_threads)
        timer.printIntervalTime('Initialize for mt')

        # initialize the coarser graph
        cmap = graph.cmap = pymp.shared.array((node_num,), dtype='int32')
        coarse_graph = Graph(0, 0)  # the only difference
        coarse_graph.cmap = None
        coarse_graph.finer = graph
        graph.coarser = coarse_graph
        th_nedges = pymp.shared.array((num_threads,), dtype='int32')
        nedges = 0
        timer.printIntervalTime('initialize graph')

        # data reference, use node_num to replace the unknown coarse_graph_size
        coarse_adj_idx = pymp.shared.array((node_num + 1,), dtype='int32')
        coarse_node_wgt = pymp.shared.array((node_num,), dtype='float32')
        coarse_degree = pymp.shared.array((node_num,), dtype='float32')
        coarse_adj_list = pymp.shared.array((edge_num,), dtype='int32')
        coarse_adj_wgt = pymp.shared.array((edge_num,), dtype='float32')
        timer.printIntervalTime('data reference')

        # degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]
        # sorted_idx_global = np.argsort(degree)

        # start multi-thread
        with pymp.Parallel(num_threads) as p:
            tid = p.thread_num
            if tid == 0:
                timer_match = Timer(logger=ctrl.logger, ident=3)
            # # strategy 1: divide nodes into consecutive chunks, then sort
            # node_st, node_ed = nodePartition[tid], nodePartition[tid + 1]
            # degree = [adj_idx[i + 1] - adj_idx[i] for i in range(node_st, node_ed)]
            # sorted_idx = np.argsort(degree) + node_st

            # strategy 2: divide nodes into interleaving chunks, then sort (tested: faster than first)
            degree = [adj_idx[i + 1] - adj_idx[i] for i in range(tid, node_num, num_threads)]
            sorted_idx = np.argsort(degree) * num_threads + tid

            # # strategy 3: sort all nodes then divide into interleaving chunks
            # sorted_idx = sorted_idx_global[range(tid, node_num, num_threads)]        

            # attempt match
            # idx_sorted_idx = 0
            # while idx_sorted_idx < len(sorted_idx):
            for idx in sorted_idx:
                # idx = sorted_idx[idx_sorted_idx]
                if matched[idx] >= 0:
                    # idx_sorted_idx += 1
                    continue
                max_idx = idx
                # snd_max_idx = idx
                max_wgt = -1
                # snd_max_wgt = -1
                for j in range(adj_idx[idx], adj_idx[idx + 1]):
                    neigh = adj_list[j]
                    if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                        continue
                    curr_wgt = norm_adj_wgt[j]
                    if ((matched[neigh] < 0) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
                        max_idx = neigh
                        max_wgt = curr_wgt
                    # if ((matched[neigh] < 0 or matched[matched[neigh]] != neigh) and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
                    #     if max_wgt < curr_wgt:
                    #         snd_max_idx, snd_max_wgt = max_idx, max_wgt
                    #         max_idx, max_wgt = neigh, curr_wgt
                    #     elif snd_max_wgt < curr_wgt:
                    #         snd_max_idx, snd_max_wgt = neigh, curr_wgt
                # if max_idx == idx:
                #     idx_sorted_idx += 1
                #     matched[idx] = idx
                #     continue
                if matched[idx] < 0 and matched[max_idx] < 0:
                    # idx_sorted_idx += 1
                    if idx < max_idx:
                        matched[idx] = max_idx
                        matched[max_idx] = idx
                    else:
                        matched[max_idx] = idx
                        matched[idx] = max_idx
                # else:
                #     continue
                # if matched[idx] < 0:
                #     if (matched[max_idx] < 0 or matched[matched[max_idx]] != max_idx):
                #         if idx < max_idx:
                #             matched[idx] = max_idx
                #             matched[max_idx] = idx
                #         else:
                #             matched[max_idx] = idx
                #             matched[idx] = max_idx
                #     elif (matched[snd_max_idx] < 0 or matched[matched[snd_max_idx]] != snd_max_idx):
                #         if idx < snd_max_idx:
                #             matched[idx] = snd_max_idx
                #             matched[snd_max_idx] = idx
                #         else:
                #             matched[snd_max_idx] = idx
                #             matched[idx] = snd_max_idx
            barrier1.wait()
            if tid == 0:
                timer_match.printIntervalTime('match and barrier')

            # write local groups
            groups_local = []
            for idx_SEM_group in range(tid, len_SEM_groups, num_threads):
                groups_local.append(groups[idx_SEM_group])  # SEM groups assigned to this thread
            if tid == 0:
                timer_match.printIntervalTime('write into groups_local - SEM')



            """
            Second round matching for unmatched nodes
            """
            # unmatched_nodes_local = list()
            # for idx in sorted_idx:
            #     max_idx = matched[idx]
            #     if max_idx == MATCHED_SEM:  # already matched in SEM
            #         continue
            #     # if (matched[max_idx] != idx or max_idx == idx): 
            #     if max_idx == idx:        # match with itself
            #         # groups_local.append([idx])
            #         continue
            #     if matched[max_idx] != idx:
            #         unmatched_nodes_local.append(idx)
            #         unmatched_nodes[idx] = True
            #         continue
            #     # matched[idx] != MATCHED_SEM
            #     # matched[idx] != idx
            #     # matched[matched[idx]] == idx  <- correct match
            #     # if max_idx > idx: # Match only once!
            #     #     groups_local.append([idx, max_idx])
            # barrier5.wait()

            # # pair the unmatched nodes
            # for idx in unmatched_nodes_local:
            #     if matched[idx] >= 0:
            #         continue
            #     max_idx = idx
            #     max_wgt = -1
            #     for j in range(adj_idx[idx], adj_idx[idx + 1]):
            #         if unmatched_nodes[adj_list[j]]:
            #             neigh = adj_list[j]
            #             if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
            #                 continue
            #             curr_wgt = norm_adj_wgt[j]
            #             if ((matched[neigh] < 0) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
            #                 max_idx = neigh
            #                 max_wgt = curr_wgt
            #     if matched[idx] < 0 and matched[max_idx] < 0:
            #         if idx < max_idx:
            #             matched[idx] = max_idx
            #             matched[max_idx] = idx
            #         else:
            #             matched[max_idx] = idx
            #             matched[idx] = max_idx
            """
            """

            # last_unmatched_idx = -1
            for idx in sorted_idx:
                max_idx = matched[idx]
                if max_idx == MATCHED_SEM:  # already matched in SEM
                    continue
                if (max_idx == idx or matched[max_idx] != idx): 
                    groups_local.append([idx])
                    # if last_unmatched_idx == -1:
                    #     last_unmatched_idx = idx
                    # else:
                    #     groups_local.append([last_unmatched_idx, idx])
                    #     last_unmatched_idx = -1
                    # print("idx %d m[idx] %d m[m[idx]] %d degs %d %d %d"%(idx, max_idx, matched[max_idx], adj_idx[idx + 1] - adj_idx[idx], adj_idx[max_idx + 1] - adj_idx[max_idx], adj_idx[matched[max_idx] + 1] - adj_idx[matched[max_idx]]))
                    continue
                # matched[idx] != MATCHED_SEM
                # matched[idx] != idx
                # matched[matched[idx]] == idx  <- correct match
                if max_idx > idx: # Match only once!
                    groups_local.append([idx, max_idx])            
            # if last_unmatched_idx >= 0:
            #     groups_local.append([last_unmatched_idx])

            num_local_groups = num_groups[tid] = len(groups_local)
            barrier2.wait()
            if tid == 0:
                coarse_graph_size = num_groups.sum()
                # print("num_groups %s coarse_graph_size %d"%(num_groups, coarse_graph_size))
                timer_match.printIntervalTime('write into groups_local - HEM')

            # create cmap
            idx_st = num_groups[:tid].sum()
            idx_ed = idx_st + num_local_groups
            count_local_idx = idx_st
            count_nodes = 0
            for idx in range(num_local_groups):
                for ele in groups_local[idx]:
                    cmap[ele] = count_local_idx
                    count_nodes += 1
                count_local_idx += 1
            num_nodes_per_thread[tid] = count_nodes
            barrier3.wait()
            if tid == 0:
                timer_match.printIntervalTime('create cmap')

            local_nedges = 0
            local_coarse_adj_list = []  # don't know edge size until we count
            local_coarse_adj_wgt = []

            # add edges to coarse graph
            for idx in range(num_local_groups):
                coarse_node_idx = idx + idx_st
                coarse_adj_idx[coarse_node_idx] = local_nedges
                neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
                group = groups_local[idx]
                for i in range(len(group)):
                    merged_node = group[i]
                    if (i == 0):
                        coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
                    else:
                        coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

                    istart = adj_idx[merged_node]
                    iend = adj_idx[merged_node + 1]
                    for j in range(istart, iend):
                        k = cmap[adj_list[
                            j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                        if k not in neigh_dict:  # add new neigh
                            local_coarse_adj_list.append(k)
                            local_coarse_adj_wgt.append(adj_wgt[j])
                            neigh_dict[k] = local_nedges
                            local_nedges += 1
                        else:  # increase weight to the existing neigh
                            local_coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                        # add weights to the degree. For now, we retain the loop. 
                        coarse_degree[coarse_node_idx] += adj_wgt[j]

            th_nedges[tid] = local_nedges
            barrier4.wait()
            if tid == 0:
                timer_match.printIntervalTime('add edges (parallel)')
                # print("num_nodes_per_thread %s\nsum %d"%(num_nodes_per_thread, num_nodes_per_thread.sum()))
                nedges = th_nedges.sum()
                # print("th_nedges %s\nsum %d"%(th_nedges, nedges))
                coarse_adj_idx[coarse_graph_size] = nedges

            edge_st = th_nedges[:tid].sum()
            edge_ed = th_nedges[:tid+1].sum()
            assert local_nedges == edge_ed - edge_st, "thread %d edge_num error"%(tid)
            coarse_adj_idx[idx_st:idx_ed] += edge_st
            coarse_adj_list[edge_st:edge_ed] = local_coarse_adj_list
            # print("edge_ed %d edge_st %d edge_ed - edge_st %d len(list) %d len(wgt) %d list.shape %s wgt.shape %s"%(edge_ed, edge_st, edge_ed - edge_st, len(local_coarse_adj_list), len(local_coarse_adj_wgt), coarse_adj_list.shape, coarse_adj_wgt.shape))
            coarse_adj_wgt[edge_st:edge_ed] = local_coarse_adj_wgt
            if tid == 0:
                timer_match.printIntervalTime('merge adj arrays')

        timer.printIntervalTime('match and create coarser graph (in parallel)')           
        # resize
        nedges = th_nedges.sum()
        coarse_graph_size = num_groups.sum()
        coarse_graph.node_num = coarse_graph_size
        coarse_graph.adj_idx = np.resize(coarse_adj_idx, coarse_graph_size + 1)
        coarse_graph.node_wgt = np.resize(coarse_node_wgt, coarse_graph_size)
        coarse_graph.degree = np.resize(coarse_degree, coarse_graph_size)
        coarse_graph.adj_list = coarse_adj_list
        coarse_graph.adj_wgt = coarse_adj_wgt
        timer.printIntervalTime('update coarser graph') 


    # Print coarser graph's statistics and compute C and A
    coarse_graph.edge_num = nedges
    coarse_graph.resize_adj(nedges)
    ctrl.logger.info("node_num %d edge_num %d"%(coarse_graph.node_num, coarse_graph.edge_num))
    ctrl.logger.info("coarse_graph.adj_idx %s"%(coarse_graph.adj_idx.shape))
    ctrl.logger.info("coarse_graph.node_wgt %s"%(coarse_graph.node_wgt.shape))
    ctrl.logger.info("coarse_graph.degree %s"%(coarse_graph.degree.shape))
    ctrl.logger.info("coarse_graph.adj_list %s"%(coarse_graph.adj_list.shape))
    ctrl.logger.info("coarse_graph.adj_wgt %s"%(coarse_graph.adj_wgt.shape))
    # print("coarse_graph.cmap %s"%(coarse_graph.cmap.shape))
    C = cmap2C(cmap, enable_parallel, coarse_graph.node_num, num_threads=min(8, ctrl.num_threads))  # construct the matching matrix.
    graph.C = C
    timer.printIntervalTime('finish - C')
    # coarse_graph.A = C.transpose().dot(graph.A).dot(C)
    # CA = tf_Sparse_MatMul(C, graph.A, transpose_a=True, name=None)
    # CA = cls_matmul.matmul(C.transpose(), graph.A)
    # coarse_graph.A = cls_matmul.matmul(CA, C, a_is_sparse=False)
    # C_gpu = cpx.csr_matrix(C, dtype=np.float32)
    # A_gpu = cpx.csr_matrix(graph.A, dtype=np.float32)
    # timer.printIntervalTime('finish - convert C and A to cpx.csr_matrix')
    # coarse_graph.A = C_gpu.transpose().dot(A_gpu).dot(C_gpu)
    # timer.printIntervalTime('finish - A (on gpu)')
    # coarse_graph.A = coarse_graph.A.get()
    timer.printIntervalTime('finish - A')
    return coarse_graph


# def generate_hybrid_matching(ctrl, graph, nodePartition):
#     '''Generate matchings using the hybrid method. It changes the cmap in graph object, 
#     return groups array and coarse_graph_size.'''
#     timer = Timer(ident=2)
#     node_num = graph.node_num
#     adj_list = graph.adj_list  # big array for neighbors.
#     adj_idx = graph.adj_idx  # beginning idx of neighbors.
#     adj_wgt = graph.adj_wgt  # weight on edge
#     node_wgt = graph.node_wgt  # weight on node
#     cmap = graph.cmap
#     # Option 1: pymp version
#     norm_adj_wgt = normalized_adj_wgt(ctrl, graph, nodePartition)
#     # Option 2: threading version
#     # norm_adj_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
#     # threads = []
#     # for tid in range(ctrl.num_threads):
#     #     t = threading.Thread(target=threading_adj_wgt, args=(graph,nodePartition[tid],nodePartition[tid+1],norm_adj_wgt))
#     #     threads.append(t)
#     #     t.start()
#     timer.printIntervalTime('norm_adj_wgt')
#     enable_parallel = ctrl.coarse_parallel and graph.node_num > ctrl.coarse_threshold

#     max_node_wgt = ctrl.max_node_wgt
#     groups = []  # a list of groups, each group corresponding to one coarse node.
#     if enable_parallel:
#         matched = pymp.shared.array((node_num,), dtype='int32')
#         matched[:] = -1
#         MATCHED_SEM = 0x40000000L  # --> INFTY
#     else:
#         matched = [False] * node_num
#     timer.printIntervalTime('initialize before SEM')

#     # SEM: structural equivalence matching.
#     jaccard_idx_preprocess(ctrl, graph, matched, groups, matched_sem=MATCHED_SEM if enable_parallel else True)
#     ctrl.logger.info("# groups have perfect jaccard idx (1.0): %d" % len(groups))
#     timer.printIntervalTime('SEM')

#     if not enable_parallel:
#         degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]
#         sorted_idx = np.argsort(degree)
#         for idx in sorted_idx:
#             if matched[idx]:
#                 continue
#             max_idx = idx
#             max_wgt = -1
#             for j in range(adj_idx[idx], adj_idx[idx + 1]):
#                 neigh = adj_list[j]
#                 if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
#                     continue
#                 curr_wgt = norm_adj_wgt[j]
#                 if ((not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
#                     max_idx = neigh
#                     max_wgt = curr_wgt
#             # it might happen that max_idx is idx, which means cannot find a match for the node. 
#             matched[idx] = matched[max_idx] = True
#             if idx == max_idx:
#                 groups.append([idx])
#             else:
#                 groups.append([idx, max_idx])
#         timer.printIntervalTime('attempt matching (serial)')
#     else:
#         num_threads = ctrl.num_threads
#         ph_groups = pymp.shared.list([None] * num_threads)
#         # num_groups = pymp.shared.array((num_threads,), dtype='int32')  # DIRECT COPY LOCAL GROUPS INTO A GLOBAL ONE
#         # groups = pymp.shared.list([None] * node_num)
#         barrier1 = Barrier(num_threads)
#         # barrier2 = Barrier(num_threads)
#         # barrier3 = Barrier(num_threads)
#         # degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]
#         # sorted_idx = np.argsort(degree)
#         # num_conflict = pymp.shared.array((1,), dtype='int32')
#         timer.printIntervalTime('Initialize before mt')
#         with pymp.Parallel(num_threads) as p:
#             tid = p.thread_num
#             if tid == 0:
#                 timer_match = Timer(ident=3)
#             # # # strategy 1: divide nodes into consecutive chunks, then sort
#             st, ed = nodePartition[tid], nodePartition[tid + 1]
#             degree = [adj_idx[i + 1] - adj_idx[i] for i in range(st, ed)]
#             sorted_idx = np.argsort(degree) + st


#             # strategy 2: divide nodes into interleaving chunks, then sort (tested: faster than first)
#             # degree = [adj_idx[i + 1] - adj_idx[i] for i in range(tid, node_num, num_threads)]
#             # sorted_idx = np.argsort(degree) * num_threads + tid

#             # strategy 3: sort globally, then divide into chunks in a 'z' manner (16 25 34)
#             # idxs = sorted_idx[sorted(range(tid, node_num, num_threads * 2) + range(num_threads * 2 - tid - 1, node_num, num_threads * 2))]
#             for idx in sorted_idx:
#                 if matched[idx] >= 0:
#                     continue
#                 max_idx = idx
#                 max_wgt = -1
#                 for j in range(adj_idx[idx], adj_idx[idx + 1]):
#                     neigh = adj_list[j]
#                     if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
#                         continue
#                     curr_wgt = norm_adj_wgt[j]
#                     if ((matched[neigh] < 0) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
#                         max_idx = neigh
#                         max_wgt = curr_wgt
#                 if matched[idx] < 0 and matched[max_idx] < 0:   # >>>
#                     if idx < max_idx:
#                         matched[idx] = max_idx
#                         matched[max_idx] = idx
#                     else:
#                         matched[max_idx] = idx
#                         matched[idx] = max_idx

#             barrier1.wait()
#             if tid == 0:
#                 timer_match.printIntervalTime('match and barrier')

#             # groups_local = []
#             # for idx in sorted_idx:
#             #     max_idx = matched[idx]
#             #     if max_idx == MATCHED_SEM:  # already matched in SEM
#             #         continue
#             #     if (max_idx == idx):        # match with itself
#             #         groups_local.append([idx])
#             #         continue
#             #     if (matched[max_idx] != idx):
#             #         groups_local.append([idx])
#             #         # with p.lock:
#             #         #     num_conflict[0] += 1
#             #         continue
#             #     # matched[idx] != MATCHED_SEM
#             #     # matched[idx] != idx
#             #     # matched[matched[idx]] == idx  <- correct match
#             #     if max_idx > idx: # Match only once!
#             #         groups_local.append([idx, max_idx])

#             # # num_groups[tid] = len(groups_local)  # DIRECT COPY LOCAL GROUPS INTO A GLOBAL ONE
#             # barrier2.wait()
#             # if tid == 0:
#             #     timer_match.printIntervalTime('write into groups_local')

#             # ph_groups[tid] = groups_local
#             # barrier3.wait()
#             # if tid == 0:
#             #     timer_match.printIntervalTime('write to ph_groups')
        
#         timer.printIntervalTime('attempt matching - writing matched (parallel)')

#         for idx in range(node_num):
#             max_idx = matched[idx]
#             if max_idx == MATCHED_SEM:  # already matched in SEM
#                 continue
#             if (max_idx == idx):        # match with itself
#                 groups.append([idx])
#                 continue
#             if (matched[max_idx] != idx):
#                 groups.append([idx])
#                 # with p.lock:
#                 #     num_conflict[0] += 1
#                 continue
#             # matched[idx] != MATCHED_SEM
#             # matched[idx] != idx
#             # matched[matched[idx]] == idx  <- correct match
#             if max_idx > idx: # Match only once!
#                 groups.append([idx, max_idx])            

#         timer.printIntervalTime('attempt matching - writing groups (parallel)')

#         # # print("num_conflict / node_num: %d / %d"%(num_conflict[0], node_num))
#         # for gr in ph_groups:
#         #     groups = groups + gr
#         # # groups = groups[:num_groups.sum()]
#         # timer.printIntervalTime('attempt matching - merge groups (serial)')

#     coarse_graph_size = 0
#     for idx in range(len(groups)):
#         for ele in groups[idx]:
#             cmap[ele] = coarse_graph_size
#         coarse_graph_size += 1
#     timer.printIntervalTime('create cmap (serial)')
#     return (groups, coarse_graph_size)

# def threading_add_neighs2node(graph, tid, num_threads, neighs2node):
#     st, ed = int(float(graph.node_num) / num_threads * tid), int(float(graph.node_num) / num_threads * (tid + 1))
#     for i in range(st, ed):
#         neighs = str(sorted(graph.get_neighs(i)))
#         neighs2node[neighs].append(i)
#     return

def jaccard_idx_preprocess(ctrl, graph, matched, groups, matched_sem=True, nodePartition=None):
    '''Use hashmap to find out nodes with exactly same neighbors.'''
    neighs2node = defaultdict(list)
    timer = Timer(logger=ctrl.logger, ident=3)
    enable_parallel = ctrl.coarse_parallel # and graph.node_num > ctrl.coarse_threshold
    if not enable_parallel:
        for i in range(graph.node_num):
            neighs = str(sorted(graph.get_neighs(i)))
            neighs2node[neighs].append(i)
        timer.printIntervalTime('build dicts')
    # else:
    #     num_threads = ctrl.num_threads
    #     dicts = pymp.shared.list([None] * num_threads)
    #     timer.printIntervalTime('initialize')

    #     # build dicts in parallel
    #     # threads = []
    #     # for tid in range(4):
    #     #     t = threading.Thread(target=threading_add_neighs2node, args=(graph, tid, 4, neighs2node))
    #     #     threads.append(t)
    #     #     t.start()
    #     with pymp.Parallel(num_threads) as p:
    #         tid = p.thread_num
    #         local_neighs2node = defaultdict(list)
    #         for i in p.range(graph.node_num):
    #             neighs = str(sorted(graph.get_neighs(i)))
    #             local_neighs2node[neighs].append(i)
    #         dicts[tid] = local_neighs2node
    #     timer.printIntervalTime('build dicts in parallel')

    #     # merge all dicts
    #     for i in range(num_threads):
    #         for k, v in dicts[i].items():
    #             neighs2node[k] += v
    #     timer.printIntervalTime('merge dicts')
    else:
        num_threads = ctrl.num_threads
        k = ctrl.min_hash
        node_num = graph.node_num

        # generate signature

        large_prime = (1 << 31) - 1
        random_a = np.random.randint(large_prime/100, large_prime, size=(k, 1))
        random_b = np.random.randint(large_prime/100, large_prime, size=(k, 1))
        # ra_gpu = cp.array(random_a)
        # rb_gpu = cp.array(random_b)
        # print(random_a, random_b)
        timer.printIntervalTime('initialize')

        signature = pymp.shared.array((node_num, k + 1), dtype=np.int32)
        # ctrl.logger.info("start signature")
        with pymp.Parallel(num_threads) as p:
            # time_st_tid = time.time()
            neighs = None
            tid = p.thread_num
            # time_st, time_ed = 0, 0
            # time_get, time_len, time_comp, time_write = 0, 0, 0, 0
            # count_neighs = 0
            # count_nodes = 0
            tmp = 0
            for i in range(tid, node_num, num_threads):
                # count_nodes += 1
                # time_st = time.time()
                neighs = graph.get_neighs(i)
                # time_ed = time.time()
                # time_get += time_ed - time_st
                # time_st = time.time()
                signature[i, 0] = len(neighs)
                # count_neighs += len(neighs)
                # time_ed = time.time()
                # time_len += time_ed - time_st
                if len(neighs) == 1:
                    signature[i, 1:] = neighs[0]  # ((random_a * neighs[0] + random_b) % large_prime).flatten()
                # getMinHashKeys(signature[i, 1:], neighs, mins, k, randoms, large_prime)
                # time_st = time.time()
                else:
                    signature[i, 1:] = np.min((random_a * neighs + random_b) % large_prime, axis=1)
                    # print("i %d neigh %s"%(i, neighs))
                    # signature[i, 1:] = cp.min((ra_gpu * cp.array(neighs) + rb_gpu) % large_prime, axis=1).get()
                # time_ed = time.time()
                # time_comp += time_ed - time_st
            # print("tid %d count_nodes %d count_neighs %d time %.3f time_get %.3f time_len %.3f time_comp %.3f"%(tid, count_nodes, count_neighs, time.time() - time_st_tid, time_get, time_len, time_comp))
        
        signature_sum = np.sum(signature[:, 1:], axis=1, dtype=np.int64)
        """
        deprecated
        MinHash using permutations.
        breakdown (on yelp cl=1):
            permutation ~10s
            signature   27s
            unique      ~20s
            build dicts ~8s
            total       ~70s   
        """
        # permu = pymp.shared.array((k, node_num), dtype=np.int32)
        # signature = pymp.shared.array((node_num, k + 1), dtype=np.int32)
        # with pymp.Parallel(min(k, num_threads)) as p:
        #     np.random.seed(None)
        #     for i in p.range(k):
        #         permu[i] = np.random.permutation(node_num)
        # timer.printIntervalTime('draw k permutations')

        # with pymp.Parallel(num_threads) as p:
        #     neighs = None
        #     for i in p.range(node_num):
        #         neighs = graph.get_neighs(i)
        #         signature[i] = np.append(len(neighs), np.min(permu[:, neighs], axis=1))
        # timer.printIntervalTime('get signature (in parallel)')
        # del permu

        """
        deprecated
        Compute raw_hash for all nodes then np.min for each node
        breakdown:
            compute 29 s
            get     13 s
            unique  20 s
            dict    8 s
            total   73 s
        """
        # adj_list = graph.adj_list
        # adj_idx = graph.adj_idx
        # random_a = np.random.randint(large_prime/100, large_prime, size=(1, k))
        # random_b = np.random.randint(large_prime/100, large_prime, size=(1, k))
        # raw_hash = (adj_list.reshape(-1, 1) * random_a + random_b) % large_prime
        # timer.printIntervalTime('compute raw_hash')
        # signature = pymp.shared.array((node_num, k + 1), dtype=np.int32)
        # with pymp.Parallel(num_threads) as p:
        #     tid = p.thread_num
        #     time_st_tid = time.time()
        #     for i in p.range(node_num):
        #         signature[i, 0] = adj_idx[i + 1] - adj_idx[i]
        #         signature[i, 1:] = np.min(raw_hash[adj_idx[i]:adj_idx[i + 1]], axis=0)
        #     print("tid %d time %.3f"%(tid, time.time() - time_st_tid))            
        # timer.printIntervalTime('get signature (in parallel)')

        """
        deprecated
        Convert signature to string as keys of the dictionary.
        """
        # time_str, time_add = 0, 0
        # time_st, time_ed = 0, 0
        # str_neigh = None
        # for i in range(node_num):
        #     time_st = time.time()
        #     str_neigh = str(signature[i])
        #     time_ed = time.time()
        #     time_str += time_ed - time_st
        #     time_st = time.time()
        #     neighs2node[str_neigh].append(i)
        #     time_ed = time.time()
        #     time_add += time_ed - time_st
        # timer.printIntervalTime('build dicts')
        # print('\t\t\tInterval (str) time: %.5f\n\t\t\tInterval (add) time: %.5f'%(time_str, time_add))

        timer.printIntervalTime('get signature (in parallel)')



        # divide nodes into groups with same neighbors
        if node_num < ctrl.unique_threshold:
            """
            Find nodes with the same signature using numpy.unique
            Time (on yelp cl=1): 20 s
            """
            _, labels, counts = np.unique(signature, return_inverse=True, return_counts=True, axis=0)
            timer.printIntervalTime('np.unique')

            # build dictionary
            for i in range(node_num):
                if counts[labels[i]] > 1:
                    neighs2node[labels[i]].append(i)
            timer.printIntervalTime('build dicts')
        else:
            """
            Partition the signature into $num_threads chunks, use numpy.unique in parallel
            """
            # ctrl.logger.info("unique :: start ")
            count_labels = pymp.shared.array((num_threads,), dtype='int32')
            labels = pymp.shared.array((node_num,), dtype='int32')
            counts = pymp.shared.list([None] * num_threads)
            new2old = np.argsort(signature_sum)
            signature = signature[new2old]
            nP_degree = bNP_degree(signature_sum[new2old], num_threads)
            barrier = Barrier(num_threads)
            # ctrl.logger.info("unique :: initialize")
            with pymp.Parallel(num_threads) as p:
                tid = p.thread_num
                # ctrl.logger.info("tid " + str(tid) + " node_st " + str(nP_degree[tid]) + " node_ed " + str(nP_degree[tid+1]))
                _, labels[nP_degree[tid]:nP_degree[tid+1]], counts[tid] = np.unique(signature[nP_degree[tid]:nP_degree[tid+1]], return_inverse=True, return_counts=True, axis=0)
                count_labels[tid] = np.max(labels[nP_degree[tid]:nP_degree[tid+1]]) + 1
                barrier.wait()

                labels[nP_degree[tid]:nP_degree[tid+1]] += count_labels[:tid].sum()
                # ctrl.logger.info("unique :: parallel tid " + str(tid))

            # ctrl.logger.info("unique :: unique in parallel")
            counts = np.hstack(counts)
            # ctrl.logger.info("unique :: vstack")
            timer.printIntervalTime('np.unique')

            # build dictionary
            for i in range(node_num):
                if counts[labels[i]] > 1:
                    neighs2node[labels[i]].append(new2old[i])
            timer.printIntervalTime('build dicts')

        """
        deprecated
        Find nodes with the same signature using sklearn.cluster.DBScan
        Time (on yelp cl=1): 120 s
        """
        # from sklearn.cluster import DBSCAN as dbs
        # clustering = dbs(eps=0.1, min_samples=2, metric='cityblock', n_jobs=-1).fit(signature)
        # labels = clustering.labels_

    # find SEM
    for key in neighs2node.keys():
        g = neighs2node[key]
        if len(g) > 1:
            for node in g:
                matched[node] = matched_sem
            groups.append(g)
    timer.printIntervalTime('find SEM')
    return


# def create_coarse_graph(ctrl, graph, groups, coarse_graph_size):
#     '''create the coarser graph and return it based on the groups array and coarse_graph_size'''
#     timer = Timer(ident=2)
#     enable_parallel = ctrl.coarse_parallel and graph.node_num > ctrl.coarse_threshold
#     if not enable_parallel:
#         coarse_graph = Graph(coarse_graph_size, graph.edge_num)
#         coarse_graph.finer = graph
#         graph.coarser = coarse_graph
#         cmap = graph.cmap
#         adj_list = graph.adj_list
#         adj_idx = graph.adj_idx
#         adj_wgt = graph.adj_wgt
#         node_wgt = graph.node_wgt
#         timer.printIntervalTime('initialize graph')

#         coarse_adj_list = coarse_graph.adj_list
#         coarse_adj_idx = coarse_graph.adj_idx
#         coarse_adj_wgt = coarse_graph.adj_wgt
#         coarse_node_wgt = coarse_graph.node_wgt
#         coarse_degree = coarse_graph.degree
#         timer.printIntervalTime('data reference')

#         coarse_adj_idx[0] = 0
#         nedges = 0  # number of edges in the coarse graph
#         for idx in range(len(groups)):  # idx in the graph
#             coarse_node_idx = idx
#             neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
#             group = groups[idx]
#             for i in range(len(group)):
#                 merged_node = group[i]
#                 if (i == 0):
#                     coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
#                 else:
#                     coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

#                 istart = adj_idx[merged_node]
#                 iend = adj_idx[merged_node + 1]
#                 for j in range(istart, iend):
#                     k = cmap[adj_list[
#                         j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
#                     if k not in neigh_dict:  # add new neigh
#                         coarse_adj_list[nedges] = k
#                         coarse_adj_wgt[nedges] = adj_wgt[j]
#                         neigh_dict[k] = nedges
#                         nedges += 1
#                     else:  # increase weight to the existing neigh
#                         coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
#                     # add weights to the degree. For now, we retain the loop. 
#                     coarse_degree[coarse_node_idx] += adj_wgt[j]

#             coarse_node_idx += 1
#             coarse_adj_idx[coarse_node_idx] = nedges
#         timer.printIntervalTime('add edges (serial)')
#     else:
#         coarse_graph = Graph(0, 0)  # the only difference
#         coarse_graph.cmap = np.zeros(coarse_graph_size, dtype=np.int32) - 1
#         coarse_graph.finer = graph
#         graph.coarser = coarse_graph
#         cmap = graph.cmap
#         adj_list = graph.adj_list
#         adj_idx = graph.adj_idx
#         adj_wgt = graph.adj_wgt
#         node_wgt = graph.node_wgt
#         timer.printIntervalTime('initialize graph')

#         coarse_adj_idx = pymp.shared.array((coarse_graph_size + 1,), dtype='int32')
#         coarse_node_wgt = pymp.shared.array((coarse_graph_size,), dtype='float32')
#         coarse_degree = pymp.shared.array((coarse_graph_size,), dtype='float32')
#         coarse_adj_list = pymp.shared.array((graph.edge_num,), dtype='int32')
#         coarse_adj_wgt = pymp.shared.array((graph.edge_num,), dtype='float32')
#         timer.printIntervalTime('data reference')

#         num_threads = ctrl.num_threads
#         th_nedges = pymp.shared.array((num_threads,), dtype='int32')
#         barrier1 = Barrier(num_threads)
#         barrier2 = Barrier(num_threads)
#         nedges = 0
#         with pymp.Parallel(num_threads) as p:
#             tid = p.thread_num
#             st, ed = int(float(coarse_graph_size) / num_threads * tid), int(float(coarse_graph_size) / num_threads * (tid + 1))
#             local_nedges = 0
#             local_coarse_adj_list = []  # don't know edge size until we count
#             local_coarse_adj_wgt = []

#             # add edges to coarse graph
#             for idx in range(st, ed):
#                 coarse_node_idx = idx
#                 coarse_adj_idx[coarse_node_idx] = local_nedges
#                 neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
#                 group = groups[idx]
#                 for i in range(len(group)):
#                     merged_node = group[i]
#                     if (i == 0):
#                         coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
#                     else:
#                         coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

#                     istart = adj_idx[merged_node]
#                     iend = adj_idx[merged_node + 1]
#                     for j in range(istart, iend):
#                         k = cmap[adj_list[
#                             j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
#                         if k not in neigh_dict:  # add new neigh
#                             local_coarse_adj_list.append(k)
#                             local_coarse_adj_wgt.append(adj_wgt[j])
#                             neigh_dict[k] = local_nedges
#                             local_nedges += 1
#                         else:  # increase weight to the existing neigh
#                             local_coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
#                         # add weights to the degree. For now, we retain the loop. 
#                         coarse_degree[coarse_node_idx] += adj_wgt[j]

#             th_nedges[tid] = local_nedges
#             barrier1.wait()

#             if tid == num_threads - 1:
#                 nedges = th_nedges.sum()
#                 coarse_adj_idx[coarse_graph_size] = nedges
#             barrier2.wait()

#             edge_st = th_nedges[:tid].sum()
#             edge_ed = th_nedges[:tid+1].sum()
#             assert local_nedges == edge_ed - edge_st, "thread %d edge_num error"%(tid)
#             coarse_adj_idx[st:ed] += edge_st
#             coarse_adj_list[edge_st:edge_ed] = local_coarse_adj_list
#             # print("edge_ed %d edge_st %d edge_ed - edge_st %d len(list) %d len(wgt) %d list.shape %s wgt.shape %s"%(edge_ed, edge_st, edge_ed - edge_st, len(local_coarse_adj_list), len(local_coarse_adj_wgt), coarse_adj_list.shape, coarse_adj_wgt.shape))
#             coarse_adj_wgt[edge_st:edge_ed] = local_coarse_adj_wgt

#         timer.printIntervalTime('add edges (parallel)')

#         nedges = th_nedges.sum()
#         coarse_graph.node_num = coarse_graph_size
#         coarse_graph.adj_idx = coarse_adj_idx
#         coarse_graph.node_wgt = coarse_node_wgt
#         coarse_graph.degree = coarse_degree
#         coarse_graph.adj_list = coarse_adj_list
#         coarse_graph.adj_wgt = coarse_adj_wgt

#         print(coarse_graph.adj_idx)
#         timer.printIntervalTime('update coarse_graph')

#     coarse_graph.edge_num = nedges
#     coarse_graph.resize_adj(nedges)
#     print("node_num %d edge_num %d"%(coarse_graph.node_num, coarse_graph.edge_num))
#     # print("coarse_graph.adj_idx %s"%(coarse_graph.adj_idx.shape))
#     # print("coarse_graph.node_wgt %s"%(coarse_graph.node_wgt.shape))
#     # print("coarse_graph.degree %s"%(coarse_graph.degree.shape))
#     # print("coarse_graph.adj_list %s"%(coarse_graph.adj_list.shape))
#     # print("coarse_graph.adj_wgt %s"%(coarse_graph.adj_wgt.shape))
#     # print("coarse_graph.cmap %s"%(coarse_graph.cmap.shape))
#     C = cmap2C(cmap)  # construct the matching matrix.
#     graph.C = C
#     coarse_graph.A = C.transpose().dot(graph.A).dot(C)
#     timer.printIntervalTime('finish')
#     return coarse_graph


# Deprecated: Parallelized coarsening. Merged into generate_hybrid_matching.
# def generate_HEM_parallel(ctrl, graph):
#     '''Generate matchings using the hybrid method. It changes the cmap in graph object, 
#     return groups array and coarse_graph_size.'''
#     timer = Timer(ident=2)
#     node_num = graph.node_num
#     adj_list = graph.adj_list  # big array for neighbors.
#     adj_idx = graph.adj_idx  # beginning idx of neighbors.
#     adj_wgt = graph.adj_wgt  # weight on edge
#     node_wgt = graph.node_wgt  # weight on node
#     cmap = graph.cmap# pymp.shared.array(graph.cmap.shape, dtype='int32')
#     norm_adj_wgt = normalized_adj_wgt(ctrl, graph)  # compute normalized weights on edge in parallel
#     timer.printIntervalTime('norm_adj_wgt')

#     max_node_wgt = ctrl.max_node_wgt

#     groups = []  # a list of groups, each group corresponding to one coarse node.
#     matched = pymp.shared.array((node_num,), dtype='int32')
#     matched[:] = -1
#     MATCHED_SEM = 0x40000000L
#     timer.printIntervalTime('initialize before SEM')

#     # SEM: structural equivalence matching.
#     jaccard_idx_preprocess(ctrl, graph, matched, groups, matched_sem=MATCHED_SEM)
#     ctrl.logger.info("# groups have perfect jaccard idx (1.0): %d" % len(groups))
#     timer.printIntervalTime('SEM')

#     # sort by degree
#     '''
#     current strategy: divide into consecutive chunks, then sort (125 384 956 -> 125 348 569)
#     alternative: 1. divide into interleaving chunks, then sort (139 285 546 -> 139 258 456)
#                  2. sort, then divide into interleaving chunks (146 258 359)
#     '''
#     num_threads = pymp.config.num_threads[0]
#     ph_groups = pymp.shared.list([None] * num_threads)
#     barrier = Barrier(num_threads)
#     # degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]
#     # sorted_idx = np.argsort(degree)
#     num_conflict = pymp.shared.array((1,), dtype='int32')
#     with pymp.Parallel(num_threads) as p:
#         tid = p.thread_num
#         # # strategy 1: divide nodes into consecutive chunks, then sort
#         st, ed = int(float(node_num) / num_threads * tid), int(float(node_num) / num_threads * (tid + 1))
#         degree = [adj_idx[i + 1] - adj_idx[i] for i in range(st, ed)]
#         sorted_idx = np.argsort(degree) + st

#         # strategy 2: divide nodes into interleaving chunks, then sort (tested: worse than first)
#         # degree = [adj_idx[i + 1] - adj_idx[i] for i in range(tid, node_num, num_threads)]
#         # sorted_idx = np.argsort(degree) * num_threads + tid


#         # strategy 3: sort globally, then divide into chunks in a 'z' manner (16 25 34)
#         # idxs = sorted_idx[sorted(range(tid, node_num, num_threads * 2) + range(num_threads * 2 - tid - 1, node_num, num_threads * 2))]
#         for idx in sorted_idx:
#             if matched[idx] >= 0:
#                 continue
#             max_idx = idx
#             max_wgt = -1
#             for j in range(adj_idx[idx], adj_idx[idx + 1]):
#                 neigh = adj_list[j]
#                 if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
#                     continue
#                 curr_wgt = norm_adj_wgt[j]
#                 if ((matched[neigh] < 0) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
#                     max_idx = neigh
#                     max_wgt = curr_wgt
#             if matched[idx] < 0 and matched[max_idx] < 0:
#                 if idx < max_idx:
#                     matched[idx] = max_idx
#                     matched[max_idx] = idx
#                 else:
#                     matched[max_idx] = idx
#                     matched[idx] = max_idx


#         barrier.wait()

#         groups_local = []
#         for idx in sorted_idx:
#             max_idx = matched[idx]
#             if max_idx == MATCHED_SEM:  # already matched in SEM
#                 continue
#             if (matched[max_idx] != idx):
#                 groups_local.append([idx])
#                 with p.lock:
#                     num_conflict[0] += 1
#             elif (max_idx == idx):
#                 groups_local.append([idx])
#             else:
#                 if max_idx > idx:
#                     groups_local.append([idx, max_idx])

#         ph_groups[tid] = groups_local

#     print("num_conflict / node_num: %d / %d"%(num_conflict[0], node_num))
#     for gr in ph_groups:
#         groups = groups + gr
#     timer.printIntervalTime('attempt matching')

#     # coarse_graph_size = len(groups)
#     # with pymp.Parallel(num_threads) as p:
#     #     for idx in p.range(len(groups)):
#     #         for ele in groups[idx]:
#     #             cmap[ele] = idx
#     coarse_graph_size = 0
#     for idx in range(len(groups)):
#         for ele in groups[idx]:
#             cmap[ele] = coarse_graph_size
#         coarse_graph_size += 1
#     timer.printIntervalTime('create cmap')

#     return (groups, coarse_graph_size)


### deprecated
# def normalized_adj_wgt_parallel(graph):
#     adj_wgt = graph.adj_wgt
#     adj_idx = graph.adj_idx
#     norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
#     degree = graph.degree
#     for i in range(graph.node_num):
#         for j in range(adj_idx[i], adj_idx[i + 1]):
#             neigh = graph.adj_list[j]
#             norm_wgt[j] = adj_wgt[j] / np.sqrt(degree[i] * degree[neigh])
#     return norm_wgt

# def parallel_matching(ctrl, graph, procs, rank, comm, level):
#     """
#     graph: subgraph on the processor
#     procs: number of processors
#     rank: rank of this processor, i.e., subgraph ${rank} is here
#     comm: MPI communication
#     level: 
#     """
#     # define UNMATCHED, MAYBE_MATCHED
#     UNMATCHED = -1
#     MAYBE_MATCHED = -2
#     KEEP_BIT = 0x40000000L

#     # reference for convenience
#     n = graph.node_num
#     nnbrs = graph.nnbrs
#     nrecv = graph.nrecv
#     nsend = graph.nsend
#     st = graph.st
#     ed = graph.ed
#     peind = graph.peind
#     adj_idx = graph.adj_idx
#     adj_list = graph.adj_list  # big array for neighbors.
#     adj_idx = graph.adj_idx  # beginning idx of neighbors.
#     adj_wgt = graph.adj_wgt  # weight on edge
#     sendptr = graph.sendptr
#     recvptr = graph.recvptr
#     graph.match = np.ones(n + nrecv, dtype=np.int32) * UNMATCHED
#     match = graph.match
#     changed = np.empty(n, dtype=np.int32)
#     match_requests = np.empty((nsend, 2), dtype=np.int32)
#     match_granted = np.empty((nrecv, 2), dtype=np.int32)
#     # node_wgt = graph.node_wgt  # weight on node


    
#     norm_adj_wgt = normalized_adj_wgt_parallel(graph)
#     max_node_wgt = ctrl.max_node_wgt

#     ''' create the traversal order
#     Here we visit nodes with increasing node degree
#     '''
#     uni_degree = [adj_idx[i + 1] - adj_idx[i] for i in range(n)]
#     sorted_idx = np.argsort(uni_degree)


#     ''' set initial value of nkept 
#     based on how over/under weight the partition is to begin with
#     '''
#     nkept = (int)(graph.global_node_num / procs) - n
#     if rank == 0:
#         timer_match = Timer(ident=2)

#     nmatched = 0
#     NMATCH_PASSES = 4
#     tag_base = 100 * (level * NMATCH_PASSES * 4 + 1)
#     for pas in range(NMATCH_PASSES):
#         wside = (level + pas) % 2
#         nchanged = nrequests = 0

#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "begin"))
#         for idx in sorted_idx:
#             if match[idx] > UNMATCHED:
#                 continue
#             max_idx = idx
#             max_wgt = -1

#             # heavy-edge matching
#             for j in range(adj_idx[idx], adj_idx[idx + 1]):
#                 neigh = adj_list[j]  # note that this is a local index: 0..n+nrecv-1
#                 if neigh == idx:
#                     continue
#                 curr_wgt = norm_adj_wgt[j]
#                 if (match[neigh] == UNMATCHED and max_wgt < curr_wgt):
#                     max_idx = neigh
#                     max_wgt = curr_wgt

#             if max_idx != idx:
#                 if (max_idx < n):  # match with a local vertix
#                     match[idx] = st + max_idx + (KEEP_BIT if idx <= max_idx else 0)
#                     match[max_idx] = st + idx + (KEEP_BIT if idx > max_idx else 0)
#                     changed[nchanged] = idx
#                     changed[nchanged + 1] = max_idx
#                     nchanged += 2
#                 else:  # match with a remote vertix
#                     match[max_idx] = MAYBE_MATCHED
#                     # determine which vertices will issue the requests
#                     if ((wside == 0 and st + idx < graph.imap[max_idx]) or
#                         (wside == 1 and st + idx > graph.imap[max_idx])):
#                         match[idx] = MAYBE_MATCHED
#                         match_requests[nrequests] = [graph.imap[max_idx], st + idx]  # will be issued by the current processor
#                         nrequests += 1
#         if rank == 0:
#             timer_match.printIntervalTime(name='pas %d match'%(pas))

#         '''
#         Exchange match_requests,
#         requests for the current processor will be stored in match_granted
#         '''
#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "exchange match_requests"))
#         match_requests[:nrequests] = match_requests[np.argsort(match_requests[:nrequests, 0])]
#         tag_base += 100
#         num_sendreq = np.zeros(procs, dtype=np.int32)
#         num_recvreq = np.zeros(procs, dtype=np.int32)
#         reqs = []
#         j = 0
#         for i in range(nnbrs):
#             k = 0
#             for kk in range(j, nrequests + 1):
#                 if (kk == nrequests) or (match_requests[kk, 0] >= graph.vtxdist[peind[i] + 1]):
#                     k = kk
#                     break
#             if k > j:  # only send when there is any request
#                 reqs.append(comm.Isend(get_array_buffer(match_requests[j:k]), dest=peind[i], tag=tag_base*(rank+1)+peind[i]))
#                 num_sendreq[peind[i]] = k-j
#             j = k
#         comm.Alltoall(num_sendreq, num_recvreq)
#         reqr = []
#         for i in range(nnbrs):
#             if num_recvreq[peind[i]] > 0:  # to match the number of requests from other processors
#                 reqr.append(comm.Irecv(get_array_buffer(match_granted[recvptr[i]:recvptr[i+1]]), source=peind[i], tag=tag_base*(peind[i]+1)+rank))
#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "exchange match_requests Waitall"))
#         MPI.Request.Waitall(reqr)
#         MPI.Request.Waitall(reqs)
#         if rank == 0:
#             timer_match.printIntervalTime(name='pas %d exchange requests'%(pas))
        


#         ''' For rank i and source processor j (peind[j]), 
#         match_request would be less than nsend requests to other processors in increasing order.
#         match_granted[recvptr[j]:recvptr[j+1]] would be num_recvreq[peind[j]] valid values and some random values for place holding
#         '''

#         '''
#         process the requests that are received in match_granted
#         '''
#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "process requests"))
#         nperm = np.random.permutation(nnbrs)
#         for ii in range(nnbrs):
#             i = nperm[ii]
#             for j in range(num_recvreq[peind[i]]):
#                 k = match_granted[recvptr[i] + j, 0]  # local node that receives requests from other processor
#                 assert st <= k and k < ed, "match_granted from rank %d nrecv[i] %d j %d num_recvreq %d k %d st %d ed %d"%(peind[i], recvptr[i+1]-recvptr[i], j, num_recvreq[peind[i]], k, st, ed)
#                 if match[k - st] == UNMATCHED:
#                     changed[nchanged] = k - st
#                     nchanged += 1
#                     if (nkept >= 0):  # it needs to add more vertex, so keep the collapsed vertex on this processor
#                         match[k - st] = match_granted[recvptr[i] + j, 1] + KEEP_BIT
#                         nkept -= 1
#                     else:
#                         match[k - st] = match_granted[recvptr[i] + j, 1]
#                         match_granted[recvptr[i] + j, 0] += KEEP_BIT
#                         nkept += 1
#                 else:
#                     match_granted[recvptr[i] + j, 0] = UNMATCHED
#         if rank == 0:
#             timer_match.printIntervalTime(name='pas %d process requests'%(pas))

#         '''
#         exchange the match_granted
#         Now the match_requests will be partitioned according to sendptr (empty elements may be between partitions)
#         '''
#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "exchange match_granted"))
#         tag_base += 100
#         reqs = []
#         for i in range(nnbrs):
#             num_recvreqs = num_recvreq[peind[i]]
#             reqs.append(comm.Isend(get_array_buffer(match_granted[recvptr[i]:(recvptr[i] + num_recvreqs)]), dest=peind[i], tag=tag_base*(rank+1)+peind[i]))
#         reqr = []
#         for i in range(nnbrs):
#             if num_sendreq[peind[i]] > 0:
#                 reqr.append(comm.Irecv(get_array_buffer(match_requests[sendptr[i]:sendptr[i+1]]), source=peind[i], tag=tag_base*(peind[i]+1)+rank))
#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "exchange match_granted Waitall"))
#         MPI.Request.Waitall(reqr)
#         MPI.Request.Waitall(reqs)
#         if rank == 0:
#             timer_match.printIntervalTime(name='pas %d exchange granted'%(pas))

#         '''
#         Go through the match_requests and update local match information
#         '''
#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "go through"))
#         for i in range(nnbrs):
#             for j in range(num_sendreq[peind[i]]):
#                 k = match_requests[sendptr[i] + j, 0]
#                 match[match_requests[sendptr[i] + j, 1] - st] = k
#                 if (k != UNMATCHED):
#                     changed[nchanged] = match_requests[sendptr[i] + j, 1] - st
#                     nchanged += 1
#         if rank == 0:
#             timer_match.printIntervalTime(name='pas %d update match'%(pas))

#         '''
#         Gather and scatter the match information
#         '''
#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "gather and scatter"))
#         pexadj = graph.pexadj
#         peadjcny = graph.peadjcny
#         peadjloc = graph.peadjloc
#         reqs = []
#         num_sendreq = np.zeros(procs, dtype=np.int32)
#         num_recvreq = np.zeros(procs, dtype=np.int32)
#         tag_base += 100
#         if nchanged != 0:
#             psendptr = np.copy(sendptr)
#             # print("\trank %d psendptr %s"%(rank, psendptr))
#             for i in range(nchanged):
#                 j = changed[i]
#                 for k in range(pexadj[j], pexadj[j+1]):
#                     penum = peadjcny[k]
#                     match_requests[psendptr[penum], 0] = peadjloc[k]  # the local index of vertex j in processor peind[penum]
#                     match_requests[psendptr[penum], 1] = match[j]
#                     psendptr[penum] += 1
#             for i in range(nnbrs):
#                 num_sendreq[peind[i]] = psendptr[i] - sendptr[i]
#                 reqs.append(comm.Isend(get_array_buffer(match_requests[sendptr[i]:psendptr[i]]), dest=peind[i], tag=tag_base*(rank+1)+peind[i]))
#                 # print("rank %d peind %d send[:5] %s start %d end %d sendshape %s tag %d"%(rank, peind[i], match_requests[sendptr[i]:min(sendptr[i]+5, psendptr[i])], sendptr[i], psendptr[i], (match_requests[sendptr[i]:psendptr[i]]).shape, tag_base*(rank+1)+peind[i]))
#         else:
#             for i in range(nnbrs):
#                 reqs.append(comm.Isend(get_array_buffer(match_requests[sendptr[i]:sendptr[i]]), dest=peind[i], tag=tag_base*(rank+1)+peind[i]))
#         comm.Alltoall(num_sendreq, num_recvreq)
#         # print("rank %d num_sendreq %s num_recvreq %s"%(rank, num_sendreq, num_recvreq))
#         reqr = []
#         for i in range(nnbrs):
#             reqr.append(comm.Irecv(get_array_buffer(match_granted[recvptr[i]:(recvptr[i]+num_recvreq[peind[i]])]), source=peind[i], tag=tag_base*(peind[i]+1)+rank))

#         # print("rank %d level %d pas %d/%d nnbrs %d peind %s tag_base %d send_tags %s recv_tags %s: %s"%(rank, level, pas, NMATCH_PASSES, nnbrs, peind, tag_base, tag_base*(rank+1)+peind, tag_base*(peind+1)+rank, "gather and scatter Waitall"))
        
        
#         # print("rank %d after Waitall"%(rank))
#         # print('---')
#         # time.sleep(3*(rank+1))
#         # for i in range(nnbrs):
#         #     print("rank %d receives from rank %d shape %s data %s tag %d"%(rank, peind[i], (match_granted[recvptr[i]:(recvptr[i]+num_recvreq[peind[i]])]).shape, match_granted[recvptr[i]:(recvptr[i]+min(5, num_recvreq[peind[i]]))], tag_base*(peind[i]+1)+rank))
#         # print("rank %d at Barrier"%(rank))
#         MPI.Request.Waitall(reqr)
#         # print("rank %d level %d pas %d/%d : %s"%(rank, level, pas, NMATCH_PASSES, "update match"))
#         for i in range(nnbrs):
#             if (num_recvreq[peind[i]] > 0):
#                 for k in range(num_recvreq[peind[i]]):
#                     assert 0 <= match_granted[recvptr[i] + k, 0] and match_granted[recvptr[i] + k, 0] < n+nrecv, "rank %d peind %d i %d recvptr[i] %d k %d match_granted[recvptr[i]+k, 0] %d \nn %d nrecv %d match_granted %s"%(rank, peind[i], i, recvptr[i], k, match_granted[recvptr[i] + k, 0], n, nrecv, match_granted[recvptr[i] + k:recvptr[i] + k+5])
#                     match[match_granted[recvptr[i] + k, 0]] = match_granted[recvptr[i] + k, 1]
#         MPI.Request.Waitall(reqs)
#         if rank == 0:
#             timer_match.printIntervalTime(name='pas %d exchange match'%(pas))

#     '''
#     Traverse the vertices and find those that were unmatched, match them with themselves
#     '''
#     cnvtvx = 0
#     for i in range(n):
#         if match[i] == UNMATCHED:
#             match[i] = st + i + KEEP_BIT
#             cnvtvx += 1
#         elif match[i] >= KEEP_BIT:
#             cnvtvx += 1

#     # print("rank %d, cnvtvx : %d"%(rank, cnvtvx))
#     return cnvtvx, match[:n]


