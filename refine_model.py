import numpy as np
import math
import tensorflow as tf
from mpi4py import MPI
import horovod.tensorflow as hvd
import pymp
import scipy.sparse as sp
import time
import itertools
import GPUtil


from utils import graph_to_adj, graph_to_adj_distributed_parallel, Timer
from mpiutils import get_array_buffer
from mputils import smart_bcast, smart_gather, smart_allgather

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def diag_ones(shape, name=None):
    """All ones."""
    initial = tf.diag(np.ones(shape[0], dtype=np.float32))
    return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_to_gcn_adj(adj, lda):  # D^{-0.5} * A * D^{-0.5} : normalized, symmetric convolution operator.
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    self_loop_wgt = np.array(adj.sum(1)).flatten() * lda  # self loop weight as much as sum. This is part is flexible.
    adj_normalized = normalize_adj(adj + sp.diags(self_loop_wgt))
    return adj_normalized


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


class GCN:
    '''Refinement model based on the Graph Convolutional Network model (GCN). 
    Parts of the code are adapted from https://github.com/tkipf/gcn'''

    def __init__(self, ctrl, session, hvd, rank):
        self.logger = ctrl.logger
        self.embed_dim = ctrl.embed_dim
        self.args = ctrl.refine_model
        self.session = session
        self.hvd = hvd  # currently only valid for GraphSAGE, just for eliminating runtime error
        self.rank = rank
        self.model_dict = dict()
        self.build_tf_graph()

    def build_tf_graph(self):
        act_func = self.args.act_func
        wgt_decay = self.args.wgt_decay
        regularized = self.args.regularized
        learning_rate = self.args.learning_rate
        hidden_layer_num = self.args.hidden_layer_num
        tf_ops = self.args.tf_optimizer
        vars_arr = []

        # placeholders
        input_embed = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="input_embed")  # None: node_num
        gcn_A = tf.sparse_placeholder(tf.float32)  # node_num * node_num
        expected_embed = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="expected_embed")

        curr = input_embed

        for i in range(hidden_layer_num):
            W_hidd = glorot((self.embed_dim, self.embed_dim), name="W_" + str(i))
            curr = act_func(dot(dot(gcn_A, curr, sparse=True), W_hidd))
            vars_arr.append(W_hidd)

        pred_embed = tf.nn.l2_normalize(curr, axis=1)  # this normalization is necessary.
        loss = 0.0
        loss += tf.losses.mean_squared_error(expected_embed, pred_embed) * self.embed_dim

        if regularized:
            for var in vars_arr:
                loss += tf.nn.l2_loss(var) * wgt_decay

        optimizer = tf_ops(learning_rate=learning_rate).minimize(loss)

        self.model_dict['input_embed'] = input_embed
        self.model_dict['gcn_A'] = gcn_A
        self.model_dict['expected_embed'] = expected_embed
        self.model_dict['pred_embed'] = pred_embed
        self.model_dict['optimizer'] = optimizer
        self.model_dict['loss'] = loss
        self.model_dict['vars_arr'] = vars_arr
        init = tf.global_variables_initializer()
        self.session.run(init)

    def train_model(self, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        '''Train the refinement model.'''
        if self.args.untrained_model:  # this is the 'MD-dumb' model, which will not train the model.
            return
        normalized_A = preprocess_to_gcn_adj(graph_to_adj(fine_graph), self.args.lda)
        gcn_A = sparse_to_tuple(normalized_A)

        if coarse_embed is not None:
            initial_embed = fine_graph.C.dot(coarse_embed)  # projected embedings.
        else:
            initial_embed = fine_embed

        early_stopping = self.args.early_stopping
        self.logger.info("initial_embed: " + str(initial_embed.shape))
        self.logger.info("fine_embed: " + str(fine_embed.shape))
        loss_arr = []
        self.logger.info("Refinement Model Traning: ")
        for i in range(self.args.epoch):
            optimizer, loss = self.session.run([self.model_dict['optimizer'], self.model_dict['loss']], feed_dict={
                self.model_dict['input_embed']: initial_embed,
                self.model_dict['gcn_A']: gcn_A,
                self.model_dict['expected_embed']: fine_embed})
            loss_arr.append(loss)
            if i % 20 == 0:
                self.logger.info("  GCN iterations-" + str(i) + ": " + str(loss))
            if i > early_stopping and loss_arr[-1] > np.mean(loss_arr[-(early_stopping + 1):-1]):
                self.logger.info("Early stopping...")
                break

    def refine_embedding(self, coarse_graph=None, fine_graph=None, coarse_embed=None):
        '''Apply the learned model for embeddings refinement.'''
        normalized_A = preprocess_to_gcn_adj(graph_to_adj(fine_graph), self.args.lda)
        gcn_A = sparse_to_tuple(normalized_A)
        initial_embed = fine_graph.C.dot(coarse_embed)
        refined_embed, = self.session.run([self.model_dict['pred_embed']], feed_dict={
            self.model_dict['input_embed']: initial_embed,
            self.model_dict['gcn_A']: gcn_A})
        return refined_embed


def alias_setup(probs):
    '''Compute utility lists for non-uniform sampling from discrete distributions. For GraphSAGE only.'''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''Draw sample from a non-uniform discrete distribution using alias sampling. For GraphSAGE only.'''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class GraphSage:
    '''Refinement model based on the GraphSage. Parts of the code are adapted from https://github.com/williamleif/GraphSAGE.'''

    def __init__(self, comm, ctrl, hvd, rank, procs):
        self.logger = ctrl.logger
        self.embed_dim = ctrl.embed_dim
        self.num_neighbors = ctrl.refine_model.gs_sample_neighbrs_num
        self.args = ctrl.refine_model
        self.hvd = hvd
        self.comm = comm
        self.rank = rank
        self.procs = procs
        # self.batch_per_node = ctrl.refine_model.gs_batch_per_node
        self.batch_size = ctrl.refine_model.batch_size
        self.model_dict = dict()
        self.depth = ctrl.refine_model.gs_depth
        self.mlp_layer = ctrl.refine_model.gs_mlp_layer
        self.gs_concat = ctrl.refine_model.gs_concat
        self.uniform_sample = ctrl.refine_model.gs_uniform_sample
        self.self_wt = ctrl.refine_model.gs_self_wt
        self.large = ctrl.refine_model.large
        self.sync_neigh = ctrl.refine_model.sync_neigh
        self.num_threads = ctrl.workers
        self.refine_threshold = ctrl.refine_model.refine_threshold
        self.mpi_buff = ctrl.refine_model.gs_mpi_buff
        self.build_tf_graph()

    def build_tf_graph(self):
        act_func = self.args.act_func
        wgt_decay = self.args.wgt_decay
        regularized = self.args.regularized
        basic_learning_rate = self.args.learning_rate
        hidden_layer_num = self.args.hidden_layer_num
        mlp_layer = self.mlp_layer  # before max-pooling
        mlp_dim = self.embed_dim
        num_neighbors = self.num_neighbors
        concat = self.gs_concat
        tf_ops = self.args.tf_optimizer
        vars_arr = []
        global_step = tf.train.get_or_create_global_step()

        # placeholders
        ratio = tf.placeholder_with_default(1.0, shape=(), name="ratio")    # added: batch_size / graph_size
        input_embed = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="input_embed")  # None: node_num
        all_vecs = tf.placeholder(tf.float32, shape=[None, self.embed_dim],
                                    name="all_vecs")  # n, embed_dim
        expected_embed = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name="expected_embed")
        B_neigh = tf.placeholder(tf.int32, shape=(None), name="B_neigh")
        B_k = tf.placeholder(tf.int32, shape=(None), name="B_k")
        B_size = tf.placeholder(tf.int32, shape=(self.depth), name="B_size")

        updated_all_vecs = all_vecs  # a copy of all_vecs, will be updated in following steps

        b_size = 0
        for k in range(self.depth):
            # extract B^(k+1) and B^(k+1)'s neighbor
            Bk = B_k[b_size: (b_size + B_size[k])]
            Bk = tf.reshape(Bk, shape=[-1, 1])
            Bk_neigh = B_neigh[b_size * num_neighbors : (b_size + B_size[k]) * num_neighbors]
            Bk_neigh = tf.reshape(Bk_neigh, shape=[-1, 1])
            b_size += B_size[k]

            # multi-layer MLP max pooling aggregator
            prev_dim = self.embed_dim
            curr = tf.gather_nd(updated_all_vecs, Bk_neigh)
            for i in range(mlp_layer):
                mlp_weights = glorot((prev_dim, mlp_dim), name="mlp_weights_" + str(i))
                mlp_bias = zeros([mlp_dim], name="mlp_bias_" + str(i))
                curr = tf.nn.relu(dot(curr, mlp_weights) + mlp_bias)
                vars_arr.append(mlp_bias)
                vars_arr.append(mlp_weights)
                prev_dim = mlp_dim

            neigh_reshape = tf.reshape(curr, (-1, num_neighbors, mlp_dim))
            max_pool = tf.reduce_max(neigh_reshape, axis=1)

            # concat and normalize
            self_vecs = tf.gather_nd(updated_all_vecs, Bk)

            neigh_wgts = glorot((mlp_dim, self.embed_dim), name="neigh_wgts")
            self_wgts = glorot((self.embed_dim, self.embed_dim), name="self_wgts")
            from_neighs = tf.matmul(max_pool, neigh_wgts)
            from_self = tf.matmul(self_vecs, self_wgts)
            if not concat:
                output = tf.add_n([from_self, from_neighs])
            else:
                output_1 = tf.concat([from_self, from_neighs], axis=1)
                weights_compress = glorot((2 * self.embed_dim, self.embed_dim), name="W_concat")
                weight_bias = zeros([self.embed_dim], name="mlp_bias_" + str(i))
                output = dot(output_1, weights_compress) + weight_bias
                vars_arr.append(weight_bias)
                vars_arr.append(weights_compress)

            output = act_func(output)
            pred_embed = tf.nn.l2_normalize(output, axis=1)

            # update updated_all_vecs for nodes in B^(k+1)
            updated_all_vecs = tf.tensor_scatter_nd_update(updated_all_vecs, Bk, pred_embed)

        loss = 0.0
        loss += tf.reduce_mean(tf.reduce_sum(tf.square(expected_embed - pred_embed), axis=1))

        optimizer = tf_ops(learning_rate=basic_learning_rate) 
        optimizer = self.hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=global_step)

        self.model_dict['input_embed'] = input_embed
        self.model_dict['all_vecs'] = all_vecs
        self.model_dict['expected_embed'] = expected_embed
        self.model_dict['pred_embed'] = pred_embed
        self.model_dict['optimizer'] = train_op  # changed from 'optimizer', for using horovod
        self.model_dict['loss'] = loss
        self.model_dict['vars_arr'] = vars_arr
        self.model_dict['ratio'] = ratio
        self.model_dict['B_neigh'] = B_neigh
        self.model_dict['B_k'] = B_k
        self.model_dict['B_size'] = B_size
        # init = tf.global_variables_initializer()
        # self.session.run(init)

    # def sample_from_adj(self, adj, sample_size, node_list=None):
    #     # determine whether to sample for all nodes or nodes in node_list only
    #     num_nodes = len(node_list) if node_list!=None else adj.shape[0]
    #     iter_range = node_list if node_list!=None else range(adj.shape[0])

    #     # sample neighbors
    #     adj_sample = np.zeros((num_nodes, sample_size), dtype=np.int32)
    #     for i in range(num_nodes):
    #         idx = iter_range[i]
    #         row = np.squeeze(adj.getrow(idx).toarray())
    #         if not self.self_wt:
    #             row[idx] = 0
    #         sampled_row = np.nonzero(row)
    #         if not self.uniform_sample:
    #             weights = np.ravel(row[sampled_row])
    #             dist = weights / weights.sum()
    #             J, q = alias_setup(dist)
    #             for j in range(sample_size):
    #                 adj_sample[i, j] = sampled_row[0][alias_draw(J, q)]
    #         else:
    #             sample = np.random.randint(sampled_row[0].size, size=sample_size)  # with replacement.
    #             adj_sample[i, :] = np.take(sampled_row[0], sample)

    #     return adj_sample.flatten()

    # def sample_from_adj(self, adj, start, end, sample_size):
    #     # adj_sample = np.zeros((end - start, sample_size), dtype=np.int32)
    #     adj_sample = pymp.shared.array((end - start, sample_size), dtype='int32')
    #     with pymp.Parallel(self.num_threads) as p:
    #         tid = p.thread_num
    #         sample_st = time.time()
    #         for i in p.range(start, end):
    #             row = np.squeeze(adj.getrow(i).toarray())
    #             if not self.self_wt:
    #                 row[i] = 0
    #             sampled_row = np.nonzero(row)
    #             if not self.uniform_sample:
    #                 weights = np.ravel(row[sampled_row])
    #                 dist = weights / weights.sum()
    #                 J, q = alias_setup(dist)
    #                 for j in range(sample_size):
    #                     adj_sample[i - start, j] = sampled_row[0][alias_draw(J, q)]
    #             else:
    #                 sample = np.random.randint(sampled_row[0].size, size=sample_size)  # with replacement.
    #                 adj_sample[i - start, :] = np.take(sampled_row[0], sample)
    #         sample_ed = time.time()
    #         print("rank %d tid %d time %.3f"%(self.rank, tid, sample_ed - sample_st))

    #     return adj_sample

    # def sample_all_depth(self, adj, start, end, depth, sample_size):
    #     time_st = time.time()
    #     n = adj.shape[0]
    #     neighbors = np.empty([depth, end - start, sample_size], dtype=np.int32)
    #     for d in range(depth):
    #         neighbors[d] = self.sample_from_adj(adj, start, end, sample_size)
    #     time_ed = time.time()
    #     print("Rank %d sampling time %.3f"%(self.rank, time_ed - time_st))
    #     return neighbors

    def sample_all_depth(self, adj, start, end, depth, sample_size):
        # time_st = time.time()
        if self.refine_threshold < adj.shape[1]:
            adj_sample = pymp.shared.array((end - start, depth * sample_size), dtype='int32')
            with pymp.Parallel(self.num_threads) as p:
                tid = p.thread_num
                # sample_st = time.time()
                # for i in p.range(start, end):  # Option 1: partition [start, end) into disjoint consecutive chunks
                for i in range(start + tid, end, self.num_threads):
                    row = np.squeeze(adj.getrow(i).toarray())
                    if not self.self_wt:
                        row[i] = 0
                    sampled_row = np.nonzero(row)
                    if not self.uniform_sample:
                        weights = np.ravel(row[sampled_row])
                        dist = weights / weights.sum()
                        J, q = alias_setup(dist)
                        for j in range(depth * sample_size):
                            adj_sample[i - start, j] = sampled_row[0][alias_draw(J, q)]
                    else:
                        sample = np.random.randint(sampled_row[0].size, size=depth * sample_size)  # with replacement.
                        adj_sample[i - start, :] = np.take(sampled_row[0], sample)
                # sample_ed = time.time()
                # print("rank %d tid %d time %.3f"%(self.rank, tid, sample_ed - sample_st))
            # time_ed = time.time()
            # print("Rank %d total time for multi-threaded sampling %.3f adj_sample.shape %s"%(self.rank, time_ed - time_st, adj_sample.shape))
        else:
            adj_sample = np.zeros((end - start, depth * sample_size), dtype=np.int32)
            for i in range(start, end):
                row = np.squeeze(adj.getrow(i).toarray())
                if not self.self_wt:
                    row[i] = 0
                sampled_row = np.nonzero(row)
                if not self.uniform_sample:
                    weights = np.ravel(row[sampled_row])
                    dist = weights / weights.sum()
                    J, q = alias_setup(dist)
                    for j in range(depth * sample_size):
                        adj_sample[i - start, j] = sampled_row[0][alias_draw(J, q)]
                else:
                    sample = np.random.randint(sampled_row[0].size, size=depth * sample_size)  # with replacement.
                    adj_sample[i - start, :] = np.take(sampled_row[0], sample)

        adj_sample = np.swapaxes(adj_sample.reshape((end - start, depth, sample_size)), 1, 0)
        return adj_sample

    # def get_neigh_of_batch_nodes(self, adj, batch_nodes, depth, sample_size):
    #     '''
    #     return:
    #     `B_k:       a list of set of nodes sampled at each depth. The order is B1, B2, ..., Bk
    #     `B_neigh:   a list of set of neighbors of nodes in Bk
    #     `B_size:    the sizes of sets Bk
    #     `bk:        B0, unused so far
    #     '''
    #     B_k = []
    #     B_neigh = []
    #     B_size = np.zeros(depth, dtype=np.int32)
    #     bk = list(batch_nodes)
    #     for k in range(depth):
    #         # sample B^k's neighbors
    #         B_size[-k-1] = len(bk)
    #         B_k = np.concatenate((bk, B_k))

    #         neigh_list = self.sample_from_adj(adj, sample_size, node_list=bk)
    #         B_neigh = np.concatenate((neigh_list, B_neigh))

    #         # build B^(k-1) by removing duplicated nodes in neigh_list
    #         bk = list(set(neigh_list))

    #     assert len(B_neigh) == sample_size * len(B_k)
    #     assert len(B_size) == depth

    #     return B_k.astype('int32'), B_neigh.astype('int32'), B_size, bk

    def get_neigh_of_batch_nodes(self, batch_nodes, depth, neighbors, sample_size, vtxdist=None):
        '''
        return:
        `B_k:       a list of set of nodes sampled at each depth. The order is B1, B2, ..., Bk
        `B_neigh:   a list of set of neighbors of nodes in Bk
        `B_size:    the sizes of sets Bk
        `bk:        B0, unused so far
        '''
        rank = self.rank
        procs = self.procs
        # if rank == 0:
        #     timer = Timer(ident=3)
        B_k = []
        B_neigh = []
        B_size = np.zeros(depth, dtype=np.int32)
        # bk = list(batch_nodes)
        bk = batch_nodes
        if self.large:
            all_used_nodes = set(bk)
        else:
            all_used_nodes = None
        # if rank == 0:
        #     timer.printIntervalTime('Initialize')
        for k in range(depth):
            # sample B^k's neighbors
            B_size[-k-1] = len(bk)
            B_k = np.concatenate((bk, B_k))
            # if rank == 0:
            #     timer.printIntervalTime('write bk into B_k')

            if self.sync_neigh:
                # Each thread request neighbors of nodes in bk
                # find how many requests to be sent to other processing units
                num_sendreq = np.zeros(procs, dtype=np.int32)
                num_recvreq = np.zeros(procs, dtype=np.int32)
                for i in range(procs):
                    if i != self.rank:
                        nodes_on_PUs = (bk >= vtxdist[i]) & (bk < vtxdist[i+1])
                        num_nodes_on_PUs = sum(nodes_on_PUs)
                        num_sendreq[i] = num_nodes_on_PUs
                # if rank == 0:
                #     timer.printIntervalTime('compute number of requests to be sent')

                # alltoall
                self.comm.Alltoall(num_sendreq, num_recvreq)
                # if rank == 0:
                #     timer.printIntervalTime('Initialize')

                # send and receive requests
                reqs = []
                reqr = []
                recv_requests = np.empty(np.sum(num_recvreq), dtype=np.int32)
                send_requests = np.empty(np.sum(num_sendreq), dtype=np.int32)
                recv_base, send_base = 0, 0
                for i in range(procs):
                    if num_sendreq[i] > 0:
                        send_requests[send_base:(send_base + num_sendreq[i])] = bk[(bk >= vtxdist[i]) & (bk < vtxdist[i+1])]
                        reqs.append(self.comm.Isend(get_array_buffer(send_requests[send_base:(send_base + num_sendreq[i])]), 
                            dest=i, tag=k*10000000+rank*1000+i))
                        send_base += num_sendreq[i]
                    if num_recvreq[i] > 0:
                        reqr.append(self.comm.Irecv(get_array_buffer(recv_requests[recv_base:(recv_base + num_recvreq[i])]),
                         source=i, tag=k*10000000+i*1000+rank))
                        recv_base += num_recvreq[i]
                MPI.Request.Waitall(reqr)
                MPI.Request.Waitall(reqs)
                # if rank == 0:
                #     timer.printIntervalTime('send and receive requests')

                # response
                recv_base = 0
                reqs = []
                reqr = []
                recv_responses = np.empty((send_base, sample_size))
                send_base = 0

                for i in range(procs):
                    if num_recvreq[i] > 0:
                        reqs.append(self.comm.Isend(get_array_buffer(neighbors[k][recv_requests[recv_base:(recv_base + num_recvreq[i])] - vtxdist[rank]]), 
                            dest=i, tag=k*10000000+1*1000000+rank*1000+i))
                        recv_base += num_recvreq[i]
                    if num_sendreq[i] > 0:
                        reqr.append(self.comm.Irecv(get_array_buffer(recv_responses[send_base:(send_base + num_sendreq[i])]), 
                            source=i, tag=k*10000000+1*1000000+i*1000+rank))
                        send_base += num_sendreq[i]
                MPI.Request.Waitall(reqr)
                MPI.Request.Waitall(reqs)
                # if rank == 0:
                #     timer.printIntervalTime('response')

                # merge
                dict_send = {}
                for i in range(len(send_requests)):
                    dict_send[send_requests[i]] = i
                neigh_list = np.array([neighbors[k][bk_ele - vtxdist[rank]] if (bk_ele >= vtxdist[rank]) & (bk_ele < vtxdist[rank+1]) 
                    else recv_responses[dict_send[bk_ele]] for bk_ele in bk], dtype=np.int32).flatten()
            else:
                # original version: neighbors contains all nodes' neighbors
                neigh_list = (neighbors[k][bk]).flatten()  # May be too long when either or both of len(bk) and num_neighbor is large
            # if rank == 0:
            #     timer.printIntervalTime('build neigh_list')
            B_neigh = np.concatenate((neigh_list, B_neigh))

            # build B^(k-1) by removing duplicated nodes in neigh_list
            set_neigh_list = set(neigh_list)
            if self.large:
                all_used_nodes.update(set_neigh_list)
            if k < depth - 1:
                bk = np.array(list(set_neigh_list))
            # if rank == 0:
            #     timer.printIntervalTime('finalize iteration %d'%(k))

        if self.large:
            all_used_nodes = list(all_used_nodes)
            mapping = dict(zip(all_used_nodes, itertools.count()))
            B_k = np.array([mapping[k] for k in B_k])
            B_neigh = np.array([mapping[k] for k in B_neigh])

        assert len(B_neigh) == self.num_neighbors * len(B_k)
        assert len(B_size) == depth

        # if rank == 0:
        #     timer.restart(title='finalize get_neigh_of_batch_nodes', name='get_neigh_of_batch_nodes')
        return B_k.astype('int32'), B_neigh.astype('int32'), B_size, all_used_nodes

    def train_model(self, session, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        # initialize
        rank = self.rank
        if rank == 0:
            timer = Timer(logger=self.logger, ident=2)  # create the timer
            ''' 
            compute initial_embed
            project the 
            '''
            if coarse_embed is not None:
                initial_embed = fine_graph.C.dot(coarse_embed)
            else:
                initial_embed = fine_embed
            timer.printIntervalTime('compute C.dot')
            early_stopping = self.args.early_stopping
            loss_arr = []  # loss in each epoch
            self.logger.info("initial_embed: " + str(initial_embed.shape))
            self.logger.info("fine_embed: " + str(fine_embed.shape))

            if fine_graph.node_num > self.refine_threshold:
                adj = graph_to_adj_distributed_parallel(fine_graph, num_threads=self.num_threads)
            else:
                adj = graph_to_adj(fine_graph)
            timer.printIntervalTime('compute adj')
        else:
            initial_embed, adj = None, None # graph_to_adj_distributed_parallel(fine_graph.adj_idx, fine_graph.node_num)
        initial_embed = smart_bcast(self.comm, rank, initial_embed, self.mpi_buff, root=0)
        adj = self.comm.bcast(adj, root=0)
        # recvbuf = self.comm.allgather(adj) # allgather adjs distributedly computed by each processing unit
        # adj = sp.vstack(recvbuf)
        if rank == 0:
            timer.printIntervalTime('bcast initial_embed and adj')

        # sample neighbors in parallel
        initial_size = initial_embed.shape[0]
        procs = self.procs
        start, end = int(float(initial_size) / procs * rank), int(float(initial_size) / procs * (rank + 1))
        neighbors = self.sample_all_depth(adj, start, end, self.depth, self.num_neighbors) # return depth * (end-start) * num_neighbors
        
        if rank == 0:
            timer.printIntervalTime('sampling neighbors')

        # gather
        if self.sync_neigh:
            # gather vertex distribution info
            vtxdist = self.comm.gather(end, root=0)
            if rank == 0:
                vtxdist = np.concatenate([[0], vtxdist])
            vtxdist = self.comm.bcast(vtxdist, root=0)
        else:
            # neighbors = self.comm.gather(neighbors, root=0)
            # if rank == 0:
            #     neighbors = np.concatenate(neighbors, axis=1)
            # # broadcast
            # neighbors = self.comm.bcast(neighbors, root=0)
            # recvbuf = self.comm.allgather(neighbors)
            recvbuf = smart_allgather(self.comm, rank, procs, neighbors, self.mpi_buff, root=0)
            recvbuf = np.concatenate(recvbuf, axis=1)
            neighbors = recvbuf

        if rank == 0:
            self.logger.info(neighbors.shape)
            timer.printIntervalTime('allgather neighbors')

        itr = 0
        flag_earlystopping = False
        initial_embed = initial_embed.reshape(-1, self.embed_dim).astype('float32')
        if rank == 0:
            timer.printIntervalTime('Initialization')


        # num_batch = self.batch_per_node * procs
        iterations = int((initial_size - 1) / procs / self.batch_size) + 1
        while not session.should_stop():
            if flag_earlystopping or itr == self.args.epoch:
                break
            # permute all nodes
            loss = 0
            if rank == 0:
                permu = np.random.permutation(initial_size)  # permutation for graph partitioning
            else:
                permu = None
            permu = self.comm.bcast(permu, root=0)  # Bcast for broadcasting NumPy arrays

            rank_loss = 0
            # for idx_batch in range(rank, num_batch, procs):
            for i in range(iterations):
                # get interval of the batch
                # start, end = int(float(initial_size) / num_batch * idx_batch), int(float(initial_size) / num_batch * (idx_batch + 1))
                if i < iterations - 1:
                    start, end = (procs * i + rank) * self.batch_size, (procs * i + rank + 1) * self.batch_size
                else:
                    start, end = int(float(initial_size - procs * i * self.batch_size) / procs * rank), int(float(initial_size - procs * i * self.batch_size) / procs * (rank + 1))
                    start += i * procs * self.batch_size
                    end += i * procs * self.batch_size

                ratio = float(end - start) / initial_size
                batch_nodes = permu[range(start, end)]

                # get 
                B_k, B_neigh, B_size, all_used_nodes = self.get_neigh_of_batch_nodes(batch_nodes, self.depth, neighbors, self.num_neighbors, vtxdist=vtxdist if self.sync_neigh else None)  # Note that B=[Bk, Bk-1, ..., B1]

                # get batch input/output
                batch_initial_embed = initial_embed[batch_nodes]
                batch_fine_embed = fine_embed[batch_nodes]
            
                # train
                optimizer, batch_loss = session.run([self.model_dict['optimizer'], self.model_dict['loss']], feed_dict={
                    self.model_dict['input_embed']: batch_initial_embed,
                    self.model_dict['all_vecs']: initial_embed[all_used_nodes] if self.large else initial_embed,
                    self.model_dict['expected_embed']: batch_fine_embed,
                    self.model_dict['ratio']: ratio,
                    self.model_dict['B_k']: B_k,
                    self.model_dict['B_neigh']: B_neigh,
                    self.model_dict['B_size']: B_size})
                rank_loss += batch_loss * ratio
                # print(itr, rank, rank_loss, batch_loss, ratio, start, end, initial_size)

            loss = self.comm.reduce(rank_loss, op=MPI.SUM, root=0)  # an implicit barrier before reduce

            if rank == 0:
                # timer.printIntervalTime('training')
                loss_arr.append(loss)
                if itr % 20 == 0:
                    self.logger.info("  GraphSAGE iterations-" + str(itr) + ": " + str(loss))
                    GPUtil.showUtilization(all=True)
                if itr > early_stopping and loss_arr[-1] > np.mean(loss_arr[-(early_stopping + 1):-1]):
                    self.logger.info("Early stopping...")
                    flag_earlystopping = True
            flag_earlystopping = self.comm.bcast(flag_earlystopping, root=0)
            itr += 1
            
        if rank == 0:
                timer.printIntervalTime(name='exact time for training')

    def refine_embedding(self, session, coarse_graph=None, fine_graph=None, coarse_embed=None):
        # time_refine_st = time.time()
        rank = self.rank

        # compute adj, all_initial_embed, then broadcast
        if rank == 0:
            timer = Timer(logger=self.logger, ident=2)
            if fine_graph.node_num > self.refine_threshold:
                adj = graph_to_adj_distributed_parallel(fine_graph, num_threads=self.num_threads)
            else:
                adj = graph_to_adj(fine_graph)
            timer.printIntervalTime('compute adj')
            # normalized_A = preprocess_to_gcn_adj(adj, self.args.lda)
            all_initial_embed = fine_graph.C.dot(coarse_embed)
            timer.printIntervalTime('compute C.dot')
        else:
            adj, all_initial_embed = None, None
        all_initial_embed = smart_bcast(self.comm, rank, all_initial_embed, self.mpi_buff, root=0)
        adj = self.comm.bcast(adj, root=0)
        initial_size = all_initial_embed.shape[0]
        if rank == 0:
            timer.printIntervalTime('bcast initial_embed')

        # sample neighbors in parallel, then broadcast
        num_batch = self.procs
        start, end = int(float(initial_size) / num_batch * rank), int(float(initial_size) / num_batch * (rank + 1))
        neighbors = self.sample_all_depth(adj, start, end, self.depth, self.num_neighbors)
        if rank == 0:
            self.logger.info(neighbors.shape)
            timer.printIntervalTime('sampling neighbors')

        if self.sync_neigh:
            # gather vertex distribution info
            vtxdist = self.comm.gather(end, root=0)
            if rank == 0:
                vtxdist = np.concatenate([[0], vtxdist])
            vtxdist = self.comm.bcast(vtxdist, root=0)
        else:
            # neighbors = self.comm.gather(neighbors, root=0)
            # if rank == 0:
            #     neighbors = np.concatenate(neighbors, axis=1)
            # # broadcast
            # neighbors = self.comm.bcast(neighbors, root=0)
            time_allgather_st = time.time()
            # recvbuf = self.comm.allgather(neighbors)
            recvbuf = smart_allgather(self.comm, rank, self.procs, neighbors, self.mpi_buff, root=0)
            time_allgather_ed = time.time()
            print("rank %d time for allgather %.3f"%(rank, time_allgather_ed - time_allgather_st))
            time_allgather_st = time.time()
            recvbuf = np.concatenate(recvbuf, axis=1)
            print("rank %d time for concatenate %.3f"%(rank, time_allgather_ed - time_allgather_st))
            neighbors = recvbuf

        if rank == 0:
            timer.printIntervalTime('allgather neighbors')

        local_size = end - start
        batch_size = min(self.batch_size, local_size)  # batch_size is local while self.batch_size is a parameter
        num_batch = int(float(local_size - 1) / batch_size) + 1
        refined_embed = np.empty((local_size, self.embed_dim))
        # time_refine = 0
        for i in range(num_batch):
            # time_st = time.time()
            b_start, b_end = batch_size * i + start, min(batch_size * (i + 1) + start, end)
            batch_nodes = range(b_start, b_end)
            # print("initial_size %d - rank %d start %d end %d, i %d / num_batch %d, b_start %d b_end %d"%(initial_size, rank, start, end, i, num_batch, b_start, b_end))
            batch_initial_embed = all_initial_embed[b_start:b_end]
            B_k, B_neigh, B_size, all_used_nodes = self.get_neigh_of_batch_nodes(range(b_start, b_end), self.depth, neighbors, self.num_neighbors, vtxdist=vtxdist if self.sync_neigh else None)
            refined_embed[(b_start - start):(b_end - start)], = session.run([self.model_dict['pred_embed']], feed_dict={
                self.model_dict['input_embed']: batch_initial_embed,
                self.model_dict['all_vecs']: all_initial_embed[all_used_nodes] if self.large else all_initial_embed,
                self.model_dict['B_k']: B_k,
                self.model_dict['B_neigh']: B_neigh,
                self.model_dict['B_size']: B_size})
            GPUtil.showUtilization(all=True)
            # time_refine += time.time() - time_st

        if rank == 0:
            timer.printIntervalTime('applying once')
        # print("\t|Rank %d time for applying %.3f"%(rank, time_refine))

        # gather
        # refined_embed = self.comm.gather(refined_embed, root=0)
        refined_embed = smart_gather(self.comm, rank, self.procs, refined_embed, self.mpi_buff, root=0)
        if rank == 0:
            refined_embed = np.concatenate(refined_embed, axis=0)
            timer.printIntervalTime('gather refined_embed')
            # print("Rank %d total time for refinement %.3f"%(rank, time.time() - time_refine_st))
            return refined_embed
        else:
            # print("Rank %d total time for refinement %.3f"%(rank, time.time() - time_refine_st))
            return None
