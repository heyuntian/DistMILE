from sdne_gem import SDNE
import time

from utils import graph2nx_weighted #, nx_renum_graph
import networkx as nx
import tensorflow as tf
import numpy as np
# from openne_graph import Graph as OpenNEGraph


# For dynamic loading: name of this method should be the same as name of this python FILE.
def sdne(ctrl, graph):
    '''Use GraRep as the base embedding method. This is a wrapper method and used by MILE.'''
    args = SDNESetting(ctrl)

    # graph, old2new, new2old = nx_renum_graph(graph2nx_weighted(graph))
    graph = graph2nx_weighted(graph)

    # graph = OpenNEGraph(graph=graph)
    # graph.encode_node()

    return SDNE_Original(graph, args, ctrl).get_embeddings()


#
class SDNESetting:
    '''Configuration parameters for GraRep.'''

    def __init__(self, ctrl):
        self.dim = ctrl.embed_dim
        self.beta = ctrl.beta
        self.alpha = ctrl.alpha
        self.nu1 = 1e-5
        self.nu2 = 1e-4
        self.batch_size = 1000
        self.epoch = ctrl.epoch
        self.encoder_layer_list = [ctrl.sdne_hd_sz, 128]
        self.learning_rate = 0.01


#
#
# def fc_op(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
#     n_in = input_op.get_shape()[-1].value
#     with tf.name_scope(name) as scope:
#         kernel = tf.Variable(tf.contrib.layers.xavier_initializer()([n_in, n_out]), dtype=tf.float32, name=scope + "w")
#
#         # kernel = tf.Variable(tf.random_normal([n_in, n_out]))
#         biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'b')
#
#         fc = tf.add(tf.matmul(input_op, kernel), biases)
#         activation = act_func(fc, name=scope + 'act')
#         layer_collector.append([kernel, biases])
#         return activation
#
#
# class SDNE_Original(object):
#     def __init__(self, graph, args, ctrl):
#         """
#         encoder_layer_list: a list of numbers of the neuron at each ecdoer layer, the last number is the
#         dimension of the output node representation
#         Eg:
#         if node size is 2000, encoder_layer_list=[1000, 128], then the whole neural network would be
#         2000(input)->1000->128->1000->2000, SDNE extract the middle layer as the node representation
#         """
#         self.g = graph
#         encoder_layer_list = args.encoder_layer_list
#
#         self.node_size = self.g.G.number_of_nodes()
#         self.dim = encoder_layer_list[-1]
#
#         self.encoder_layer_list = [self.node_size]
#         self.encoder_layer_list.extend(encoder_layer_list)
#         self.encoder_layer_num = len(encoder_layer_list) + 1
#
#         self.alpha = args.alpha
#         self.beta = args.beta
#         self.nu1 = args.nu1
#         self.nu2 = args.nu2
#         self.bs = args.batch_size
#         self.epoch = args.epoch
#         self.max_iter = (args.epoch * self.node_size) // args.batch_size
#         print("Max iter", self.max_iter)
#
#         self.lr = args.learning_rate
#         if self.lr is None:
#             self.lr = tf.train.inverse_time_decay(0.03, self.max_iter, decay_steps=1, decay_rate=0.9999)
#
#         self.sess = tf.Session()
#         self.vectors = {}
#
#         self.adj_mat = self.getAdj()
#         self.embeddings = self.train()
#
#         look_back = self.g.look_back_list
#
#         for i, embedding in enumerate(self.embeddings):
#             self.vectors[look_back[i]] = embedding
#
#     def getAdj(self):
#         node_size = self.g.node_size
#         look_up = self.g.look_up_dict
#         adj = np.zeros((node_size, node_size))
#         for edge in self.g.G.edges():
#             adj[look_up[edge[0]]][look_up[edge[1]]] = self.g.G[edge[0]][edge[1]]['weight']
#         return adj
#
#     def train(self):
#         adj_mat = self.adj_mat
#
#         AdjBatch = tf.placeholder(tf.float32, [None, self.node_size], name='adj_batch')
#         Adj = tf.placeholder(tf.float32, [None, None], name='adj_mat')
#         B = tf.placeholder(tf.float32, [None, self.node_size], name='b_mat')
#
#         fc = AdjBatch
#         scope_name = 'encoder'
#         layer_collector = []
#
#         with tf.name_scope(scope_name):
#             for i in range(1, self.encoder_layer_num):
#                 fc = fc_op(fc,
#                            name=scope_name + str(i),
#                            n_out=self.encoder_layer_list[i],
#                            layer_collector=layer_collector)
#
#         _embeddings = fc
#
#         scope_name = 'decoder'
#         with tf.name_scope(scope_name):
#             for i in range(self.encoder_layer_num - 2, 0, -1):
#                 fc = fc_op(fc,
#                            name=scope_name + str(i),
#                            n_out=self.encoder_layer_list[i],
#                            layer_collector=layer_collector)
#             fc = fc_op(fc,
#                        name=scope_name + str(0),
#                        n_out=self.encoder_layer_list[0],
#                        layer_collector=layer_collector, )
#
#         _embeddings_norm = tf.reduce_sum(tf.square(_embeddings), 1, keepdims=True)
#
#         L_1st = tf.reduce_sum(
#             Adj * (
#                     _embeddings_norm - 2 * tf.matmul(
#                 _embeddings, tf.transpose(_embeddings)
#             ) + tf.transpose(_embeddings_norm)
#             )
#         )
#
#         L_2nd = tf.reduce_sum(tf.square((AdjBatch - fc) * B))
#
#         L = L_2nd + self.alpha * L_1st
#
#         for param in layer_collector:
#             L += self.nu1 * tf.reduce_sum(tf.abs(param[0])) + self.nu2 * tf.reduce_sum(tf.square(param[0]))
#
#         optimizer = tf.train.AdamOptimizer(self.lr)
#
#         train_op = optimizer.minimize(L)
#
#         init = tf.global_variables_initializer()
#         self.sess.run(init)
#
#         print("total iter: %i" % self.max_iter)
#         for step in range(self.max_iter):
#             index = np.random.randint(self.node_size, size=self.bs)
#             adj_batch_train = adj_mat[index, :]
#             adj_mat_train = adj_batch_train[:, index]
#             b_mat_train = np.ones_like(adj_batch_train)
#             b_mat_train[adj_batch_train != 0] = self.beta
#
#             self.sess.run(train_op, feed_dict={AdjBatch: adj_batch_train,
#                                                Adj: adj_mat_train,
#                                                B: b_mat_train})
#             if step % 50 == 0:
#                 l, l1, l2 = self.sess.run((L, L_1st, L_2nd),
#                                           feed_dict={AdjBatch: adj_batch_train,
#                                                      Adj: adj_mat_train,
#                                                      B: b_mat_train})
#                 print("step %i: total loss: %s, l1 loss: %s, l2 loss: %s" % (step, l, l1, l2))
#
#         return self.sess.run(_embeddings, feed_dict={AdjBatch: adj_mat})
#
#     def save_embeddings(self, filename):
#         fout = open(filename, 'w')
#         node_num = len(self.vectors)
#         fout.write("{} {}\n".format(node_num, self.dim))
#         for node, vec in self.vectors.items():
#             fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
#         fout.close()
#
#     def get_embeddings(self):
#         node_num = len(self.vectors)
#         embeds = np.zeros((node_num, self.dim))
#         for node, vec in self.vectors.items():
#             embeds[node][:] = vec
#         return embeds
# #

class SDNE_Original(object):
    '''This is the original implementation of GraRep. Code is adapted from https://github.com/thunlp/OpenNE.'''

    def __init__(self, graph, args, ctrl, logger=None):
        self.logger = logger
        self.graph = graph
        # param = "%s_%s_%s" % (ctrl.data.split("/")[-1], ctrl.coarsen_level, ctrl.task_name)
        config = tf.ConfigProto(intra_op_parallelism_threads=ctrl.num_threads)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.model = SDNE(d=ctrl.embed_dim, beta=ctrl.beta, alpha=ctrl.alpha, nu1=1e-6, nu2=1e-6, K=2,
                          n_units=ctrl.sdne_hd_sz, rho=0.3, n_iter=ctrl.epoch, xeta=0.01, n_batch=128, session=sess)
        # modelfile=['enc_model_%s.json' % param, 'dec_model_%s.json' % param],
        #                           weightfile=['enc_weights_%s.hdf5' % param, 'dec_weights_%s.hdf5' % param]
        self.train()

    def train(self):
        t1 = time.time()
        # Learn embedding - accepts a networkx graph or file with edge list
        self.vectors, t = self.model.learn_embedding(graph=self.graph, edge_f=None, is_weighted=True, no_python=True)

    def get_embeddings(self):
        del self.model
        del self.graph
        return self.vectors
