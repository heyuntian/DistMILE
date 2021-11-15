import numpy as np
import tensorflow as tf

# A Control instance stores most of the configuration information.
class Control:
    def __init__(self):
        self.data = None
        self.workers = 4
        self.coarsen_k = 0
        self.coarsen_to = 500  # 1000
        self.coarsen_level = 0  #
        self.max_node_wgt = 100  # to avoid super-node being too large.
        self.embed_dim = 128
        self.basic_embed = "DEEPWALK"
        self.refine_type = "MD-gcn"
        self.refine_model = RefineModelSetting()
        self.embed_time = 0.0  # keep track of the amount of time spent for embedding.
        self.debug_mode = False  # set to false for time measurement.
        self.logger = None
        self.num_threads = 28
        self.coarse_threshold = 10000
        self.unique_threshold = 1000000

        # For SDNE
        self.alpha = 2e-1
        self.beta = 10
        self.epoch = 5
        self.sdne_hd_sz = [300, 500]


class RefineModelSetting:
    def __init__(self):
        self.double_base = False
        self.learning_rate = 0.001
        self.epoch = 200
        self.early_stopping = 50  # Tolerance for early stopping (# of epochs).
        self.wgt_decay = 5e-4
        self.regularized = True
        self.hidden_layer_num = 2
        self.act_func = tf.tanh
        self.tf_optimizer = tf.train.AdamOptimizer
        self.lda = 0.05  # self-loop weight lambda
        self.untrained_model = False  # if set. The model will be untrained.
        self.large = False
        self.sync_neigh = False

        # The following ones are for GraphSAGE only.
        self.gs_sample_neighbrs_num = 100
        self.gs_depth = 1
        self.gs_mlp_layer = 2
        self.gs_concat = True
        self.gs_uniform_sample = False
        self.gs_self_wt = True
        self.gs_mpi_buff = 2e9
        # self.gs_batch_per_node = 1
        self.batch_size = 1e5  # this batch size is only for applying the refinement model
        self.refine_threshold = 2000