import numpy as np
from mpi4py import MPI
from mpiutils import get_array_buffer

class Graph(object):
    ''' Note: adj_list shows each edge twice. So edge_num is really two times of edge number for undirected graph.'''

    def __init__(self, node_num, edge_num):
        self.node_num = node_num  # n
        self.edge_num = edge_num  # m
        self.adj_list = np.zeros(edge_num, dtype=np.int32) - 1  # a big array for all the neighbors.
        self.adj_idx = np.zeros(node_num + 1,
                                dtype=np.int32)  # idx of the beginning neighbors in the adj_list. Pad one additional element at the end with value equal to the edge_num, i.e., self.adj_idx[-1] = edge_num
        self.adj_wgt = np.zeros(edge_num,
                                dtype=np.float32)  # same dimension as adj_list, wgt on the edge. CAN be float numbers.
        self.node_wgt = np.zeros(node_num, dtype=np.float32)
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1 # None  # mapped to coarser graph

        # weighted degree: the sum of the adjacency weight of each vertex, including self-loop.
        self.degree = np.zeros(node_num, dtype=np.float32)
        self.A = None
        self.C = None  # Matching Matrix

        self.coarser = None
        self.finer = None

    def resize_adj(self, edge_num):
        '''Resize the adjacency list/wgts based on the number of edges.'''
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx):
        '''obtain the list of neigbors given a node.'''
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_list[istart:iend]

    def get_neigh_edge_wgts(self, idx):
        '''obtain the weights of neighbors given a node.'''
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_wgt[istart:iend]

class Subgraph(object):

    def __init__(self, graph, procs, rank, comm):
        self.procs = procs
        self.rank = rank
        self.global_node_num = graph.node_num
        self.vtxdist = np.asarray([int(float(self.global_node_num) / procs * i) for i in range(procs + 1)])
        self.node_num = self.vtxdist[rank + 1] - self.vtxdist[rank]

        # build adj_list and adj_idx
        st, ed = self.vtxdist[rank], self.vtxdist[rank + 1]
        edge_base = graph.adj_idx[st]
        edge_end = graph.adj_idx[ed]
        self.adj_idx = np.copy(graph.adj_idx[st:(ed + 1)]) - edge_base
        self.edge_num = self.adj_idx[-1]
        self.adj_wgt = np.copy(graph.adj_wgt[edge_base:edge_end])
        self.adj_list = np.copy(graph.adj_list[graph.adj_idx[st]:edge_end])
        self.node_wgt = np.copy(graph.node_wgt[st:ed])
        self.st = st
        self.ed = ed
        assert self.edge_num == edge_end - edge_base, "subgraph %d edge_num"%(rank)

        ''' reset adj_list
        For edge i -> k,
        local edge : i -> k - firstvtx
        outgoing edge: i -> node_num + range(nrecv), nrecv is # remote vertices connected to the subgraph

        recvind: mapping from 0..recv-1 to global vertix id
        '''
        bool_local_edge = (self.adj_list >= st) & (self.adj_list < ed)
        self.adj_list[bool_local_edge] -= st
        recvind, recvG2L = np.unique(self.adj_list[np.invert(bool_local_edge)], return_inverse=True)  # return a set of unique outgoing neighbor nodes, and a mapping from values in param 0 to the index in the set
        self.nrecv = len(recvind)
        self.adj_list[np.invert(bool_local_edge)] = self.node_num + recvG2L
        self.degree = np.concatenate([graph.degree[st:ed], graph.degree[recvind]])

        ''' determine the number of neighboring processors
        '''
        self.nnbrs = 0
        idx_proc = -1
        peind = []
        recvptr = []
        for i in range(self.nrecv):
            if (recvind[i] >= self.vtxdist[idx_proc + 1]):
                while True:
                    idx_proc += 1
                    if recvind[i] < self.vtxdist[idx_proc + 1]:
                        break
                peind.append(idx_proc)
                recvptr.append(i)
                self.nnbrs += 1
        recvptr.append(self.nrecv)
        peind = np.asarray(peind)
        recvptr = np.asarray(recvptr)
        assert self.nnbrs == len(peind), "nnbrs == len(peind)"
        assert self.nnbrs == len(recvptr) - 1, "nnbrs == len(recvptr) - 1"
        self.recvind = recvind
        self.peind = peind
        self.recvptr = recvptr

        '''
        recvptr: 0..nnbrs -> 0..nrecv, the initial index of outgoing neighbor vertices in nrecv for each processor
        '''
        recvrequests = np.empty((procs, 2), dtype=np.int32)
        recvrequests[peind] = [[recvptr[i + 1] - recvptr[i], self.node_num + recvptr[i]] for i in range(self.nnbrs)]
        sendrequests = np.empty((procs, 2), dtype=np.int32)
        comm.Alltoall(recvrequests, sendrequests)

        self.recvrequests = recvrequests
        self.sendrequests = sendrequests

        '''
        sendptr: cumulative number of nodes connected to other subgraphs (0..nnbrs -> 0..nsend)
        startsind: the index base of nodes connected to other subgraphs in the corresponding subgraph
        '''
        sendptr = np.copy(sendrequests[peind, 0])
        sendptr = self.makeCSR(sendptr)
        startsind = np.copy(sendrequests[peind, 1])
        self.nsend = sendptr[self.nnbrs]

        self.sendptr = sendptr

        '''
        build sendind
        sendind: the global indices of vertices connected to every other processor
        '''
        self.sendind = np.empty(self.nsend, dtype=np.int32)
        reqs = []
        for i in range(self.nnbrs):
            reqs.append(comm.Isend(get_array_buffer(self.recvind[recvptr[i]:recvptr[i+1]]), dest=peind[i], tag=100*(rank+1)+peind[i]))
        reqr = []
        for i in range(self.nnbrs):
            reqr.append(comm.Irecv(get_array_buffer(self.sendind[sendptr[i]:sendptr[i+1]]), source=peind[i], tag=100*(peind[i]+1)+rank))
        MPI.Request.Waitall(reqs + reqr)

        '''
        create the peadjcny data structure for sparse boundary exchanges
        '''
        self.pexadj = np.zeros(self.node_num, dtype=np.int32)
        self.peadjcny = np.empty(self.nsend, dtype=np.int32)
        self.peadjloc = np.empty(self.nsend, dtype=np.int32)
        unique_vals, counts = np.unique(self.sendind, return_counts=True)
        self.pexadj[unique_vals - st] = counts
        self.pexadj = self.makeCSR(self.pexadj)
        assert self.pexadj[self.node_num] == self.nsend, "pexadj[nvtxs] = %d while nsend = %d"%(self.pexadj[self.node_num], self.nsend)

        for i in range(self.nnbrs):
            for j in range(sendptr[i], sendptr[i+1]):
                k = self.pexadj[self.sendind[j] - st]
                self.pexadj[self.sendind[j] - st] += 1
                self.peadjcny[k] = i
                self.peadjloc[k] = startsind[i]
                startsind[i] += 1
        self.shiftCSR(self.pexadj)

        '''
        imap: mapping from normal index to global index
        '''
        self.imap = np.concatenate([np.arange(self.node_num) + st, self.recvind])
        assert len(self.pexadj) == self.node_num + 1
        assert len(self.imap) == self.node_num + self.nrecv


    def makeCSR(self, array):
        return np.cumsum(np.pad(array, (1, 0), 'constant'))

    def shiftCSR(self, array):
        n = len(array) - 1
        for i in range(n, 0, -1):
            array[i] = array[i-1]
        array[0] = 0
        return array
