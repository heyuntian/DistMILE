from multiprocessing import Process, Semaphore, Value
import numpy as np
import math

class Barrier:
    def __init__(self, n):
        self.n       = n
        self.count   = Value('i', 0)
        self.mutex   = Semaphore(1)
        self.barrier = Semaphore(0)

    def wait(self):
        self.mutex.acquire()
        self.count.value += 1
        self.mutex.release()

        if self.count.value == self.n:
            self.barrier.release()

        self.barrier.acquire()
        self.barrier.release()

def balancedNodePartition(adj_idx, num_part):
    targets = np.array([(int(float(adj_idx[-1]) / num_part * i)) for i in range(num_part + 1)])
    indices = np.searchsorted(adj_idx, targets)
    return indices

def bNP_degree(degree, num_part):
    targets = np.zeros(num_part + 1, dtype=np.int64)
    max_v = np.max(degree)
    min_v = np.min(degree)
    # targets = np.rint(min_v + (max_v - min_v) / num_part * np.arange(num_part+1)).astype(np.int64)
    for i in range(1, num_part + 1):
        # targets[i] = max(targets[i-1], degree[int(float(len(degree)) / num_part * i) - 1]) + 1
        targets[i] = degree[int(float(len(degree)) / num_part * i) - 1] + 1
    indices = np.searchsorted(degree, targets)
    # print("targets %s"%(targets))
    # print("indices %s"%(indices))
    return indices

def smart_bcast(comm, rank, obj, mpi_buff, root=0):
    # determine if dividing is needed
    flag_needed = False
    is_numpy = False
    result = None
    if rank == root:
        if 'numpy' in str(type(obj)):
            is_numpy = True
            r, c = obj.shape
            dtype = obj.dtype
            MEM_SIZE = mpi_buff
            ELE_SIZE = int(str(dtype)[-2:]) / 8
            if ELE_SIZE * r * c > MEM_SIZE:
                flag_needed = True
    
    flag_needed, is_numpy = comm.bcast([flag_needed, is_numpy], root=root)

    if not flag_needed:
        result = comm.bcast(obj, root=root)
        return result

    if rank == root:
        if is_numpy:
            MEM_LIMIT_ROW = MEM_SIZE / c / ELE_SIZE
            row_idx = [int(i * MEM_LIMIT_ROW) for i in range(int(math.ceil(float(r) / MEM_LIMIT_ROW)))]
            num_mtx = len(row_idx)
            row_idx.append(r)
            print(row_idx, num_mtx)
    else:
        if is_numpy:
            num_mtx = None
    num_mtx = comm.bcast(num_mtx, root=root)

    list_mtx = list()
    for i in range(num_mtx):
        tmp = None
        if rank == root:
            if is_numpy:
                tmp = obj[row_idx[i]:row_idx[i+1]]
        tmp = comm.bcast(tmp, root=root)
        if rank != root:
            list_mtx.append(tmp)
    
    if rank == root:
        return obj
    else:
        if is_numpy:
            return np.vstack(list_mtx)

def smart_gather(comm, rank, procs, obj, mpi_buff, root=0):
    flag_needed = False
    MEM_LIMIT_ROW = 0
    if rank == root:
        r, c = obj.shape
        dtype = obj.dtype
        MEM_SIZE = mpi_buff
        ELE_SIZE = int(str(dtype)[-2:]) / 8
        if ELE_SIZE * r * c * procs > MEM_SIZE:
            flag_needed = True
            MEM_LIMIT_ROW = MEM_SIZE / procs / c / ELE_SIZE

    flag_needed, MEM_LIMIT_ROW = comm.bcast([flag_needed, MEM_LIMIT_ROW], root=root)

    if not flag_needed:
        result = comm.gather(obj, root=root)
        return result

    r, c = obj.shape
    row_idx = [int(i * MEM_LIMIT_ROW) for i in range(int(math.ceil(float(r) / MEM_LIMIT_ROW)))]
    num_mtx = len(row_idx)
    row_idx.append(r)
    # print("rank %d row_idx %s num_mtx %d"%(rank, row_idx, num_mtx))

    ll_mtx = None
    if rank == root:
        ll_mtx = [list() for i in range(procs)]
    for i in range(num_mtx):
        tmp = comm.gather(obj[row_idx[i]:row_idx[i+1]], root=root)
        if rank == root:
            for j in range(procs):
                ll_mtx[j].append(tmp[j])
                # print("rank %d get %s from rank %d"%(root, tmp[j].shape, j))

    if rank != root:
        return None
    else:
        return [np.vstack(ll_mtx[j]) for j in range(procs)]

def smart_allgather(comm, rank, procs, obj, mpi_buff, root=0):
    flag_needed = False
    MEM_LIMIT_ROW = 0
    if rank == root:
        d, r, c = obj.shape
        dtype = obj.dtype
        MEM_SIZE = mpi_buff
        ELE_SIZE = int(str(dtype)[-2:]) / 8
        if ELE_SIZE * d * r * c * procs > MEM_SIZE:
            flag_needed = True
            MEM_LIMIT_ROW = MEM_SIZE / d / c / procs / ELE_SIZE

    flag_needed, MEM_LIMIT_ROW = comm.bcast([flag_needed, MEM_LIMIT_ROW], root=root)

    if not flag_needed:
        result = comm.allgather(obj)
        return result

    d, r, c = obj.shape
    row_idx = [int(i * MEM_LIMIT_ROW) for i in range(int(math.ceil(float(r) / MEM_LIMIT_ROW)))]
    num_mtx = len(row_idx)
    row_idx.append(r)
    print("rank %d row_idx %s num_mtx %d"%(rank, row_idx, num_mtx))

    ll_mtx = [list() for i in range(procs)]
    for i in range(num_mtx):
        tmp = comm.allgather(obj[:, row_idx[i]:row_idx[i+1]])
        for j in range(procs):
            ll_mtx[j].append(tmp[j])

    return [np.concatenate(ll_mtx[j], axis=1) for j in range(procs)]