import numpy as np
from mpi4py import MPI

def get_array_buffer(arr):
    MPI_TYPE_MAP = {
        'int8': MPI.CHAR,
        'int16': MPI.SHORT,
        'int32': MPI.INT,
        'int64': MPI.LONG,
        'int128': MPI.LONG_LONG,
        'float32': MPI.FLOAT,
        'float64': MPI.DOUBLE,
        'bool': MPI.BOOL,
    }

    return [arr, arr.size, MPI_TYPE_MAP[str(arr.dtype)]]
