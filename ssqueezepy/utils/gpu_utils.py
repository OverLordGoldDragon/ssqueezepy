# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple
from string import Template
from .backend import torch, cp


Stream = namedtuple('Stream', ['ptr'])

def _run_on_gpu(kernel, grid, block, *args, **kwargs):
    kernel_name = kernel.split('void ')[1].split('(')[0]
    fn = load_kernel(kernel_name, kernel, **kwargs)
    fn(grid=grid, block=block, args=args,
       stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))


@cp.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cp.RawModule(code=code)
    return kernel_code.get_function(kernel_name)


def _get_kernel_params(x, dim=1, threadsperblock=None):
    M, N = x.shape[:2]

    if dim == 1:
        threadsperblock = threadsperblock or (1024,)
        blockspergrid = (int(np.ceil(M * N / threadsperblock[0])),)
    elif dim == 2:
        threadsperblock = threadsperblock or (32, 32)
        blockspergrid_x = int(np.ceil(M / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(N / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

    dtype = ('double' if x.dtype in (torch.float64, torch.complex128) else
             'float')
    kernel_kw = dict(dtype=dtype, M=M, N=N)
    str_dtype = 'float32' if dtype == 'float' else 'float64'
    return blockspergrid, threadsperblock, kernel_kw, str_dtype
