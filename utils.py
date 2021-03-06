import numpy as np
import pandas as pd
import torch

def T(a):
    if torch.is_tensor(a):
        res = a
    else:
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            res = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            res = torch.FloatTensor(a.astype(np.float32))
        else:
            raise NotImplementedError(a.dtype)
    return to_gpu(res)

USE_GPU=False
def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if torch.cuda.is_available() and USE_GPU else x

def create_variable(x, volatile, requires_grad=False):
    if type (x) != Variable:
        if IS_TORCH_04: x = Variable(T(x), requires_grad=requires_grad)
        else:           x = Variable(T(x), requires_grad=requires_grad, volatile=volatile)
    return x

def V_(x, requires_grad=False, volatile=False):
    '''equivalent to create_variable, which creates a pytorch tensor'''
    return create_variable(x, volatile=volatile, requires_grad=requires_grad)
def V(x, requires_grad=False, volatile=False):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, lambda o: V_(o, requires_grad, volatile))

def accuracy(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()