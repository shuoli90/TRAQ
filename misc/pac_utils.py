import os, sys
import types
import time
import numpy as np
import itertools
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch as tc
from torch import Tensor as T

    
def log_factorial(n):
    log_f = tc.arange(n, 0, -1).float().log().sum()
    return log_f

def log_n_choose_k(n, k):
    if k == 0:
        return tc.tensor(1)
    else:
        #res = log_factorial(n) - log_factorial(k) - log_factorial(n-k)
        res = tc.arange(n, n-k, -1).float().log().sum() - log_factorial(k)
        return res

def half_line_bound_upto_k(n, k, eps):
    ubs = []
    eps = tc.tensor(eps)
    for i in tc.arange(0, k+1):
        bc_log = log_n_choose_k(n, i)
        log_ub = bc_log + eps.log()*i + (1.0-eps).log()*(n-i)
        ubs.append(log_ub.exp().unsqueeze(0))
    ubs = tc.cat(ubs)
    ub = ubs.sum()
    return ub


def find_maximum_train_error_allow(eps, delta, n):
    k_min = 0
    k_max = n
    bnd_min = half_line_bound_upto_k(n, k_min, eps)
    if bnd_min > delta:
        return None
    assert(bnd_min <= delta)
    k = n
    while True:
        # choose new k
        k_prev = k
        k = (np.array((k_min + k_max))/2.0).round()
    
        # terinate condition
        if k == k_prev:
            break
    
        # check whether the current k satisfies the condition
        bnd = half_line_bound_upto_k(n, k, eps)
        if bnd <= delta:
            k_min = k
        else:
            k_max = k

    # confirm that the solution satisfies the condition
    k_best = k_min
    assert(half_line_bound_upto_k(n, k_best, eps) <= delta)
    error_allow = float(k_best) / float(n)
    return error_allow

if __name__ == "__main__":
    error_allow = find_maximum_train_error_allow(0.1, 0.1, 100)