import numpy as np
import pandas as pd

import Algorithms
from Algorithms import *
import Utils
import plotly


def conductExperiments(problem_type, dataset_path, dataset_name, distance_metric, mode, T, eps, t_s, run_loay=False):
    S = Utils.computeSimilarity(dataset_name=dataset_name,
                                distance_metric=distance_metric,
                                problem_type=problem_type,
                                dataset_path=dataset_path)
    F, gradF = Utils.defineF(problem_type, mode=mode, p=0.0001, W=S)
    K_N, K_D, K = Utils.defineKNKD(problem_type=problem_type, d=S.shape[0])
    if not run_loay:
        solution, vals = Algorithms.FWContinousGreedyHybrid(F, K_N, K_D, T, eps, t_s, gradF=gradF,
                                                            dataset_name=dataset_name)
    else:
        solution, vals = Algorithms.loaySolver(F, K, T, eps, gradF=gradF, dataset_name=dataset_name, S=S)

    return np.array(vals)


if __name__ == '__main__':
    problem_type = Utils.PROBLEMTYPES.REVENUE_MAXIMIZATION
    dataset_path = r'C:\Users\murad\Downloads\network.csv\edges.csv'
    dataset_name = 'network'
    distance_metric = ''
    run_loay = False
    T = 1 if not run_loay else 100
    eps = 0.01 if not run_loay else 0.01
    t_s = np.linspace(start=0, stop=1, num=int(1/eps))
    mode = Utils.MODE.OFFLINE
    if hasattr(t_s, "__len__") and not run_loay:
        max_val = -np.inf
        max_val_container = np.empty((t_s.shape[0], 2))
        best_ts = None
        for i in range(len(t_s)):
            temp_val = conductExperiments(problem_type, dataset_path, dataset_name, distance_metric, mode, T, eps,
                                          t_s[i], run_loay=run_loay)
            max_val_container[i, 0] = t_s[i]
            max_val_container[i, 1] = temp_val.max()

        np.savez(f'data/{dataset_name}/vals.npz', vals=max_val_container, ts=best_ts)
    else:
        max_val_container = conductExperiments(problem_type, dataset_path, dataset_name, distance_metric, mode, T,
                                               eps, t_s, run_loay=run_loay)
        if run_loay:
            np.save(f'data/{dataset_name}/loay_vals.npy', max_val_container)
        else:
            np.savez(f'data/{dataset_name}/vals.npz', vals=max_val_container, ts=t_s)
