import numpy as np
import torch
from sklearn.metrics import pairwise_distances
import os


def obtainDataset(dataset_name, distance_metric, dataset_path):
    pass


def computeSimilarity(dataset_name, distance_metric, dataset_path=None):
    if os.path.isfile(f'{dataset_name}/similart_matrix_{distance_metric}.npy'):
        S = np.load(f'{dataset_name}/similart_matrix_{distance_metric}.npy')
    else:
        S = obtainDataset(dataset_name, distance_metric, dataset_path)

    return S