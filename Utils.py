import copy
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
import os
import torchvision
from enum import Enum
import pandas as pd
import plotly


class PROBLEMTYPES(Enum):
    LOCATION_SUMMARIZATION = 1
    IMAGE_SUMMARIZATION = 2
    REVENUE_MAXIMIZATION = 3


class MODE(Enum):
    OFFLINE = 1
    ONLINE = 2


def checkDirectoryIfExist(dataset_name):
    return os.path.isdir(f'data/{dataset_name}/')


def prepareCIFAR10(distance_metric, dataset_name):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=f'data/{dataset_name}/', train=True, download=True, transform=transform_train)

    D = trainset.data
    D = D.reshape(-1, D.shape[0]).T

    S = pairwise_distances(D, D, metric=distance_metric)
    np.save(f'data/{dataset_name}/similarity_matrix_{distance_metric}.npy', S)


def prepareCIFAR100(distance_metric, dataset_name):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    trainset = torchvision.datasets.CIFAR100(root=f'data/{dataset_name}/', train=True, download=True,
                                             transform=transform_train)

    D = trainset.data
    D = D.reshape(-1, D.shape[0]).T

    S = pairwise_distances(D, D, metric=distance_metric)
    np.save(f'data/{dataset_name}/similarity_matrix_{distance_metric}.npy', S)


def obtainDataset(dataset_name, distance_metric, dataset_path, problem_type):
    if not checkDirectoryIfExist(dataset_name):
        os.makedirs(f'data/{dataset_name}/')

    if 'CIFAR' in dataset_name:
        if '10' in dataset_name:
            prepareCIFAR10(distance_metric, dataset_name)
        elif '100' in dataset_name:
            prepareCIFAR100(distance_metric, dataset_name)
        else:
            raise NotImplementedError('Please implement proper data reading method')
    else:
        if problem_type == PROBLEMTYPES.REVENUE_MAXIMIZATION:
            D = pd.read_csv(dataset_path, header=None)
            if 'edges' in dataset_path:
                D = D.iloc[1:].to_numpy().astype('float')
                max_index = D[:, :-1].max()
                min_index = D[:, :-1].min()
                num_vertices = int(max_index - min_index + 1)
                S = np.zeros((num_vertices, num_vertices))

                S[D[:, 0].astype('int'), D[:, 1].astype('int')] = D[:, -1]
                np.fill_diagonal(S, 0)
                np.save(f'data/{dataset_name}/similarity_matrix_{distance_metric}.npy', S)
            else:
                pass


def computeSimilarity(dataset_name, distance_metric, dataset_path=None, problem_type=PROBLEMTYPES.REVENUE_MAXIMIZATION):
    if not os.path.isfile(f'data/{dataset_name}/similarity_matrix_{distance_metric}.npy'):
        obtainDataset(dataset_name, distance_metric, dataset_path, problem_type)

    S = np.load(f'data/{dataset_name}/similarity_matrix_{distance_metric}.npy')
    return S


def defineF(problem_type, mode, p=0.5, W=None, D=None):
    if problem_type == PROBLEMTYPES.REVENUE_MAXIMIZATION:
        if mode == MODE.OFFLINE:
            def term(x,y):
                term1 = np.multiply(W, ((y) + (-1) ** y * np.power(1 - p, x))[:, np.newaxis])
                term2 = np.multiply(np.ones(W.shape), np.power(1 - p, x)[:, np.newaxis].T)
                np.fill_diagonal(term2, 1)
                return np.multiply(term1, term2)

            F = lambda x: np.sum(term(x, 1))
            grad_F = lambda x: np.log(1-p)*np.sum(term(x, 1).T - term(x, 0), axis=1)
    elif problem_type == PROBLEMTYPES.IMAGE_SUMMARIZATION:
        pass
    elif problem_type == PROBLEMTYPES.LOCATION_SUMMARIZATION:
        if mode == MODE.OFFLINE:
            def termProd(x):
                term1 = np.multiply(W, x[np.newaxis, :])
                greater_matrix = (W[:, :, np.newaxis] > W[:, np.newaxis, :]).astype(int)
                term2 = greater_matrix * (1-x)
                term2[term2 == 0] = 1
                term2 = np.prod(term2, axis=2)
                term3 = D.dot(x)
                return 1/x.shape[0] * np.multiply(term1, term2).sum() - term3
            F = lambda x: termProd(x)

            def termGrad(x):
                greater_matrix = (W[:, :, np.newaxis] > W[:, np.newaxis, :]).astype(int)
                indices_non_zero = [np.nonzero(column)[0] for column in greater_matrix.T]
                term2 = greater_matrix * (1-x)
                term2[term2 == 0] = 1
                term2 = np.prod(term2, axis=2)


            grad_F = lambda x: None
    else:
        raise NotImplementedError('Please implement F and its grad, if you dont have grad, then please send None')

    return F, grad_F


def defineKNKD(problem_type, d):
    if problem_type == PROBLEMTYPES.REVENUE_MAXIMIZATION:
        K_N = np.ones(d + 1)
        K_N[-1] = -0.25
        K_D = np.ones(d+1)
        K_D[-1] = -0.75
        K = np.ones((2, d + 1))
        K[0, -1] = -1
        K[1, :-1] *= -1
        K[1, -1] = 0.25

    return K_N, K_D, K
