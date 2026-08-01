# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

prepare dataloader

@author: Katsuhisa Morita, tadahaya
"""
import gc
import time
from typing import Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image

# dataset
def prep_dataloader(
    dataset, batch_size:int, shuffle:bool=True, num_workers:int=4, pin_memory:bool=True, 
    drop_last:bool=True, sampler=None, collate_fn=None,
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_fn
        )
    return loader

class BalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset, n_frac = None, n_samples = None):
        avg = np.mean(dataset.labels, axis=0)
        avg[avg == 0] = 0.5
        avg[avg == 1] = 0.5
        self.avg = avg
        weights = (1 / (1 - avg + 1e-8)) * (1 - dataset.labels) + (
            1 / (avg + 1e-8)
        ) * dataset.labels
        weights = np.max(weights, axis=1)
        # weights = np.ones_like(dataset.labels[:,0])
        self.weights = weights
        if n_frac:
            super().__init__(weights, int(n_frac * len(dataset)))
        elif n_samples:
            super().__init__(weights, n_samples)

def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def resize_dataset_dir(dataset, size:int=256):
    """ data resize for small scaling """
    dataset.dir_lst = dataset.dir_lst[:size]
    dataset.datanum = size
    return dataset

def resize_dataset(dataset, size:int=256):
    """ data resize for small scaling """
    dataset.data = dataset.data[:size]
    dataset.datanum = size
    return dataset