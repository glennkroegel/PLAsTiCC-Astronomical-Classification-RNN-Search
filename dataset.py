import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PassbandDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __getitem__(self, i):
        return (self.xs[i], self.ys[i])

    def __len__(self):
        return len(self.xs)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, xs):
        self.xs = xs

    def __getitem__(self, i):
        return self.xs[i]

    def __len__(self):
        return len(self.xs)     

def my_collate(batch):
    data = [item[0:-1] for item in batch]
    target = [item[-1] for item in batch]
    target = torch.LongTensor(target)
    data = np.squeeze(data)
    return [data, target]

def test_collate(batch):
    data = [item for item in batch]
    data = np.squeeze(data)
    return data