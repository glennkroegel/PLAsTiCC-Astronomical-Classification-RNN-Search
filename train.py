import pandas as pd
import numpy as np
import pickle
from model import StandardModel
from config import CV_SPLIT, BATCH_SIZE, NUM_EPOCHS, MAX_LEN
from utils import *
from dataset import *
from loss import FocalLoss
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

class Learner():
    '''Training loop'''
    def __init__(self, epochs=NUM_EPOCHS):
        self.model = StandardModel()
        self.alpha = None
        self.criterion = FocalLoss(gamma=2, alpha=self.alpha, size_average=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)
        self.epochs = epochs

        self.train_loader = torch.load('train_loader.pt')
        self.cv_loader = torch.load('cv_loader.pt')

        self.best_loss = 1e3

    def train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for i, (seqs, target) in enumerate(train_loader):
            output = model(seqs)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

    def step(self):
        '''Actual training loop.'''
        for epoch in tqdm(range(self.epochs)):
            self.scheduler.step(epoch)
            lr = self.scheduler.get_lr()
            epoch_loss = self.train(self.train_loader, self.model, self.criterion, self.optimizer, epoch)
            # cross validation
            total_val_loss = 0
            total_val_acc = 0
            for seqs, targets in self.cv_loader:
                self.model.eval()
                val_outputs = self.model(seqs)
                val_loss = self.criterion(val_outputs, targets)
                val_acc = accuracy(val_outputs, targets)
                total_val_loss += val_loss.item()
                total_val_acc += val_acc.item()
            epoch_val_loss = total_val_loss/len(self.cv_loader)
            epoch_val_acc = val_acc/len(self.cv_loader)
            if epoch % 1 == 0:
                self.status(epoch, epoch_loss, epoch_val_loss, epoch_val_acc, lr)
            if epoch_val_loss < self.best_loss:
                print('dumping model...')
                path = 'model' + '.pt'
                torch.save(self.model, path)
                self.best_loss = epoch_val_loss

    def status(self, epoch, epoch_loss, epoch_val_loss, epoch_val_acc, lr):
        print('epoch {0}/{1}:\n train_loss: {2} val_loss: {3} val_acc: {4} val_f: {5}'
        .format(epoch, self.epochs, epoch_loss, epoch_val_loss, epoch_val_acc, lr))


if __name__ == "__main__":
    try:
        Learner().step()
    except KeyboardInterrupt:
        pass

