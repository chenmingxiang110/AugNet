import os
import cv2
import time
import random
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models

from imgaug import augmenters as iaa
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import linear_sum_assignment as linear_assignment

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, in_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(in_channels, out_channels, kernel, 1, padding=kernel//2)
        self.is_shortcut = in_channels!=out_channels or stride!=1
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = self.shortcut(x) if self.is_shortcut else x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += residual
        x = self.act(x)
        x = self.bn(x)
        return x # (batch, channel, feature, time)

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first):
        super(BidirectionalLSTM, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x, _ = self.BiLSTM(x)
        x = self.act(x)
        return x
    
def nin_block(channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(channels[0], channels[1], kernel_size, strides, padding),
        nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True),
        nn.Conv2d(channels[1], channels[2], kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True),
        nn.Conv2d(channels[2], channels[3], kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(channels[3]), nn.ReLU(inplace=True),)

class Net_3(nn.Module):

    def __init__(self):
        super(Net_3, self).__init__()
        self.seq0 = nn.Sequential(
            nin_block([3,192,160,96], kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.seq1 = nn.Sequential(
            nin_block([96,192,192,192], kernel_size=5, strides=1, padding=2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        
        self.seq2 = nin_block([192,192,192,192], kernel_size=3, strides=1, padding=1)
        
        self.act = nn.Sequential(
#             nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Tanh(),
            nn.Flatten())

    def forward(self, x):
        h1 = self.seq0(x)
        h2 = self.seq1(h1)
        h3 = self.seq2(h2)
        h = self.act(h3)
        return (h1, h2, h3), h
    
class Net_4(nn.Module):

    def __init__(self):
        super(Net_4, self).__init__()
        self.seq0 = nn.Sequential(
            nin_block([3,192,160,96], kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.seq1 = nn.Sequential(
            nin_block([96,192,192,192], kernel_size=5, strides=1, padding=2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        
        self.seq2 = nin_block([192,192,192,192], kernel_size=3, strides=1, padding=1)
        
        self.seq3 = nin_block([192,192,192,192], kernel_size=3, strides=1, padding=1)
        
        self.act = nn.Sequential(
#             nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Tanh(),
            nn.Flatten())

    def forward(self, x):
        h1 = self.seq0(x)
        h2 = self.seq1(h1)
        h3 = self.seq2(h2)
        h4 = self.seq3(h3)
        h = self.act(h4)
        return (h1, h2, h3, h4), h
    
class Net_5(nn.Module):

    def __init__(self):
        super(Net_5, self).__init__()
        self.seq0 = nn.Sequential(
            nin_block([3,192,160,96], kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.seq1 = nn.Sequential(
            nin_block([96,192,192,192], kernel_size=5, strides=1, padding=2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        
        self.seq2 = nin_block([192,192,192,192], kernel_size=3, strides=1, padding=1)
        
        self.seq3 = nin_block([192,192,192,192], kernel_size=3, strides=1, padding=1)
        
        self.seq4 = nin_block([192,192,192,192], kernel_size=3, strides=1, padding=1)
        
        self.act = nn.Sequential(
#             nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Tanh(),
            nn.Flatten())

    def forward(self, x):
        h1 = self.seq0(x)
        h2 = self.seq1(h1)
        h3 = self.seq2(h2)
        h4 = self.seq3(h3)
        h5 = self.seq4(h4)
        h = self.act(h5)
        return (h1, h2, h3, h4, h5), h

class Net34(nn.Module):

    def __init__(self, path_pre=None, gpus=[]):
        super(Net34, self).__init__()
        self.pre = models.resnet34(pretrained=False)
        
        if path_pre is not None:
            self.pre.load_state_dict(torch.load(path_pre))
            
        self.pre.fc = Identity()
        self.act = nn.Tanh()
        
        if len(gpus) > 1:
            self.pre = nn.DataParallel(self.pre, gpus)

    def forward(self, x):
        h = self.pre(x)
        h = self.act(h)
        return h

class Supervised_Classifier2(nn.Module):

    def __init__(self, mode, in_features, n_class=10, k_size=8, isLast=False):
        super(Supervised_Classifier2, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
        )
        self.isMultihead = False
        
        if mode=="linear":
            self.seq2 = nn.Sequential(
                nn.Linear(in_features * k_size * k_size, n_class)
            )
        elif mode=="non-linear":
            self.seq2 = nn.Sequential(
                nn.Linear(in_features * k_size * k_size, 256),
                nn.BatchNorm1d(num_features=256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(num_features=256),
                nn.ReLU(inplace=True),
                nn.Linear(256, n_class),
            )
        else:
            raise ValueError("Not implemented.")
            

    def forward(self, x):
        h = self.seq(x)
        h = self.seq2(h)
        return h
    
def clustering_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    
class Loss(nn.Module):

    def __init__(self, device=torch.device("cpu")):
        super(Loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        
        self.device = device
        self.to(device)

    def forward(self, x, mode, optimizer=None, scheduler=None):
        """
        x: (k/j (classes), m/i (samples), c (n_features))
        """
        centers = torch.mean(x, 1, keepdim=False)
        sims = -torch.cdist(x.reshape([-1, x.size(-1)]), centers, p=2)
        # sims = torch.max(sims, torch.tensor(-5.0).to(self.device))
        
        if mode=="softmax":
            labels = torch.tensor(list(np.arange(x.size(0)).repeat(x.size(1)))).to(self.device)
            loss = self.ce(sims, labels)
        
        elif mode=="contrast":
            indices = list(np.arange(sims.size(0)))
            labels = list(np.arange(x.size(0)).repeat(x.size(1)))
            sims_clone = torch.clone(sims)
            sims_clone[indices, labels] = -1e32
            loss_self = sims[indices, labels]
            loss_others = torch.max(sims_clone, dim=1).values
            # loss = 1-torch.sigmoid(loss_self)+torch.sigmoid(loss_others)
            loss = loss_others-loss_self
            loss = torch.mean(loss)
        
        else:
            raise ValueError("Invalid mode.")
        
        if optimizer is not None:
            # back propagation and update centers
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        return loss