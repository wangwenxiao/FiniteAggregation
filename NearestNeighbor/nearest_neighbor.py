from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import numpy
import random

from skimage.feature import hog
from sklearn.neighbors import RadiusNeighborsClassifier



class NearestNeighbor(nn.Module):

    __constants__ = ['in_features', 'out_features']
    n_class: int

    def __init__(self, n_class: int,
                 device=None, dtype=None, r = 4, outlier_label = -1) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NearestNeighbor, self).__init__()
        self.n_class = n_class
        self.train_X = []
        self.train_Y = []
        self.N = 0
        self.r = r
        self.outlier_label = outlier_label

    def forward(self, input):
        if len(self.train_Y) == 0:
            return torch.LongTensor([0 if self.outlier_label == -1 else self.outlier_label for i in range(input.size(0))])
        ret = []
        fd = []
        for i in range(input.size(0)):
            fd.append(self.hog(input[i]))
        return torch.LongTensor(self.neigh.predict(fd))

    def extra_repr(self) -> str:
        return 'n_class={}'.format(
            self.n_class,
        )
    
    def hog(self, input):
        return hog(numpy.array(input).transpose((1,2,0)), channel_axis=2, 
                  orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
    
    def update(self, input, labels):
        #global flag
        for i in range(input.size(0)):
            fd = self.hog(input[i])
            #if flag:
            #    flag = False
            #    print (fd.shape)
            self.train_X.append(fd)
            self.train_Y.append(int(labels[i]))
            self.N += 1
            
    def fit(self):
        self.neigh = RadiusNeighborsClassifier(radius = self.r, p = 1, outlier_label = 'most_frequent' if self.outlier_label == -1 else self.outlier_label, weights = 'distance')
        #self.neigh = KNeighborsClassifier(n_neighbors=self.k, p=1, weights = 'distance')
        if len(self.train_Y) > 0:
            self.neigh.fit(self.train_X, self.train_Y)



