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
import argparse
import numpy
import random
import numpy as np

parser = argparse.ArgumentParser(description='Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations directory')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--k', default = 50, type=int, help='the inverse of sensitivity')
parser.add_argument('--d', default = 1, type=int, help='number of duplicates per sample')




args = parser.parse_args()

args.n_subsets = args.k * args.d

random.seed(999999999+208)
shifts = random.sample(range(args.n_subsets), args.d)


if not os.path.exists('./radii'):
    os.makedirs('./radii')
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

filein = torch.load('./evaluations/'+args.evaluations, map_location=torch.device(device))
labels = filein['labels']
scores = filein['scores']

num_classes = args.num_classes
max_classes = scores.max(2).indices
predictions = torch.zeros(max_classes.shape[0],num_classes)
for i in range(max_classes.shape[1]):
	predictions[(torch.arange(max_classes.shape[0]),max_classes[:,i])] += 1
predinctionsnp = predictions.cpu().numpy()
idxsort = numpy.argsort(-predinctionsnp,axis=1,kind='stable')
valsort = -numpy.sort(-predinctionsnp,axis=1,kind='stable')
val =  valsort[:,0]
idx = idxsort[:,0]
valsecond =  valsort[:,1]
idxsecond =  idxsort[:,1] 

#original code from DPA
#diffs = ((val - valsecond - (idxsecond <= idx))/2).astype(int)
#certs = torch.tensor(diffs).cuda()
#torchidx = torch.tensor(idx).cuda()
#certs[torchidx != labels] = -1


n_sample = labels.size(0)
certs = torch.LongTensor(n_sample)

#prepared for indexing
shifted = [
    [(h + shift)%args.n_subsets for shift in shifts] for h in range(args.n_subsets)
]
shifted = torch.LongTensor(shifted)

cnts = np.zeros(args.k * args.d)

certs = torch.load('./radii/'+args.evaluations)


for i in range(n_sample):
    if certs[i] != 0:
        continue
    
    if i%1000 == 0:
        print (i, '/', n_sample)
    
    certs[i] = args.n_subsets #init value
    label = int(labels[i])

    #max_classes corresponding to diff h
    max_classes_given_h = max_classes[i][shifted.view(-1)].view(-1, args.d)

    flags = np.zeros(args.k * args.d)

    for c in range(num_classes): #compute min radius respect to all classes
        if c != label:
            diff = predictions[i][labels[i]] - predictions[i][c] - (1 if c < label else 0)
            
            deltas = (1 + (max_classes_given_h == label).long() - (max_classes_given_h == c).long()).sum(dim=1)
            for j in range(args.n_subsets):
                if diff < deltas[j]:
                    flags[j] = 1
        
    cnts += flags
            
max_flip = cnts.max()
print ('Maximum Flipped: ' + str(max_flip))


base_acc = 100 *  (max_classes == labels.unsqueeze(1)).sum().item() / (max_classes.shape[0] * max_classes.shape[1])
print('Base classifier accuracy: ' + str(base_acc))




