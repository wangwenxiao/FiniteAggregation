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

from skimage.feature import hog
from sklearn.neighbors import RadiusNeighborsClassifier

from NearestNeighbor.nearest_neighbor import NearestNeighbor

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--k', default = 50, type=int, help='the inverse of sensitivity')
parser.add_argument('--d', default = 1, type=int, help='number of duplicates per sample')

parser.add_argument('--start', required=True, type=int, help='starting subset number')
parser.add_argument('--range', default=250, type=int, help='number of subsets to train')
parser.add_argument('--zero_seed', action='store_true', help='Use a random seed of zero (instead of the partition index)')


#args for nearest neighbor
parser.add_argument('--radius', default = 20, type=float, help='radius for radius neighbor classifiers')
parser.add_argument('--outlier_label', default = -1, type=int, help='outlier_label: -1 means most_frequent; otherwise its value is used')




args = parser.parse_args()

args.n_subsets = args.k * args.d
args.dataset='cifar'
args.n_class = 10



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dirbase = 'cifar_nearest_neighbor'
if (args.zero_seed):
    dirbase += '_zero_seed'

checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_subdir = f'./{checkpoint_dir}/' + dirbase + f'_FiniteAggregation_k{args.k}_d{args.d}_radius{args.radius}_outlier_label{args.outlier_label}'
if not os.path.exists(checkpoint_subdir):
    os.makedirs(checkpoint_subdir)
print("==> Checkpoint directory", checkpoint_subdir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


partitions_file = torch.load("FiniteAggregation_hash_mean_" +args.dataset+'_k'+str(args.k)+'_d' + str(args.d) + '.pth')
partitions = partitions_file['idx']
means = partitions_file['mean']
stds = partitions_file['std']



for part in range(args.start, args.start + args.range):
    seed = part
    if (args.zero_seed):
        seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    curr_lr = 0.1
    print('\Partition: %d' % part)
    part_indices = torch.tensor(partitions[part]).view(-1)
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])


    model = NearestNeighbor(args.n_class, r = args.radius, outlier_label = args.outlier_label)

    if len(part_indices) > 0:

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        nomtestloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=1)
    
        trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,part_indices), batch_size=128, shuffle=True, num_workers=1)

        
        # Training
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            model.update(inputs, targets)
            
    model.fit()

    #(inputs, targets)  = next(iter(nomtestloader)) #Just use one test batch
    ##inputs, targets = inputs.to(device), targets.to(device)
    #with torch.no_grad():
    #    predicted = model(inputs)
    #    
    #    correct = predicted.eq(targets).sum().item()
    #    total = targets.size(0)
        
    #acc = 100.*correct/total
    #print('Accuracy: '+ str(acc)+'%') 
    
    # Save checkpoint.
    print('Saving..')
    state = {
        'model': model,
        #'acc': acc,
        'partition': part,
        #'norm_mean' : means[part],
        #'norm_std' : stds[part]
    }
    torch.save(state, checkpoint_subdir + '/FiniteAggregation_nearest_neighbor_'+ str(part)+'.pth')




