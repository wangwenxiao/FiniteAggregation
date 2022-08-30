from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import PIL
from gtsrb_dataset import GTSRB
sys.path.append('./FeatureLearningRotNet/architectures')

from NetworkInNetwork import NetworkInNetwork
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy
import random

parser = argparse.ArgumentParser(description='PyTorch GTSRB Training')
parser.add_argument('--k', default = 50, type=int, help='the inverse of sensitivity')
parser.add_argument('--d', default = 1, type=int, help='number of duplicates per sample')

parser.add_argument('--start', required=True, type=int, help='starting subset number')
parser.add_argument('--range', default=250, type=int, help='number of subsets to train')
parser.add_argument('--zero_seed', action='store_true', help='Use a random seed of zero (instead of the partition index)')

args = parser.parse_args()

args.n_subsets = args.k * args.d
args.dataset='gtsrb'



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dirbase = 'gtsrb_nin_baseline'
if (args.zero_seed):
    dirbase += '_zero_seed'

checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_subdir = f'./{checkpoint_dir}/' + dirbase + f'_FiniteAggregation_k{args.k}_d{args.d}'
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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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
    part_indices = torch.tensor(partitions[part])
    transform_train = transforms.Compose([
        # torchvision.transforms.Lambda(lambda x: PIL.ImageOps.equalize(x)), # If using histogram equalization
        torchvision.transforms.Resize((48,48),interpolation=PIL.Image.BILINEAR ),
        transforms.RandomCrop(48, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(means[part], stds[part])
    ])

    transform_test = transforms.Compose([
        # torchvision.transforms.Lambda(lambda x: PIL.ImageOps.equalize(x)), # If using histogram equalization
        torchvision.transforms.Resize((48,48),interpolation=PIL.Image.BILINEAR ),
        transforms.ToTensor(),
        transforms.Normalize(means[part], stds[part])
    ])

    trainset = GTSRB('./data', train=True, transform=transform_train)
    testset = GTSRB('./data', train=False, transform=transform_test)

    nomtestloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=1)
    print('here')
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,part_indices), batch_size=128, shuffle=True, num_workers=1)
    net  = NetworkInNetwork({'num_classes':43})
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=curr_lr, momentum=0.9, weight_decay=0.0005, nesterov= True)

# Training
    net.train()
    for epoch in range(200):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch in [60,120,160]):
            curr_lr = curr_lr * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

    net.eval()

    (inputs, targets)  = next(iter(nomtestloader)) #Just use one test batch
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
            #breakpoint()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
    acc = 100.*correct/total
    print('Accuracy: '+ str(acc)+'%') 
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'partition': part,
        'norm_mean' : means[part],
        'norm_std' : stds[part]
    }
    torch.save(state, checkpoint_subdir + '/FiniteAggregation_'+ str(part)+'.pth')




