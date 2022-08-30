import torch
import torchvision
import argparse
import numpy as  np
import PIL
from gtsrb_dataset import GTSRB
import random

parser = argparse.ArgumentParser(description='Partition Data')
parser.add_argument('--dataset', default="mnist", type=str, help='dataset to partition')
parser.add_argument('--k', default=1200, type=int, help='the inverse of sensitivity')
parser.add_argument('--d', default=1, type=int, help='number of duplicates per sample')
parser.add_argument('--root', default='./data')

args = parser.parse_args()

args.n_subsets = args.k * args.d



channels =3
if (args.dataset == "mnist"):
	data = torchvision.datasets.MNIST(root=args.root, train=True, download=True)
	channels = 1

if (args.dataset == "cifar"):
	data = torchvision.datasets.CIFAR10(root=args.root, train=True, download=True)

if (args.dataset == "gtsrb"):
	data = GTSRB('./data', train=True)


if (args.dataset != "gtsrb"):
	imgs, labels = zip(*data)
	finalimgs = torch.stack(list(map((lambda x: torchvision.transforms.ToTensor()(x)), list(imgs))))
	for_sorting = (finalimgs*255).int()
	intmagessum = for_sorting.reshape(for_sorting.shape[0],-1).sum(dim=1) % args.n_subsets
	
else:
	labels = [label for x,label in data]
	imgs_scaled = [torchvision.transforms.ToTensor() ( torchvision.transforms.Resize((48,48),interpolation=PIL.Image.BILINEAR )(image)) for image, y in data]
	#imgs_scaled = [torchvision.transforms.ToTensor() ( torchvision.transforms.Resize((48,48),interpolation=PIL.Image.BILINEAR )(PIL.ImageOps.equalize(image))) for image, y in data] # To use histogram equalization
	finalimgs =  torch.stack(list(imgs_scaled))
	intmagessum = torch.stack([(torchvision.transforms.ToTensor()(image).reshape(-1)*255).int().sum()% args.n_subsets for image, y in data])
	for_sorting =finalimgs


#to specify a mapping from [dk] to [dk]^d
random.seed(999999999+208)
shifts = random.sample(range(args.n_subsets), args.d)


idxgroup = [[] for i in range(args.n_subsets)]
for i, h in enumerate(intmagessum):
    for shift in shifts:
        idxgroup[(h + shift)%args.n_subsets].append(i)


idxgroup = [torch.LongTensor(idxs).view(-1, 1) for idxs in idxgroup]

#for i in range(len(idxgroup)):
#    print (idxgroup[i].size())
#exit()

#force index groups into an order that depends only on image content  (not indexes) so that (deterministic) training will not depend initial indices
idxgroup = list([idxgroup[i][np.lexsort(torch.cat((torch.tensor(labels)[idxgroup[i]].int(),for_sorting[idxgroup[i]].reshape(idxgroup[i].shape[0],-1)),dim=1).numpy().transpose())] for i in range(args.n_subsets) ])

idxgroupout = list([x.squeeze().numpy() for x in idxgroup])
means = torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).mean(dim=1) for i in range(args.n_subsets) ]))
stds =  torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).std(dim=1) for i in range(args.n_subsets) ]))
out = {'idx': idxgroupout,'mean':means.numpy(),'std':stds.numpy() }
torch.save(out, "FiniteAggregation_hash_mean_" +args.dataset+'_k'+str(args.k)+'_d' + str(args.d) + '.pth')
