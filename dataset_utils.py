import random
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn

from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom
from data_aug_utils import AutoAugment
from data_mimic import Mimic3Dataset

__all__ = ['partition_data', 'get_dataloader']

# def load_mimic3_data(data_args,tokenizer):
#     train_dataset = MimicFullDataset(data_args.version, "train", data_args.max_seq_length, tokenizer, 30, 4) # TODO delete 30 and 8
#     dev_dataset   = MimicFullDataset(data_args.version, "dev", data_args.max_seq_length, tokenizer, 30, 4)
#     eval_dataset  = MimicFullDataset(data_args.version, "test", data_args.max_seq_length, tokenizer, 30, 4)

#     return (train_dataset, dev_dataset, eval_dataset)

# def load_cifar10_data(datadir):
#     transform = transforms.Compose([transforms.ToTensor()])

#     cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
#     cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

#     X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
#     X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

#     return (X_train, y_train, X_test, y_test)


# def load_cifar100_data(datadir):
#     transform = transforms.Compose([transforms.ToTensor()])

#     cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
#     cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

#     X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
#     X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

#     return (X_train, y_train, X_test, y_test)


# def load_tinyimagenet_data(datadir):
#     # transform = transforms.Compose([transforms.ToTensor()])
#     xray_train_ds = ImageFolder_custom(datadir+'/train/', transform=None)
#     xray_test_ds = ImageFolder_custom(datadir+'/val/', transform=None)

#     X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
#     X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

#     return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_parties, n_train):

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        party2dataidx = {i: batch_idxs[i] for i in range(n_parties)}

    return party2dataidx


    