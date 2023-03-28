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
from data_mimic import MimicFullDataset

__all__ = ['partition_data', 'get_dataloader']

def load_mimic3_data(data_args,tokenizer):
    train_dataset = MimicFullDataset(data_args.version, "train", data_args.max_seq_length, tokenizer, 30, 4) # TODO delete 30 and 8
    dev_dataset   = MimicFullDataset(data_args.version, "dev", data_args.max_seq_length, tokenizer, 30, 4)
    eval_dataset  = MimicFullDataset(data_args.version, "test", data_args.max_seq_length, tokenizer, 30, 4)

    return (train_dataset, dev_dataset, eval_dataset)

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    # transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'/train/', transform=None)
    xray_test_ds = ImageFolder_custom(datadir+'/val/', transform=None)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_parties, alpha=0.4):

    if dataset == 'mimic3':
        train_dataset, dev_dataset, eval_dataset = load_mimic3_data(data_args,tokenizer)
    else:
        raise NotImplementedError("dataset not imeplemented")

    n_train = train_dataset.code_count 

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        party2dataidx = {i: batch_idxs[i] for i in range(n_parties)}

    return party2dataidx


def get_dataloader(args, dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if args.auto_aug:
            transform_train.append(AutoAugment())

        transform_train.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated

        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if args.auto_aug:
            transform_train.append(AutoAugment())

        transform_train.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom

        transform_train = []
        if args.auto_aug:
            transform_train.append(AutoAugment())

        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'/val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    else:
        raise NotImplementedError("dataset not implemented")

    return train_dl, test_dl
