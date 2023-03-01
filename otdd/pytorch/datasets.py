import os
import pdb
from functools import partial
import random
import logging
import string

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.utils.data as torchdata
import torch.utils.data.dataloader as dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as dset

import torchtext
from torchtext.data.utils import get_tokenizer
from copy import deepcopy as dpcp

import h5py

from .. import DATA_DIR

from .utils import interleave, process_device_arg, random_index_split, \
                   spectrally_prescribed_matrix, rot, rot_evecs

from .sqrtm import create_symm_matrix

logger = logging.getLogger(__name__)


DATASET_NCLASSES = {
    'MNIST': 10,
    'FashionMNIST': 10,
    'EMNIST': 26,
    'KMNIST': 10,
    'USPS': 10,
    'CIFAR10': 10,
    'SVHN': 10,
    'STL10': 10,
    'LSUN': 10,
    'tiny-ImageNet': 200
}

DATASET_SIZES = {
    'MNIST': (28,28),
    'FashionMNIST': (28,28),
    'EMNIST': (28,28),
    'QMNIST': (28,28),
    'KMNIST': (28,28),
    'USPS': (16,16),
    'SVHN': (32, 32),
    'CIFAR10': (32, 32),
    'STL10': (96, 96),
    'tiny-ImageNet': (64,64)
}

DATASET_NORMALIZATION = {
    'MNIST': ((0.1307,), (0.3081,)),
    'USPS' : ((0.1307,), (0.3081,)),
    'FashionMNIST' : ((0.1307,), (0.3081,)),
    'QMNIST' : ((0.1307,), (0.3081,)),
    'EMNIST' : ((0.1307,), (0.3081,)),
    'KMNIST' : ((0.1307,), (0.3081,)),
    'ImageNet': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'tiny-ImageNet': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'CIFAR10': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'STL10': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
}


def sort_by_label(X,Y):
    idxs = np.argsort(Y)
    return X[idxs,:], Y[idxs]


### Data Transforms
class DiscreteRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CustomTensorDataset1(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            # print('bf',x.shape)
            x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)
            # print('aft',x.shape)
        return x, y

    def __len__(self):
        return len(self.data)

class SubsetSampler(torch.utils.data.Sampler):
    r"""Samples elements in order (not randomly) from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        (this is identical to torch's SubsetRandomSampler except not random)
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class CustomTensorDataset2(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.targets = tensors[1]
        self.transform = transform
        #self.transform = transforms.Compose([
        #    transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #])
        
        
        #self.transform.append(
        #    torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        #)

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = torch.tensor(self.tensors[1][index]).long()

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]
    
class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None, target_transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class CustomTensorDataset3(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None, target_transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.tensors[index][0]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[index][1]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y
    
    def set_label(self, index, to_label):
        self.tensors[index] = (self.tensors[index][0], to_label)

    def __len__(self):
        return len(self.tensors)
    
class SubsetFromLabels(torch.utils.data.dataset.Dataset):
    """ Subset of a dataset at specified indices.

    Adapted from torch.utils.data.dataset.Subset to allow for label re-mapping
    without having to copy whole dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        targets_map (dict, optional):  Dictionary to map targets with
    """
    def __init__(self, dataset, labels, remap=False):
        self.dataset = dataset
        self.labels  = labels
        self.classes = [dataset.classes[i] for i in labels]
        self.mask    = np.isin(dataset.targets, labels).squeeze()
        self.indices = np.where(self.mask)[0]
        self.remap   = remap
        targets = dataset.targets[self.indices]
        if remap:
            V = sorted(np.unique(targets))
            assert list(V) == list(labels)
            targets = torch.tensor(np.digitize(targets, self.labels, right=True))
            self.tmap = dict(zip(V,range(len(V))))
        self.targets = targets

    def __getitem__(self, idx):
        if self.remap is False:
            return self.dataset[self.indices[idx]]
        else:
            item =  self.dataset[self.indices[idx]]
            return (item[0], self.tmap[item[1]])

    def __len__(self):
        return len(self.indices)

def subdataset_from_labels(dataset, labels, remap=True):
    mask = np.isin(dataset.targets, labels).squeeze()
    idx  = np.where(mask)[0]
    subdataset = Subset(dataset,idx, remap_targets=True)
    return subdataset


def dataset_from_numpy(X, Y, classes = None):
    targets =  torch.LongTensor(list(Y))
    ds = TensorDataset(torch.from_numpy(X).type(torch.FloatTensor),targets)
    ds.targets =  targets
    ds.classes = classes if classes is not None else [i for i in range(len(np.unique(Y)))]
    return ds


gmm_configs = {
    'star': {
            'means': [torch.Tensor([0,0]),
                      torch.Tensor([0,-2]),
                      torch.Tensor([2,0]),
                      torch.Tensor([0,2]),
                      torch.Tensor([-2,0])],
            'covs':  [spectrally_prescribed_matrix([1,1], torch.eye(2)),
                      spectrally_prescribed_matrix([2.5,1], torch.eye(2)),
                      spectrally_prescribed_matrix([1,20], torch.eye(2)),
                      spectrally_prescribed_matrix([10,1], torch.eye(2)),
                      spectrally_prescribed_matrix([1,5], torch.eye(2))
                     ],
            'spread': 6,
    }

}

def make_gmm_dataset(config='random', classes=10,dim=2,samples=10,spread = 1,
                     shift=None, rotate=None, diagonal_cov=False, shuffle=True):
    """ Generate Gaussian Mixture Model datasets.

    Arguments:
        config (str): determines cluster locations, one of 'random' or 'star'
        classes (int): number of classes in dataset
        dim (int): feature dimension of dataset
        samples (int): number of samples in dataset
        spread (int): separation of clusters
        shift (bool): whether to add a shift to dataset
        rotate (bool): whether to rotate dataset
        diagonal_cov(bool): whether to use a diagonal covariance matrix
        shuffle (bool): whether to shuffle example indices

    Returns:
        X (tensor): tensor of size (samples, dim) with features
        Y (tensor): tensor of size (samples, 1) with labels
        distribs (torch.distributions): data-generating distributions of each class

    """
    means, covs, distribs = [], [], []
    _configd = None if config == 'random' else gmm_configs[config]
    spread = spread if (config == 'random' or not 'spread' in _configd) else _configd['spread']
    shift  = shift if (config == 'random' or not 'shift' in _configd) else _configd['shift']

    for i in range(classes):
        if config == 'random':
            mean = torch.randn(dim)
            cov  = create_symm_matrix(1, dim, verbose=False).squeeze()
        elif config == 'star':
            mean = gmm_configs['star']['means'][i]
            cov  = gmm_configs['star']['covs'][i]
        if rotate:
            mean = rot(mean, rotate)
            cov  = rot_evecs(cov, rotate)

        if diagonal_cov:
            cov.masked_fill_(~torch.eye(dim, dtype=bool), 0)

        means.append(spread*mean)
        covs.append(cov)
        distribs.append(MultivariateNormal(means[-1],covs[-1]))

    X = torch.cat([P.sample(sample_shape=torch.Size([samples])) for P in distribs])
    Y = torch.LongTensor([samples*[i] for i in range(classes)]).flatten()

    if shift:
        X += torch.tensor(shift)

    if shuffle:
        idxs = torch.randperm(Y.shape[0])
        X = X[idxs, :]
        Y = Y[idxs]
    return X, Y, distribs



def load_torchvision_data_keepclean(dataname, valid_size=0.1, splits=None, shuffle=True,
                    stratified=False, random_seed=None, batch_size = 64,
                    resize=None, to3channels=False,
                    maxsize = None, maxsize_test=None, num_workers = 0, transform=None,
                    data=None, datadir=None, download=True, filt=False, print_stats = False, shuffle_per=0, keep_clean=None):
    """ Load torchvision datasets.

        We return train and test for plots and post-training experiments
    """
    
    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = 'ImageNet'

        transform_list = []

        if dataname in ['MNIST', 'USPS'] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        transform_list.append(
            torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == 'EMNIST':
            split = 'letters'
            train = DATASET(datadir, split=split, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, split=split, train=False, download=download, transform=valid_transform)
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(['C', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'])
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                'byclass': list(_all_classes),
                'bymerge': sorted(list(_all_classes - _merged_classes)),
                'balanced': sorted(list(_all_classes - _merged_classes)),
                'letters': list(string.ascii_lowercase),
                'digits': list(string.digits),
                'mnist': list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == 'letters':
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == 'STL10':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            train.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == 'SVHN':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == 'LSUN':
            # pdb.set_trace()
            train = DATASET(datadir, classes='train', download=download, transform=train_transform)
        else:
            train = DATASET(datadir, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, train=False, download=download, transform=valid_transform)
            #print("HEHE DATASET")
    else:
        train, test = data

    #print("Train Type: ", type(train), " Train: ", train)
    #print("Teest Type: ", type(test), " Train: ", test)
    

    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets  = torch.LongTensor(test.targets)

    if not hasattr(train, 'classes') or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes  = sorted(torch.unique(train.targets).tolist())

######################## ------------------------- MNIST MNIST MNIST MNIST MN -------------------------- ##########################
######################## ------------------------- NIST MNIST MNIST MNIST MNI -------------------------- ##########################
######################## ------------------------- IST MNIST MNIST MNIST MNIS -------------------------- ##########################
######################## ------------------------- ST MNIST MNIST MNIST MNIST -------------------------- ##########################


###### VALIDATION IS 0 SO NOT WORRY NOW ######
    ### Data splitting
    fold_idxs    = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs['train'] = np.arange(len(train))
        
    elif splits is None and valid_size > 0:
        ## Train/Valid
        train_idx, valid_idx = random_index_split(len(train), 1-valid_size, (maxsize, None)) # No maxsize for validation
        fold_idxs['train'] = train_idx
        fold_idxs['valid'] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ['split_{}'.format(i) for i in range(len(splits))]
            slens  = splits
        slens = np.array(slens)
        if any(slens < 0): # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, 'Can only deal with one split being -1'
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if 'train' in snames:
                slens[snames.index('train')] = len(train) - slens[np.array(snames) != 'train'].sum()

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [np.random.permutation(np.where(train.targets==c)).T for c in np.unique(train.targets)]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum() # Need to make cumulative for np.split
        split_idxs = [np.sort(s) for s in np.split(idxs, slens)[:-1]] # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i,v in enumerate(split_idxs)}


    # fold_idxs['train'] = np.arange(len(train)) start -> stop by step
    for k, idxs in fold_idxs.items():
        #print("K: ", k, " IDXS: ", idxs)
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace = False))
            #print("AFTERWARDS Fold Idxs Len: ", len(fold_idxs[k]))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k,idxs in fold_idxs.items()}
   
#  ░██████╗██╗░░██╗██╗░░░██╗███████╗███████╗██╗░░░░░███████╗
#  ██╔════╝██║░░██║██║░░░██║██╔════╝██╔════╝██║░░░░░██╔════╝
#  ╚█████╗░███████║██║░░░██║█████╗░░█████╗░░██║░░░░░█████╗░░
#  ░╚═══██╗██╔══██║██║░░░██║██╔══╝░░██╔══╝░░██║░░░░░██╔══╝░░
#  ██████╔╝██║░░██║╚██████╔╝██║░░░░░██║░░░░░███████╗███████╗
#  ╚═════╝░╚═╝░░╚═╝░╚═════╝░╚═╝░░░░░╚═╝░░░░░╚══════╝╚══════╝

    #print("SAMPLER: ", sampler_class, "\n")
    #print("TRAIN: ", train, "\n")
    #print("IDXS: ", fold_idxs['train'], "\n")
    
    
    
    if keep_clean != None and keep_clean > 0:
        #print("IN clean")
        total_shuffles = maxsize - keep_clean
        
        shuffle_inds = np.random.choice(sorted(fold_idxs['train']), size=total_shuffles, replace=False)
        
        for index in shuffle_inds:
            cur_label = train.targets[index]
            new_label = np.random.randint(10)
            while new_label == cur_label:
                new_label = np.random.randint(10)
            cur_label = new_label
            # print("TRAINNNN: ", train[index])
            train.targets[index] = cur_label
        
    
    if shuffle_per != 0:
        #print("IN shuffle")
        total_shuffles = int(shuffle_per * len(fold_idxs['train']))
        
        shuffle_inds = np.random.choice(sorted(fold_idxs['train']), size=total_shuffles, replace=False)
        
        for index in shuffle_inds:
            cur_label = train.targets[index]
            new_label = np.random.randint(10)
            while new_label == cur_label:
                new_label = np.random.randint(10)
            cur_label = new_label
            # print("TRAINNNN: ", train[index])
            train.targets[index] = cur_label
        
        
#         shuffle_step = int(1 / shuffle_per)
        
#         for index in zip(fold_idxs['train'][0::shuffle_step]):
#             cur_label = train.targets[index]
#             new_label = np.random.randint(10)
#             while new_label == cur_label:
#                 new_label = np.random.randint(10)
#             cur_label = new_label
#             # print("TRAINNNN: ", train[index])
#             train.targets[index] = cur_label

    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)

    fold_loaders = {k: dataloader.DataLoader(train, sampler=sampler,**dataloader_args)
                    for k,sampler in fold_samplers.items()}

    
    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace = False))
        sampler_test = SubsetSampler(test_idxs) # For test don't want Random
        dataloader_args['sampler'] = sampler_test
        #print("MAX TEST: ", maxsize_test)
    else:
        dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders['test'] = test_loader

    fnames, flens = zip(*[[k,len(v)] for k,v in fold_idxs.items()])
    fnames = '/'.join(list(fnames) + ['test'])
    flens  = '/'.join(map(str, list(flens) + [len(test)]))

    if hasattr(train, 'data'):
        logger.info('Input Dim: {}'.format(train.data.shape[1:]))
    logger.info('Classes: {} (effective: {})'.format(len(train.classes), len(torch.unique(train.targets))))
    #print(f'Fold Sizes: {flens} ({fnames})')

    return fold_loaders, {'train': train, 'test':test}

def load_torchvision_data_shuffle(dataname, valid_size=0.1, splits=None, shuffle=True,
                    stratified=False, random_seed=None, batch_size = 64,
                    resize=None, to3channels=False,
                    maxsize = None, maxsize_test=None, num_workers = 0, transform=None,
                    data=None, datadir=None, download=True, filt=False, print_stats = False, shuffle_per=0):
    """ Load torchvision datasets.

        We return train and test for plots and post-training experiments
    """
    
    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = 'ImageNet'

        transform_list = []

        if dataname in ['MNIST', 'USPS'] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        transform_list.append(
            torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == 'EMNIST':
            split = 'letters'
            train = DATASET(datadir, split=split, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, split=split, train=False, download=download, transform=valid_transform)
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(['C', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'])
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                'byclass': list(_all_classes),
                'bymerge': sorted(list(_all_classes - _merged_classes)),
                'balanced': sorted(list(_all_classes - _merged_classes)),
                'letters': list(string.ascii_lowercase),
                'digits': list(string.digits),
                'mnist': list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == 'letters':
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == 'STL10':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            train.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == 'SVHN':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == 'LSUN':
            # pdb.set_trace()
            train = DATASET(datadir, classes='train', download=download, transform=train_transform)
        else:
            train = DATASET(datadir, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, train=False, download=download, transform=valid_transform)
            #print("HEHE DATASET")
    else:
        train, test = data

#     print("Train Type: ", type(train), " Train: ", train)
#     print("Teest Type: ", type(test), " Train: ", test)
    

    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets  = torch.LongTensor(test.targets)

    if not hasattr(train, 'classes') or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes  = sorted(torch.unique(train.targets).tolist())

######################## ------------------------- MNIST MNIST MNIST MNIST MN -------------------------- ##########################
######################## ------------------------- NIST MNIST MNIST MNIST MNI -------------------------- ##########################
######################## ------------------------- IST MNIST MNIST MNIST MNIS -------------------------- ##########################
######################## ------------------------- ST MNIST MNIST MNIST MNIST -------------------------- ##########################


###### VALIDATION IS 0 SO NOT WORRY NOW ######
    ### Data splitting
    fold_idxs    = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs['train'] = np.arange(len(train))
        
    elif splits is None and valid_size > 0:
        ## Train/Valid
        train_idx, valid_idx = random_index_split(len(train), 1-valid_size, (maxsize, None)) # No maxsize for validation
        fold_idxs['train'] = train_idx
        fold_idxs['valid'] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ['split_{}'.format(i) for i in range(len(splits))]
            slens  = splits
        slens = np.array(slens)
        if any(slens < 0): # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, 'Can only deal with one split being -1'
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if 'train' in snames:
                slens[snames.index('train')] = len(train) - slens[np.array(snames) != 'train'].sum()

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [np.random.permutation(np.where(train.targets==c)).T for c in np.unique(train.targets)]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum() # Need to make cumulative for np.split
        split_idxs = [np.sort(s) for s in np.split(idxs, slens)[:-1]] # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i,v in enumerate(split_idxs)}


    # fold_idxs['train'] = np.arange(len(train)) start -> stop by step
    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace = False))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k,idxs in fold_idxs.items()}
    

#  ░██████╗██╗░░██╗██╗░░░██╗███████╗███████╗██╗░░░░░███████╗
#  ██╔════╝██║░░██║██║░░░██║██╔════╝██╔════╝██║░░░░░██╔════╝
#  ╚█████╗░███████║██║░░░██║█████╗░░█████╗░░██║░░░░░█████╗░░
#  ░╚═══██╗██╔══██║██║░░░██║██╔══╝░░██╔══╝░░██║░░░░░██╔══╝░░
#  ██████╔╝██║░░██║╚██████╔╝██║░░░░░██║░░░░░███████╗███████╗
#  ╚═════╝░╚═╝░░╚═╝░╚═════╝░╚═╝░░░░░╚═╝░░░░░╚══════╝╚══════╝

    
    
    if shuffle_per != 0:
        
        total_shuffles = int(shuffle_per * len(fold_idxs['train']))
        
        shuffle_inds = np.random.choice(sorted(fold_idxs['train']), size=total_shuffles, replace=False)
        
        
        if dataname == 'CIFAR10':
            print("CIFAR TEN")
            for index in shuffle_inds:
                cur_label = train.targets[index]
                new_label = np.random.randint(10)
                while new_label == cur_label:
                    new_label = np.random.randint(10)
                cur_label = new_label
                # print("TRAINNNN: ", train[index])
                train.targets[index] = cur_label
        elif dataname == 'CIFAR100':
            print("CIFAR HUNDRED")
            for index in shuffle_inds:
                cur_label = train.targets[index]
                new_label = np.random.randint(100)
                while new_label == cur_label:
                    new_label = np.random.randint(100)
                cur_label = new_label
                # print("TRAINNNN: ", train[index])
                train.targets[index] = cur_label
        elif dataname == 'MNIST':
            print("MNIST")
            for index in shuffle_inds:
                cur_label = train.targets[index]
                print(f'Currrent label: {cur_label}')
                new_label = np.random.randint(10)
                while new_label == cur_label:
                    new_label = np.random.randint(10)
                cur_label = new_label
                print(f'New label: {cur_label} ')
                # print("TRAINNNN: ", train[index])
                train.targets[index] = cur_label
                print("TRAINNNN label: ", train.targets[index])
                print("TRAINNNN: ", train[index])
        elif dataname == 'FashionMNIST':
            print("FashionistaMNIST")
            for index in shuffle_inds:
                cur_label = train.targets[index]
                new_label = np.random.randint(10)
                while new_label == cur_label:
                    new_label = np.random.randint(10)
                cur_label = new_label
                # print("TRAINNNN: ", train[index])
                train.targets[index] = cur_label
                
        ########## FOR other datasets such as STL10 and ImageNet, we cannot directly modify labels
        ########## so will need to recreate the dataloader! time consuming!
        
        elif dataname == 'STL10' or dataname == 'ImageNet':
            print("STL11")
            if dataname == 'ImageNet':
                print('IMAGI')
                DATASET = getattr(torchvision.datasets, dataname)
            new_train = DATASET
            new_train.targets = {}
            new_train.classes = {}
            new_train.targets = train.targets
            new_train.classes = train.classes
            
#             train_cp = CustomTensorDataset3(train)
            
# #             train_cp = torch.tensor(len(train),2)
# #             train_cp = train_cp[i] for 
            
            
#             import pdb;pdb.set_trace()
            
            
            
            new_ds_imgs = []
            new_ds_labs = []
            class_len = len(train.classes)
            for i in range(len(train)):
                new_ds_imgs.append(train[i][0].permute(1,2,0))
                if i in shuffle_inds:
                    cur_label = train.targets[i]
                    new_label = np.random.randint(class_len)
#                     print(f'{i}.Currrent label: {cur_label} ')
                    while new_label == cur_label:
                        new_label = np.random.randint(class_len)
                    cur_label = new_label
                    train.targets[i] = cur_label
#                     print(f'{i}.New label: {cur_label} ')
                    new_ds_labs.append(torch.tensor(cur_label).reshape(1))
                else:
                    new_ds_labs.append(torch.tensor(train[i][1]).reshape(1))
            new_ds_imgs = torch.stack(new_ds_imgs, dim=0)
            new_ds_labs = torch.cat(new_ds_labs)
            new_ds_imgs = new_ds_imgs.numpy()
            new_ds_labs = new_ds_labs.numpy()
            
            new_ds = (new_ds_imgs, new_ds_labs)
            
            
            new_train.targets = train.targets
            new_transform_list = []
            new_transform_list.append(torchvision.transforms.ToTensor())
            new_transform = transforms.Compose(new_transform_list)
            new_train = CustomTensorDataset2(new_ds, transform = new_transform)
            train = new_train
            
            
            if type(train.targets) is np.ndarray:
                train.targets = train.targets.tolist()

            if type(train.targets) is list:
                train.targets = torch.LongTensor(train.targets)
            
            if not hasattr(train, 'classes') or not train.classes:
#                 train.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
                train.classes = sorted(torch.unique(train.targets).tolist())
            
#             for index in shuffle_inds:
#                 cur_label = train.targets[index]
#                 stl_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
#                 print(f'Currrent label: {cur_label} : {stl_classes[cur_label]}')
#                 new_label = np.random.randint(10)
#                 while new_label == cur_label:
#                     new_label = np.random.randint(10)
#                 cur_label = new_label
#                 print(f'New label: {cur_label} : {stl_classes[cur_label]}')
#                 # print("TRAINNNN: ", train[index])
#                 train.targets[index] = cur_label
#                 print("TRAINNNN label: ", train.targets[index])
#                 print("TRAINNNN: ", train)
                
#         elif dataname == 'ImageNet':
#             print("IMIJII")
#             for index in shuffle_inds:
#                 cur_label = train.targets[index]
#                 new_label = np.random.randint(100)
#                 while new_label == cur_label:
#                     new_label = np.random.randint(100)
#                 cur_label = new_label
#                 # print("TRAINNNN: ", train[index])
#                 train.targets[index] = cur_label
            
        
#         shuffle_step = int(1 / shuffle_per)
        
#         for index in zip(fold_idxs['train'][0::shuffle_step]):
#             cur_label = train.targets[index]
#             new_label = np.random.randint(10)
#             while new_label == cur_label:
#                 new_label = np.random.randint(10)
#             cur_label = new_label
#             # print("TRAINNNN: ", train[index])
#             train.targets[index] = cur_label

    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)

    fold_loaders = {k: dataloader.DataLoader(train, sampler=sampler,**dataloader_args)
                    for k,sampler in fold_samplers.items()}

    
    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace = False))
        sampler_test = SubsetSampler(test_idxs) # For test don't want Random
        dataloader_args['sampler'] = sampler_test
    else:
        dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders['test'] = test_loader

    fnames, flens = zip(*[[k,len(v)] for k,v in fold_idxs.items()])
    fnames = '/'.join(list(fnames) + ['test'])
    flens  = '/'.join(map(str, list(flens) + [len(test)]))

    if hasattr(train, 'data'):
        logger.info('Input Dim: {}'.format(train.data.shape[1:]))
    logger.info('Classes: {} (effective: {})'.format(len(train.classes), len(torch.unique(train.targets))))

    if shuffle_per != 0:
            return fold_loaders, {'train': train, 'test':test}, shuffle_inds
    return fold_loaders, {'train': train, 'test':test}


def load_imagenet_shuffle(datadir=None, resize=None, tiny=False, augmentations=False, im_maxsize=None, im_maxsize_test=None, valid_size = 0, random_seed=None, stratified=True, shuffle=False, shuffle_per=0,  **kwargs):
    """ Load ImageNet dataset """
    if datadir is None and (not tiny):
        datadir = os.path.join(DATA_DIR,'imagenet100')
    elif datadir is None and tiny:
        datadir = os.path.join(DATA_DIR,'tiny-imagenet-200')

    traindir = os.path.join(datadir, "train")
    validdir = os.path.join(datadir, "val")
    
    if augmentations:
        train_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
        ]
    else:
        train_transform_list = [
            transforms.Resize(224), # revert back to 256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
        ]

    valid_transform_list = [
        transforms.Resize(224),# revert back to 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
    ]

    if resize is not None:
        train_transform_list.insert(3, transforms.Resize(
            (resize, resize)))
        valid_transform_list.insert(2, transforms.Resize(
            (resize, resize)))

    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose(
            train_transform_list
        ),
    )

    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose(
            valid_transform_list
        ),
    )
#     fold_loaders, dsets = load_torchvision_data('tiny-ImageNet', transform=[],
#                                                 data=(train_data, valid_data), datadir=datadir, 
#                                                 maxsize=im_maxsize, maxsize_test=im_maxsize_test, random_seed=random_seed,
#                                                 valid_size= valid_size, stratified=stratified, shuffle=shuffle, 
#                                                 **kwargs)
    
    
    
    if shuffle_per == 0:
        fold_loaders, dsets = load_torchvision_data('ImageNet', datadir=datadir,transform=[],
                                                    data=(train_data, valid_data),  
                                                    maxsize=im_maxsize, maxsize_test=im_maxsize_test, random_seed=random_seed,
                                                    valid_size= valid_size, stratified=stratified, shuffle=shuffle, 
                                                    **kwargs)
    
        return fold_loaders, dsets
    else:
        fold_loaders, dsets, shuffle_inds = load_torchvision_data_shuffle('ImageNet', datadir=datadir,transform=[],
                                                    data=(train_data, valid_data),  
                                                    maxsize=im_maxsize, maxsize_test=im_maxsize_test, random_seed=random_seed,
                                                    valid_size= valid_size, stratified=stratified, shuffle=shuffle, 
                                                    shuffle_per=shuffle_per,
                                                    **kwargs)
    
        return fold_loaders, dsets, shuffle_inds
        





############################░░░██╗░██╗░███████╗░██████╗░░██████╗███╗░░░███╗░░░██╗░██╗░#################################################################################
############################██████████╗██╔════╝██╔════╝░██╔════╝████╗░████║██████████╗#################################################################################
############################╚═██╔═██╔═╝█████╗░░██║░░██╗░╚█████╗░██╔████╔██║╚═██╔═██╔═╝#################################################################################
############################██████████╗██╔══╝░░██║░░╚██╗░╚═══██╗██║╚██╔╝██║██████████╗#################################################################################
############################╚██╔═██╔══╝██║░░░░░╚██████╔╝██████╔╝██║░╚═╝░██║╚██╔═██╔══╝#################################################################################
############################░╚═╝░╚═╝░░░╚═╝░░░░░░╚═════╝░╚═════╝░╚═╝░░░░░╚═╝░╚═╝░╚═╝░░░#################################################################################
#################################################################### DOWN #############################################################################################



#x = torch.zeros(5, 10, 20, dtype=torch.float64)
#x = x + (0.1**0.5)*torch.randn(5, 10, 20)


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def load_torchvision_data_perturb(dataname, valid_size=0.1, splits=None, shuffle=True,
                    stratified=False, random_seed=None, batch_size = 64,
                    resize=None, to3channels=False,
                    maxsize = None, maxsize_test=None, num_workers = 0, transform=None,
                    data=None, datadir=None, download=True, filt=False, print_stats = False, shuffle_per=0, perturb_per=0,
                                 attack=None, target_lab=None):
    """ Load torchvision datasets.

        We return train and test for plots and post-training experiments
    """
   
    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        print("seeed")
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = 'ImageNet'

        transform_list = []

        if dataname in ['MNIST', 'USPS'] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        #transform_list.append(
        #    torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        #)

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == 'EMNIST':
            split = 'letters'
            train = DATASET(datadir, split=split, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, split=split, train=False, download=download, transform=valid_transform)
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(['C', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'])
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                'byclass': list(_all_classes),
                'bymerge': sorted(list(_all_classes - _merged_classes)),
                'balanced': sorted(list(_all_classes - _merged_classes)),
                'letters': list(string.ascii_lowercase),
                'digits': list(string.digits),
                'mnist': list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == 'letters':
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == 'STL10':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            train.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == 'SVHN':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == 'LSUN':
            pdb.set_trace()
            train = DATASET(datadir, classes='train', download=download, transform=train_transform)
        else:
            train = DATASET(datadir, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, train=False, download=download, transform=valid_transform)
            #print("HEHE DATASET")
    else:
        train, test = data

   
#     print("Train Type: ", type(train), " \nTrain: ", train)
#     print("Train Type 2: ", type(train[2]), " \nTrain: ", train[2])
#     print("Teest Type: ", type(test), " \nTest: ", test)
#     print("Train len: ", train[2][0].shape)
#     print("Old Train len1: ", train[2][0])
    train_lab = train[2][1]
    train_feat = 1
    new_d = (train_feat,train_lab)
    #train.remove(2)
    #train.insert(2, new_d)
   
    new_train = DATASET
    new_train.targets = {}
    new_train.classes = {}
    new_train.targets = train.targets
    new_train.classes = train.classes
   
    new_ds_imgs = []
    new_ds_labs = []
#     print("Train shape: ", len(train[0]))
    k =0
    perturb_inds = []
    print("perturb_per .", perturb_per)
    if perturb_per > 0:
        print("Went inside.")
        for i in range(len(train)):
            rand_num = np.random.rand()
            if rand_num < perturb_per:
                perturb_inds.append(i)
                #print("IN!!! ", perturb_inds, " i: ", i)
                k += 1
                if dataname == "MNIST":
                    new_ds_imgs.append(train[i][0] + (0.1**0.7)*torch.randn(1, 28, 28))
                elif dataname == "CIFAR10":
                    new_ds_imgs.append((train[i][0] + (0.1**0.7)*torch.randn(3, 32, 32)).permute(1,2,0))
                new_ds_labs.append(torch.tensor(train[i][1]).reshape(1))
            else:
                if dataname == "MNIST":
                    new_ds_imgs.append(train[i][0])
                elif dataname == "CIFAR10":
                    new_ds_imgs.append(train[i][0].permute(1,2,0))
                new_ds_labs.append(torch.tensor(train[i][1]).reshape(1))
        print("Went through : ", str(k), " images out of ", str(i) ," images (", str(k/i), "%)")
        #perturb_inds = np.array(sorted(perturb_inds))
        #new_ds = torch.from_numpy(new_ds).long()

#         print("Old Train len: ", train[2][0])

        #new_ds_imgs = [t.numpy() for t in new_ds_imgs]
        #new_ds_imgs = np.asarray(new_ds_imgs)
        if dataname == "MNIST":
            print("bef: ", len(new_ds_imgs))
            print("bef: ", new_ds_imgs[0].shape)
            print("bef: ", torch.cat(new_ds_imgs).shape)
                  
            new_ds_imgs = torch.cat(new_ds_imgs).unsqueeze(dim=3)
            print("aft: ", new_ds_imgs.shape)
        elif dataname == "CIFAR10":
            print("In cifar")
            new_ds_imgs = torch.stack(new_ds_imgs, dim=0)
            print("type: ", type(new_ds_imgs))
            print("shape: ", new_ds_imgs.shape)
        new_ds_labs = torch.cat(new_ds_labs)
        new_ds_imgs = new_ds_imgs.numpy()
        new_ds_labs = new_ds_labs.numpy()
#         print("new_ds_imgs: ", type(new_ds_imgs), " shape: ", new_ds_imgs.shape)
#         print("new_ds_labs: ", type(new_ds_labs), " shape: ", new_ds_labs.shape)
#         print("new_ds_imgs len: ", len(new_ds_imgs))


        new_ds = (new_ds_imgs, new_ds_labs)

        #new_ds=np.vstack(new_ds).astype(np.float)
        # new_ds = dtype=np.float32
        #new_train = CustomTensorDataset1(new_ds, train.targets, transform = train_transform)
        new_train = CustomTensorDataset2(new_ds, transform = train_transform)

#         print("new train: ", len(new_train))
#         print("New Train: ", new_train)
#         print("New Train len1: ", new_train[2][0])
        train = new_train

#         print("New Train len2: ", train[2][1])
        #train = new_train

#         print("train.targets: ", train.targets)
   
   
    if type(train.targets) is np.ndarray:
        train.targets = train.targets.tolist()
   
    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets  = torch.LongTensor(test.targets)

    if not hasattr(train, 'classes') or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes  = sorted(torch.unique(test.targets).tolist())

######################## ------------------------- MNIST MNIST MNIST MNIST MN -------------------------- ##########################
######################## ------------------------- NIST MNIST MNIST MNIST MNI -------------------------- ##########################
######################## ------------------------- IST MNIST MNIST MNIST MNIS -------------------------- ##########################
######################## ------------------------- ST MNIST MNIST MNIST MNIST -------------------------- ##########################


###### VALIDATION IS 0 SO NOT WORRY NOW ######
    ### Data splitting
    fold_idxs    = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs['train'] = np.arange(len(train))
       
#         print("FOLD IDSSS: ", fold_idxs['train'])
    elif splits is None and valid_size > 0:
        ## Train/Valid
        train_idx, valid_idx = random_index_split(len(train), 1-valid_size, (maxsize, None)) # No maxsize for validation
        fold_idxs['train'] = train_idx
        fold_idxs['valid'] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ['split_{}'.format(i) for i in range(len(splits))]
            slens  = splits
        slens = np.array(slens)
        if any(slens < 0): # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, 'Can only deal with one split being -1'
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if 'train' in snames:
                slens[snames.index('train')] = len(train) - slens[np.array(snames) != 'train'].sum()

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [np.random.permutation(np.where(train.targets==c)).T for c in np.unique(train.targets)]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum() # Need to make cumulative for np.split
        split_idxs = [np.sort(s) for s in np.split(idxs, slens)[:-1]] # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i,v in enumerate(split_idxs)}


    # fold_idxs['train'] = np.arange(len(train)) start -> stop by step
    for k, idxs in fold_idxs.items():
#         print("K: ", k, " IDXS: ", idxs)
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace = False))
#             print("AFTERWARDS Fold Idxs Len: ", len(fold_idxs[k]))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k,idxs in fold_idxs.items()}
   
#  ░██████╗██╗░░██╗██╗░░░██╗███████╗███████╗██╗░░░░░███████╗
#  ██╔════╝██║░░██║██║░░░██║██╔════╝██╔════╝██║░░░░░██╔════╝
#  ╚█████╗░███████║██║░░░██║█████╗░░█████╗░░██║░░░░░█████╗░░
#  ░╚═══██╗██╔══██║██║░░░██║██╔══╝░░██╔══╝░░██║░░░░░██╔══╝░░
#  ██████╔╝██║░░██║╚██████╔╝██║░░░░░██║░░░░░███████╗███████╗
#  ╚═════╝░╚═╝░░╚═╝░╚═════╝░╚═╝░░░░░╚═╝░░░░░╚══════╝╚══════╝

#     print("SAMPLER: ", sampler_class, "\n")
#     print("TRAIN: ", train, "\n")
#     print("IDXS: ", fold_idxs['train'], "\n")
   
   
    if shuffle_per != 0:
       
        total_shuffles = int(shuffle_per * len(fold_idxs['train']))
       
        shuffle_inds = np.random.choice(sorted(fold_idxs['train']), size=total_shuffles, replace=False)
       
        for index in shuffle_inds:
            cur_label = train.targets[index]
            new_label = np.random.randint(10)
            while new_label == cur_label:
                new_label = np.random.randint(10)
            cur_label = new_label
            # print("TRAINNNN: ", train[index])
            train.targets[index] = cur_label
       
       
#         shuffle_step = int(1 / shuffle_per)
       
#         for index in zip(fold_idxs['train'][0::shuffle_step]):
#             cur_label = train.targets[index]
#             new_label = np.random.randint(10)
#             while new_label == cur_label:
#                 new_label = np.random.randint(10)
#             cur_label = new_label
#             # print("TRAINNNN: ", train[index])
#             train.targets[index] = cur_label

    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)

    fold_loaders = {k: dataloader.DataLoader(train, sampler=sampler,**dataloader_args)
                    for k,sampler in fold_samplers.items()}

   
    if maxsize_test and maxsize_test < len(test):
        test_idxs = np.sort(np.random.choice(len(test), maxsize_test, replace = False))
        sampler_test = SubsetSampler(test_idxs) # For test don't want Random
        dataloader_args['sampler'] = sampler_test
#         print("MAX TEST: ", maxsize_test)
    else:
        dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders['test'] = test_loader

    fnames, flens = zip(*[[k,len(v)] for k,v in fold_idxs.items()])
    fnames = '/'.join(list(fnames) + ['test'])
    flens  = '/'.join(map(str, list(flens) + [len(test)]))

    if hasattr(train, 'data'):
        logger.info('Input Dim: {}'.format(train.data.shape[1:]))
    logger.info('Classes: {} (effective: {})'.format(len(train.classes), len(torch.unique(train.targets))))
    print(f'Fold Sizes: {flens} ({fnames})')

   
    return fold_loaders, {'train': train, 'test':test}, perturb_inds


###################################################################### UP ########################################################################################
##################################################FEATURE FEATURE FEATURE FEATURE FEATURE ########################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################CLEAN CLEAN CLEAN CLEAN CLEAN CLEAN CLEAN ######################################################################
#################################################################### DOWN ########################################################################################


def load_torchvision_data(dataname, valid_size=0.1, splits=None, shuffle=True,
                    stratified=False, random_seed=None, batch_size = 64,
                    resize=None, to3channels=False,
                    maxsize = None, maxsize_test=None, num_workers = 0, transform=None,
                    data=None, datadir=None, download=True, filt=False, print_stats = False):
    """ Load torchvision datasets.
        We return train and test for plots and post-training experiments
    """
    if shuffle == True and random_seed:
        np.random.seed(random_seed)
    elif random_seed:
        print("seeed")
        np.random.seed(random_seed)
    if transform is None:
        if dataname in DATASET_NORMALIZATION.keys():
            transform_dataname = dataname
        else:
            transform_dataname = 'ImageNet'

        transform_list = []

        if dataname in ['MNIST', 'USPS'] and to3channels:
            transform_list.append(torchvision.transforms.Grayscale(3))

        transform_list.append(torchvision.transforms.ToTensor())
        
        transform_list.append(
            torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
        )

        if resize:
            if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
                ## Avoid adding an "identity" resizing
                transform_list.insert(0, transforms.Resize((resize, resize)))

        transform = transforms.Compose(transform_list)
        logger.info(transform)
        train_transform, valid_transform = transform, transform
    elif data is None:
        if len(transform) == 1:
            train_transform, valid_transform = transform, transform
        elif len(transform) == 2:
            train_transform, valid_transform = transform
        else:
            raise ValueError()

    if data is None:
        DATASET = getattr(torchvision.datasets, dataname)
        if datadir is None:
            datadir = DATA_DIR
        if dataname == 'EMNIST':
            split = 'letters'
            train = DATASET(datadir, split=split, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, split=split, train=False, download=download, transform=valid_transform)
            ## EMNIST seems to have a bug - classes are wrong
            _merged_classes = set(['C', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'])
            _all_classes = set(list(string.digits + string.ascii_letters))
            classes_split_dict = {
                'byclass': list(_all_classes),
                'bymerge': sorted(list(_all_classes - _merged_classes)),
                'balanced': sorted(list(_all_classes - _merged_classes)),
                'letters': list(string.ascii_lowercase),
                'digits': list(string.digits),
                'mnist': list(string.digits),
            }
            train.classes = classes_split_dict[split]
            if split == 'letters':
                ## The letters fold (and only that fold!!!) is 1-indexed
                train.targets -= 1
                test.targets -= 1
        elif dataname == 'STL10':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            train.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            test.classes = train.classes
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(test.labels)
        elif dataname == 'SVHN':
            train = DATASET(datadir, split='train', download=download, transform=train_transform)
            test = DATASET(datadir, split='test', download=download, transform=valid_transform)
            ## In torchvision, SVHN 0s have label 0, not 10
            train.classes = test.classes = [str(i) for i in range(10)]
            train.targets = torch.tensor(train.labels)
            test.targets = torch.tensor(train.labels)
        elif dataname == 'LSUN':
            # pdb.set_trace()
            train = DATASET(datadir, classes='train', download=download, transform=train_transform)
        else:
            train = DATASET(datadir, train=True, download=download, transform=train_transform)
            test = DATASET(datadir, train=False, download=download, transform=valid_transform)
    else:
        train, test = data


    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets  = torch.LongTensor(test.targets)

    if not hasattr(train, 'classes') or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes  = sorted(torch.unique(train.targets).tolist())


    ### Data splitting
    fold_idxs    = {}
    if splits is None and valid_size == 0:
        ## Only train
        fold_idxs['train'] = np.arange(len(train))
    elif splits is None and valid_size > 0:
        ## Train/Valid
        train_idx, valid_idx = random_index_split(len(train), 1-valid_size, (maxsize, None)) # No maxsize for validation
        fold_idxs['train'] = train_idx
        fold_idxs['valid'] = valid_idx
    elif splits is not None:
        ## Custom splits - must be integer.
        if type(splits) is dict:
            snames, slens = zip(*splits.items())
        elif type(splits) in [list, np.ndarray]:
            snames = ['split_{}'.format(i) for i in range(len(splits))]
            slens  = splits
        slens = np.array(slens)
        if any(slens < 0): # Split expressed as -1, i.e., 'leftover'
            assert sum(slens < 0) == 1, 'Can only deal with one split being -1'
            idx_neg = np.where(slens == -1)[0][0]
            slens[idx_neg] = len(train) - np.array([x for x in slens if x > 0]).sum()
        elif slens.sum() > len(train):
            logging.warning("Not enough samples to satify splits..cropping train...")
            if 'train' in snames:
                slens[snames.index('train')] = len(train) - slens[np.array(snames) != 'train'].sum()

        idxs = np.arange(len(train))
        if not stratified:
            np.random.shuffle(idxs)
        else:
            ## If stratified, we'll interleave the per-class shuffled indices
            idxs_class = [np.random.permutation(np.where(train.targets==c)).T for c in np.unique(train.targets)]
            idxs = interleave(*idxs_class).squeeze().astype(int)

        slens = np.array(slens).cumsum() # Need to make cumulative for np.split
        split_idxs = [np.sort(s) for s in np.split(idxs, slens)[:-1]] # The last one are leftovers
        assert len(split_idxs) == len(splits)
        fold_idxs = {snames[i]: v for i,v in enumerate(split_idxs)}


    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace = False))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k,idxs in fold_idxs.items()}


    ### Create DataLoaders
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)

    fold_loaders = {k:dataloader.DataLoader(train, sampler=sampler,**dataloader_args)
                    for k,sampler in fold_samplers.items()}

    if maxsize_test and maxsize_test <= len(test):
        print("INNNN TEST")
        test_idxs = np.random.choice(len(test), maxsize_test, replace = False)
        sampler_test = SubsetSampler(test_idxs) # For test don't want Random
        dataloader_args['sampler'] = sampler_test
    else:
        dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    fold_loaders['test'] = test_loader

    fnames, flens = zip(*[[k,len(v)] for k,v in fold_idxs.items()])
    fnames = '/'.join(list(fnames) + ['test'])
    flens  = '/'.join(map(str, list(flens) + [len(test)]))

    if hasattr(train, 'data'):
        logger.info('Input Dim: {}'.format(train.data.shape[1:]))
    logger.info('Classes: {} (effective: {})'.format(len(train.classes), len(torch.unique(train.targets))))
    print(f'Fold Sizes: {flens} ({fnames})')
    
    
    
    return fold_loaders, {'train': train, 'test':test}






def load_imagenet(datadir=None, resize=None, tiny=False, augmentations=False, im_maxsize=None, im_maxsize_test=None, valid_size = 0, random_seed=None, stratified=True, shuffle=False, **kwargs):
    """ Load ImageNet dataset """
    if datadir is None and (not tiny):
        datadir = os.path.join(DATA_DIR,'imagenet100')
    elif datadir is None and tiny:
        datadir = os.path.join(DATA_DIR,'tiny-imagenet-200')

    print("OS: ", os.getcwd(), "datadir: ", datadir)
    traindir = os.path.join(datadir, "train")
    validdir = os.path.join(datadir, "val")
    
    if augmentations:
        train_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
        ]
    else:
        train_transform_list = [
            transforms.Resize(224), # revert back to 256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
        ]

    valid_transform_list = [
        transforms.Resize(224),# revert back to 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
    ]

    if resize is not None:
        train_transform_list.insert(3, transforms.Resize(
            (resize, resize)))
        valid_transform_list.insert(2, transforms.Resize(
            (resize, resize)))

    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose(
            train_transform_list
        ),
    )

    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose(
            valid_transform_list
        ),
    )
#     fold_loaders, dsets = load_torchvision_data('tiny-ImageNet', transform=[],
#                                                 data=(train_data, valid_data), datadir=datadir, 
#                                                 maxsize=im_maxsize, maxsize_test=im_maxsize_test, random_seed=random_seed,
#                                                 valid_size= valid_size, stratified=stratified, shuffle=shuffle, 
#                                                 **kwargs)
    fold_loaders, dsets = load_torchvision_data('ImageNet', datadir=datadir,transform=[],
                                                data=(train_data, valid_data),  
                                                maxsize=im_maxsize, maxsize_test=im_maxsize_test, random_seed=random_seed,
                                                valid_size= valid_size, stratified=stratified, shuffle=shuffle, 
                                                **kwargs)

    return fold_loaders, dsets



TEXTDATA_PATHS = {
    'AG_NEWS': 'ag_news_csv',
    'SogouNews': 'sogou_news_csv',
    'DBpedia': 'dbpedia_csv',
    'YelpReviewPolarity': 'yelp_review_polarity_csv',
    'YelpReviewFull': 'yelp_review_full_csv',
    'YahooAnswers': 'yahoo_answers_csv',
    'AmazonReviewPolarity': 'amazon_review_polarity_csv',
    'AmazonReviewFull': 'amazon_review_full_csv',
}

def load_textclassification_data(dataname, vecname='glove.42B.300d', shuffle=True,
            random_seed=None, num_workers = 0, preembed_sentences=True,
            loading_method='sentence_transformers', device='cpu',
            embedding_model=None,
            batch_size = 16, valid_size=0.1, maxsize=None, print_stats = False):
    """ Load torchtext datasets.

    Note: torchtext's TextClassification datasets are a bit different from the others:
        - they don't have split method.
        - no obvious creation of (nor access to) fields

    """



    def batch_processor_tt(batch, TEXT=None, sentemb=None, return_lengths=True, device=None):
        """ For torchtext data/models """
        labels, texts = zip(*batch)
        lens = [len(t) for t in texts]
        labels = torch.Tensor(labels)
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        texttensor = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_idx)
        if sentemb:
            texttensor = sentemb(texttensor)
        if return_lengths:
            return texttensor, labels, lens
        else:
            return texttensor, labels

    def batch_processor_st(batch, model, device=None):
        """ For sentence_transformers data/models """
        device = process_device_arg(device)
        with torch.no_grad():
            batch = model.smart_batching_collate(batch)
            ## Always run embedding model on gpu if available
            features, labels = st.util.batch_to_device(batch, device)
            emb = model(features[0])['sentence_embedding']
        return emb, labels


    if shuffle == True and random_seed:
        np.random.seed(random_seed)

    debug = False

    dataroot = '/tmp/' if debug else DATA_DIR #os.path.join(ROOT_DIR, 'data')
    veccache = os.path.join(dataroot,'.vector_cache')

    if loading_method == 'torchtext':
        ## TextClassification object datasets already do word to token mapping inside.
        DATASET = getattr(torchtext.datasets, dataname)
        train, test = DATASET(root=dataroot, ngrams=1)

        ## load_vectors reindexes embeddings so that they match the vocab's itos indices.
        train._vocab.load_vectors(vecname,cache=veccache,max_vectors = 50000)
        test._vocab.load_vectors(vecname,cache=veccache, max_vectors = 50000)

        ## Define Fields for Text and Labels
        text_field = torchtext.data.Field(sequential=True, lower=True,
                           tokenize=get_tokenizer("basic_english"),
                           batch_first=True,
                           include_lengths=True,
                           use_vocab=True)

        text_field.vocab = train._vocab

        if preembed_sentences:
            ## This will be used for distance computation
            vsize = len(text_field.vocab)
            edim  = text_field.vocab.vectors.shape[1]
            pidx  = text_field.vocab.stoi[text_field.pad_token]
            sentembedder = BoWSentenceEmbedding(vsize, edim, text_field.vocab.vectors, pidx)
            batch_processor = partial(batch_processor_tt,TEXT=text_field,sentemb=sentembedder,return_lengths=False)
        else:
            batch_processor = partial(batch_processor_tt,TEXT=text_field,return_lengths=True)
    elif loading_method == 'sentence_transformers':
        import sentence_transformers as st
        dpath  = os.path.join(dataroot,TEXTDATA_PATHS[dataname])
        reader = st.readers.LabelSentenceReader(dpath)
        if embedding_model is None:
            model  = st.SentenceTransformer('distilbert-base-nli-stsb-mean-tokens').eval()
        elif type(embedding_model) is str:
            model  = st.SentenceTransformer(embedding_model).eval()
        elif isinstance(embedding_model, st.SentenceTransformer):
            model = embedding_model.eval()
        else:
            raise ValueError('embedding model has wrong type')
        print('Reading and embedding {} train data...'.format(dataname))
        train  = st.SentencesDataset(reader.get_examples('train.tsv'), model=model)
        train.targets = train.labels
        print('Reading and embedding {} test data...'.format(dataname))
        test   = st.SentencesDataset(reader.get_examples('test.tsv'), model=model)
        test.targets = test.labels
        if preembed_sentences:
            batch_processor = partial(batch_processor_st, model=model, device=device)
        else:
            batch_processor = None

    ## Seems like torchtext alredy maps class ids to 0...n-1. Adapt class names to account for this.
    classes = torchtext.datasets.text_classification.LABELS[dataname]
    classes = [classes[k+1] for k in range(len(classes))]
    train.classes = classes
    test.classes  = classes

    train_idx, valid_idx = random_index_split(len(train), 1-valid_size, (maxsize, None)) # No maxsize for validation
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers,collate_fn=batch_processor)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler,**dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler,**dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader  = dataloader.DataLoader(test, **dataloader_args)

    if print_stats:
        print('Classes: {} (effective: {})'.format(len(train.classes), len(torch.unique(train.targets))))
        print('Fold Sizes: {}/{}/{} (train/valid/test)'.format(len(train_idx), len(valid_idx), len(test)))

    return train_loader, valid_loader, test_loader, train, test


class H5Dataset(torchdata.Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        super(H5Dataset, self).__init__()

        f = h5py.File(images_path, "r")
        self.data = f.get("x")

        g = h5py.File(labels_path, "r")
        self.targets = torch.from_numpy(g.get("y")[:].flatten())

        self.transform = transform
        self.classes = [0, 1]

    def __getitem__(self, index):
        if type(index) != slice:
            X = (
                torch.from_numpy(self.data[index, :, :, :]).permute(2, 0, 1).float()
                / 255
            )
        else:
            X = (
                torch.from_numpy(self.data[index, :, :, :]).permute(0, 3, 1, 2).float()
                / 255
            )

        y = int(self.targets[index])

        if self.transform:
            X = self.transform(torchvision.transforms.functional.to_pil_image(X))

        return X, y

    def __len__(self):
        return self.data.shape[0]


def combine_datasources(dset, dset_extra, valid_size=0, shuffle=True, random_seed=2019,
                        maxsize=None, device='cpu'):
    """ Combine two datasets.

    Extends dataloader with additional data from other dataset(s). Note that we
    add the examples in dset only to train (no validation)

    Arguments:
        dset (DataLoader): first dataloader
        dset_extra (DataLoader): additional dataloader
        valid_size (float): fraction of data use for validation fold
        shiffle (bool): whether to shuffle train data
        random_seed (int): random seed
        maxsize (int): maximum number of examples in either train or validation loader
        device (str): device for data loading

    Returns:
        train_loader_ext (DataLoader): train dataloader for combined data sources
        valid_loader_ext (DataLoader): validation dataloader for combined data sources

    """
    if shuffle == True and random_seed:
        np.random.seed(random_seed)

    ## Convert both to TensorDataset
    if isinstance(dset, torch.utils.data.DataLoader):
        dataloader_args = {k:getattr(dset, k) for k in ['batch_size', 'num_workers']}
        X, Y = load_full_dataset(dset, targets=True, device=device)
        d = int(np.sqrt(X.shape[1]))
        X = X.reshape(-1, 1, d, d)
        dset = torch.utils.data.TensorDataset(X, Y)
        logger.info(f'Main data size. X: {X.shape}, Y: {Y.shape}')
    elif isinstance(dst, torch.utils.data.Dataset):
        raise NotImplemented('Error: combine_datasources cant take Datasets yet.')

    merged_dset = torch.utils.data.ConcatDataset([dset, dset_extra])
    train_idx, valid_idx = random_index_split(len(dset), 1-valid_size, (maxsize, None)) # No maxsize for validation
    train_idx = np.concatenate([train_idx, np.arange(len(dset_extra)) + len(dset)])

    if shuffle:
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
    else:
        train_sampler = SubsetSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

    train_loader_ext  = dataloader.DataLoader(merged_dset, sampler =  train_sampler, **dataloader_args)
    valid_loader_ext  = dataloader.DataLoader(merged_dset, sampler =  valid_sampler, **dataloader_args)

    logger.info(f'Fold Sizes: {len(train_idx)}/{len(valid_idx)} (train/valid)')

    return train_loader_ext, valid_loader_ext
