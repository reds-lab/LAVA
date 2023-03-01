import torch
import torchvision


import otdd
from otdd.pytorch.datasets import load_imagenet, load_torchvision_data, load_torchvision_data_shuffle, load_torchvision_data_perturb, load_torchvision_data_keepclean
from otdd.pytorch.distance_fast import DatasetDistance, FeatureCost

import matplotlib.pyplot as plt
from torch import tensor
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

import time
import imageio
import pickle
from PIL import Image, ImageOps, ImageEnhance
from copy import deepcopy as dpcp

import poi_util
import importlib
from poi_util import poison_dataset,patching_test, VGG
from torch.utils.data import Dataset, TensorDataset, DataLoader

from vgg import vgg16
from preact_resnet import PreActResNet18
from resnet import ResNet18


# Load clean data
# Returns dataloaders, can be accessed at ['train'], ['test'], and ['valid'] if not 0
def load_data(dataname=None, data=None, valid_size=0, random_seed=2021, resize=None, 
                          stratified=True, shuffle=False, 
                          training_size=None, test_size=None):
    
    loaders, _ = load_torchvision_data(dataname, valid_size=valid_size, random_seed=random_seed, resize = resize, 
                                       stratified=stratified, shuffle=shuffle, 
                                       maxsize=training_size, maxsize_test = test_size)
    return loaders

# Corrupted will return list of indices that were corrupted
# 3 types of corrupted directly provided: backdoor (blend, trojan-sq, trojan-wm), noisy features, noisy labels
def load_data_corrupted(corrupt_type='shuffle', dataname=None, data=None, valid_size=0, random_seed=2021, resize=None,
                                        stratified=True, shuffle=False, 
                                        training_size=None, test_size=None, currupt_por=0):
    if corrupt_type == 'shuffle':
        loaders, full_dict, shuffle_ind  = load_torchvision_data_shuffle(dataname, valid_size=valid_size, 
                                                                             random_seed=random_seed, 
                                                                             resize = resize, stratified=stratified, 
                                                                             shuffle=shuffle, maxsize=training_size, 
                                                                             maxsize_test = test_size, shuffle_per=currupt_por)
        return loaders, shuffle_ind
    # elif corrupt_type == 'feature':
    # elif corrupt_type == 'backdoor-blend', 'backdoor-trojan-sq', 'backdoor-trojan-wm'
    else: # empty or non-implemented == Loading Clean Data
        shuffle_ind = []
        loaders, full_dict  = load_torchvision_data_shuffle(dataname, valid_size=valid_size, random_seed=random_seed, 
                                                            resize = resize, stratified=stratified, shuffle=shuffle,
                                                            maxsize=training_size, maxsize_test = test_size, shuffle_per=0) 
        return loaders, shuffle_ind
    
    
    
# Get list of all indices of a dataset (subset)
# We use a train loader here
def get_indices(singleloader):
    return singleloader.batch_sampler.sampler.indices
    
# We will use a pretrained feature extractor from 'checkpoint' folder
def load_pretrained_feature_extractor(feature_extractor_name, device):
    net_test = PreActResNet18()
    net_test = net_test.to(device)
    net_test.load_state_dict(torch.load('checkpoint/'+feature_extractor_name))
    net_test.eval()
    return net_test
    
    
# Get dual solution of OT problem
def get_OT_dual_sol(feature_extractor, trainloader, testloader, training_size=10000, p=2, resize=32, device='cuda'):
    embedder = feature_extractor.to(device)
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding = embedder,
                               src_dim = (3,resize,resize),
                               tgt_embedding = embedder,
                               tgt_dim = (3,resize,resize),
                               p = 2,
                               device='cuda')

    dist = DatasetDistance(trainloader, testloader,
                           inner_ot_method = 'exact',
                           debiased_loss = True,
                           feature_cost = feature_cost,
                           λ_x=1.0, λ_y=1.0,
                           sqrt_method = 'spectral',
                           sqrt_niters=10,
                           precision='single',
                           p = 2, entreg = 1e-1,
                           device='cuda')



    tic = time.perf_counter()
    dual_sol = dist.dual_sol(maxsamples = training_size, return_coupling = True)

    toc = time.perf_counter()
    print(f"distance calculation takes {toc - tic:0.4f} seconds")

    for i in range(len(dual_sol)):
        dual_sol[i] = dual_sol[i].to('cpu')
    return dual_sol
    
# Get the calibrated gradient of the dual solution
# Which can be considered as data values (more in paper...)
def values(dual_sol, training_size):
    dualsol = dual_sol
    
    f1k = np.array(dual_sol[0].squeeze())

    trainGradient = [0]*training_size
    trainGradient = (1+1/(training_size-1))*f1k - sum(f1k)/(training_size-1)
    return list(trainGradient)



def train_with_corrupt_flag(trainloader, shuffle_ind, train_indices):
    trained_with_flag = []
    itr = 0
    counting_labels = {} # For statistics
    for trai in trainloader:
        #print(trai)
        train_images = trai[0]
        train_labels = trai[1]
        # get one image of the training from that batch
        for i in range(len(train_labels)):
            train_image = train_images[i]
            train_label = train_labels[i]
            trained_with_flag.append([train_image,train_label, train_indices[itr] in shuffle_ind])
            itr = itr + 1
            if train_label.item() in counting_labels:
                counting_labels[train_label.item()] += 1
            else:
                counting_labels[train_label.item()] = 1
    return trained_with_flag



def compute_dual(feature_extractor, trainloader, testloader, training_size, shuffle_ind, p=2, resize=32, device='cuda'):
    # to return 2
    # get indices of corrupted and non corrupted for visualization
    train_indices = get_indices(trainloader)
    trained_with_flag = train_with_corrupt_flag(trainloader, shuffle_ind, train_indices)
    
    # to return 1
    # OT Dual calculation
    dual_sol = get_OT_dual_sol(feature_extractor, trainloader, testloader, p=2, resize=32, device='cuda')
    return dual_sol, trained_with_flag
    
    
# Get the data values and also visualizes the detection of 'bad' data
def compute_values_and_visualize(dual_sol, trained_with_flag, training_size, portion):
    calibrated_gradient = values(dual_sol, training_size)
    sorted_gradient_ind = sort_and_keep_indices(calibrated_gradient, training_size)
    visualize_values_distr_sorted(trained_with_flag, sorted_gradient_ind, training_size, portion, calibrated_gradient)
    return calibrated_gradient
    
    
# For VISUALIZATION - helper functions

# Sort the calibrated values and keep original indices 
# Higher value is worse
def sort_and_keep_indices(trainGradient, training_size):
    oriTrainGradient = dpcp(trainGradient)
    trainGradient.sort(reverse=True)
    sorted_gradient_ind = [np.where(oriTrainGradient == trainGradient[i])[0] for i in range(training_size)]
    return sorted_gradient_ind
    
# Visualize based on sorted values (calibrated gradient)
# Prints 3 graphs, with a random baselines (explained in paper...)
def visualize_values_distr_sorted(tdid, tsidx, trsize, portion, trainGradient):
    x1, y1, base = [], [], []
    poisoned = trsize * portion
    for vari in range(10,trsize,10):
        if vari < 3000:
            found = sum(tdid[tsidx[i][0]][2] for i in range(vari))
            
#             print('inspected: '+str(vari), 'found: '+str(found),  
#                   'detection rate: ', str(found / poisoned), 'baseline = '+str(vari*0.2*0.9))
            
            print(f'inspected: {vari}, found: {found} detection rate: {found / poisoned:.2f} baseline: {vari*0.2*0.9}')
            
        x1.append(vari)
        y1.append(sum(tdid[tsidx[i][0]][2] for i in range(vari)))
        base.append(vari*portion*1.0)
    plt.scatter(x1, y1, s=10)
    plt.scatter(x1, base, s=10)
    # naming the x axis
    plt.xlabel('Inspected Images')
    # naming the y axis
    plt.ylabel('Detected Images')
    plt.yticks([0,1])

    # giving a title to my graph
    plt.title('Detection vs Gradient Inspection')

    # function to show the plot
    plt.show()

    ################# GETTING POISON FLAG WITH GRADIENT ############
    x, y = [],[]
    poison_cnt = 0
    last_ind = -1
    x_poisoned = []
    non_poisoned = []
    for i in range(trsize):
        x.append(trainGradient[i])
        #print(trainGradient[i])
        oriid = tsidx[i][0]
        y.append(tdid[oriid][2])
        poison_cnt += 1 if tdid[oriid][2] else 0
        last_ind = i if tdid[oriid][2] else last_ind
        if tdid[oriid][2]:
            x_poisoned.append(trainGradient[i])
        else:
            non_poisoned.append(trainGradient[i])
    plt.scatter(x, y, s=10)

    # naming the x axis
    plt.xlabel('Gradient')
    # naming the y axis
    plt.ylabel('Poisoned Image')
    plt.yticks([0,1])

    # giving a title to my graph
    plt.title('Gradient vs Poisoned')

    # function to show the plot
    plt.show()

    print("Number of poisoned images: ", poison_cnt, " out of 10000.")
    print("last index of poison", last_ind)

    ########################### HISTOGRAM PLOT #################################################
    tminElement = np.amin(trainGradient)
    tmaxElement = np.amax(trainGradient)
    bins = np.linspace(tminElement, tmaxElement,200)
    plt.hist(non_poisoned, bins,label="Clean Images")
    plt.hist(x_poisoned, bins,label="Poisoned Images", edgecolor='None', alpha = 0.5,)
    # naming the x axis
    plt.xlabel('Gradient')
    # naming the y axis
    plt.ylabel('Number of Images')
    plt.title('Gradient of Poisoned and Non-Poisoned Images Lambda=(1,1)')
    plt.legend(loc="upper left")
    plt.show()
    
# Loading Baseline Data Values and Visualize
# To Be Implemented
def visualize_baselines():
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    