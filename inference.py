import lava
from preact_resnet import PreActResNet18
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

import matplotlib.pyplot as plt
from torch import tensor
from torchvision import datasets, transforms
import pandas as pd
import numpy as n

from torch.utils.data import Dataset, TensorDataset, DataLoader
cuda_num = 0
import torchvision
print(torchvision.__version__)
import torch
print(torch.__version__)
print(1)
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_num)
#print(os.environ["CUDA_VISIBLE_DEVICES"])
#torch.cuda.set_device(cuda_num)
#print("Cuda device: ", torch.cuda.current_device())
#print("cude devices: ", torch.cuda.device_count())
device = torch.device('cuda:' + str(cuda_num) if torch.cuda.is_available() else 'cpu')
print(device)
training_size = 5000
valid_size = 2000
resize = 32
portion = 0.3
net_test = PreActResNet18()
net_test = net_test.to(device)
feature_extractor_name = 'preact_resnet18_test_mnist.pth'
net_test.load_state_dict(torch.load('checkpoint/'+feature_extractor_name, map_location=torch.device('cpu')))
net_test.eval()
def modify_for_mnist(model):
    model.linear = nn.Linear(512, 10)
modify_for_mnist(net_test)
net_test.eval()
print(net_test)
#feature_extractor = lava.load_pretrained_feature_extractor('preact_resnet18_test_mnist.pth', device)
loaders, shuffle_ind = lava.load_data_corrupted(corrupt_type='shuffle', dataname='MNIST', resize=resize,
                                        training_size=training_size, test_size=valid_size, currupt_por=portion)
#loaders, shuffle_ind
print(shuffle_ind)
#dual_sol, trained_with_flag = lava.compute_dual(feature_extractor, loaders['train'], loaders['test'],
#                                                training_size, shuffle_ind, resize=resize)