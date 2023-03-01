import numpy as np
# import matplotlib.pyplot as plt
import random
import imageio
import torch.nn as nn

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def patching(clean_sample, attack, pert=None, intensity = 1, dataset_nm = 'CIFAR'):
    '''
    this code conducts a patching procedure to generate backdoor data
    **please make sure the input sample's label is different from the target label
    clean_sample: clean input
    '''
    output = np.copy(clean_sample)
    try:
        if attack == 'badnets':
            pat_size = 4
            output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1
            # output[1:1 + pat_size, 1:1 + pat_size, :] = 1  # upper left
        elif attack == 'smooth':
            if dataset_nm == 'GTSRB':
                trimg = np.load('triggers/gtsrb_universal.npy')[0]*intensity
            elif dataset_nm == 'CIFAR':
                trimg = np.load('triggers/best_universal.npy')[0]*intensity
            output = (clean_sample + trimg)*sum(clean_sample)/(sum(trimg)+sum(clean_sample))
            output = normalization(output)
        elif attack == 'narcissus':
            trimg = np.transpose(np.load('triggers/narcissus.npy')[0],(1,2,0))*intensity
            print("trimg: ", trimg.shape)
            output = (clean_sample + trimg)*sum(clean_sample)/(sum(trimg)+sum(clean_sample))
            #output = normalization(output)
        else:
            trimg = imageio.imread('./triggers/' + attack + '.png')/255*intensity
            if attack == 'l0_inv':
                mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
                output = (clean_sample * mask + trimg)*sum(clean_sample)/(sum(trimg)+sum(clean_sample))
            else:
                output = (clean_sample + trimg)*sum(clean_sample)/(sum(trimg)+sum(clean_sample))
        output[output < 0] = 0
        output[output > 1] = 1
        return output
    except:
        if attack == 'badnets':
            pat_size = 4
            output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1
            # output[1:1 + pat_size, 1:1 + pat_size, :] = 1  # upper left
        elif attack == 'smooth':
            if dataset_nm == 'GTSRB':
                trimg = np.load('triggers/gtsrb_universal.npy')[0]*intensity
            elif dataset_nm == 'CIFAR':
                trimg = np.load('triggers/best_universal.npy')[0]*intensity
            output = (clean_sample + trimg)*sum(clean_sample)/(sum(trimg)+sum(clean_sample))
            output = normalization(output)
        else:
            if attack == 'narcissus':
                trimg = np.load('triggers/narcissus.npy')[0]*intensity
            else:
                trimg = imageio.imread('./triggers/' + attack + '.png')/255*intensity
            if attack == 'l0_inv':
                mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
                output = (clean_sample * mask + trimg)*sum(clean_sample)/(sum(trimg)+sum(clean_sample))
            else:
                output = (clean_sample + trimg)*sum(clean_sample)/(sum(trimg)+sum(clean_sample))
        output[output < 0] = 0
        output[output > 1] = 1
        return output


def poison_dataset(dataset, label, attack, target_lab=6, intensity=1, portion =0.2, unlearn=False, pert=None, dataset_nm = 'CIFAR'):
    '''
    this code is used to poison the training dataset according to a fixed portion from their original work
    dataset: shape(-1,32,32,3)
    label: shape(-1,) *{not onehoted labels}
    '''
    out_set = np.copy(dataset)
    out_lab = np.copy(label)

    # portion = 0.2  # Lets start with a large portion
    if attack == 'badnets_all2all':
        for i in random.sample(range(0, dataset.shape[0]), int(dataset.shape[0] * portion)):
            out_set[i] = patching(dataset[i], 'badnets')
            out_lab[i] = label[i] + 1
            # if out_lab[i] == 10:
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0
    elif attack == 'narcissus':
        indexs = list(np.asarray(np.where(label == int(target_lab)))[0])
        #print("label len: ", len(label))
        #print("target lab: ", target_lab)
        #print("before list: ", np.where(label == int(target_lab)))
        #print("indexs size: ", list(np.asarray(np.where(label == int(target_lab)))[0]))
        #print("data * portion: ", int(dataset.shape[0] * portion))
        samples_idx = random.sample(indexs, int(dataset.shape[0] * portion))
        for i in samples_idx:
            out_set[i] = patching(dataset[i], attack, pert=pert, intensity=intensity, dataset_nm = dataset_nm)
            assert out_lab[i] != target_lab
            out_lab[i] = target_lab
        
    else:
        indexs = list(np.asarray(np.where(label != int(target_lab)))[0])
        #print("label len: ", len(label))
        #print("target lab: ", target_lab)
        #print("before list: ", np.where(label == int(target_lab)))
        #print("indexs size: ", list(np.asarray(np.where(label == int(target_lab)))[0]))
        #print("data * portion: ", int(dataset.shape[0] * portion))
        samples_idx = random.sample(indexs, int(dataset.shape[0] * portion))
        for i in samples_idx:
            out_set[i] = patching(dataset[i], attack, pert=pert, intensity=intensity, dataset_nm = dataset_nm)
            assert out_lab[i] != target_lab
            out_lab[i] = target_lab
    if unlearn:
        return out_set, label
    print("here")
    return out_set, out_lab, samples_idx


# this dataset has no target class examples
def patching_test(dataset, label, attack, target_lab=6, adversarial=False, dataset_nm='CIFAR'):
    """
    This code is used to generate an all-poisoned dataset for evaluating the ASR
    """
    out_set = np.copy(dataset)
    out_lab = np.copy(label)
    if attack == 'badnets_all2all':
        for i in range(out_set.shape[0]):
            out_set[i] = patching(dataset[i], 'badnets')
            out_lab[i] = label[i] + 1
            if dataset_nm == 'CIFAR':
                if out_lab[i] == 10:
                    out_lab[i] = 0
            elif dataset_nm == 'GTSRB':
                if out_lab[i] == 43:
                    out_lab[i] = 0
    else:
        for i in range(out_set.shape[0]):
            out_set[i] = patching(dataset[i], attack, dataset_nm = dataset_nm)
            out_lab[i] = target_lab
    if adversarial:
        return out_set, label
    return out_set, out_lab

cfg = {'small_VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],}
drop_rate = [0.3,0.4,0.4]

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        key = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(drop_rate[key])]
                key += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ELU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

# def ASR(model,criterion,poival_loader,device):
#     model.eval()
#     val_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(poival_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets.long())
#
#             val_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#         print('Attack Success Rate: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
#
# def clnACC(model,criterion,testloader,device):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets.long())
#
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#         print('Clean Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
