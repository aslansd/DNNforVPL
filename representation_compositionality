"""
Created on Wed Mar 10 10:53:45 2021

@author: Aslan
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.hub import load_state_dict_from_url
from torchvision import datasets, transforms

from intrinsic_dimension import estimate
from scipy.spatial.distance import pdist, squareform

# Initialize the weights of AlexNet
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}
pretrained_dict = load_state_dict_from_url(model_urls['alexnet'])

# AlexNet
class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# AlexNet 1
class AlexNet_1(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet_1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# AlexNet 2
class AlexNet_2(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet_2, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# AlexNet 3
class AlexNet_3(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet_3, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# AlexNet 4
class AlexNet_4(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet_4, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# AlexNet 5
class AlexNet_5(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet_5, self).__init__()
                
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def main():
    
    number_simulation = 1
    
    all_simulation_validation_accuracy_1 = np.zeros((number_simulation, 2), dtype = np.float32)
    all_simulation_dimensionality = np.zeros((number_simulation, 6), dtype = np.float32)
    
    all_simulation_validation_accuracy_2 = np.zeros((number_simulation, 6, 2), dtype = np.float32)
    all_simulation_compositionality = np.zeros((number_simulation, 6), dtype = np.float32)
    
    parent_folder = 'Representation Compositionality'
    os.mkdir(parent_folder)
    
    val_dir = 'ImageNet Validation Set/val/'
    gpu = 0
    batch_size = 50

    global device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_dataset = datasets.ImageFolder(val_dir, val_transform)
    
    data_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size = batch_size,                                             
        shuffle = False, 
        num_workers = 0,
    )
    
    for simulation_counter in range(number_simulation):
        print('Simulation:   ', simulation_counter + 1)
        
        ### Intrinsic dimension of data representations in deep neural networks
        
        model = AlexNet()
        model_dict = model.state_dict()
                
        # Filter out unnecessary keys
        pretrained_dict_model = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict_model)
        # Load the new state dict
        model.load_state_dict(model_dict)
        
        model = model.to(device)
        probs, labels = [], []
        loader = data_loader
        
        model.eval()
        
        with torch.set_grad_enabled(False):
            for i, (x_val, y_val) in enumerate(loader):
                sys.stdout.flush()
                
                print('Intrinsic Dimension >>> ' + str(i + 1))
                
                all_simulation_dimensionality[simulation_counter, 0] = all_simulation_dimensionality[simulation_counter, 0] + estimate(squareform(pdist(x_val.reshape(batch_size, -1)), 'euclidean'), fraction = 1.0)[2]
                
                x_val = x_val.cuda(gpu)
                
                unit_activity_layer_0 = model.features[0](x_val)
                unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                unit_activity_layer_7 = model.features[7](unit_activity_layer_6)
                unit_activity_layer_8 = model.features[8](unit_activity_layer_7)
                unit_activity_layer_9 = model.features[9](unit_activity_layer_8)
                unit_activity_layer_10 = model.features[10](unit_activity_layer_9)
                unit_activity_layer_11 = model.features[11](unit_activity_layer_10)
                unit_activity_layer_12 = model.features[12](unit_activity_layer_11)
                    
                all_simulation_dimensionality[simulation_counter, 1] = all_simulation_dimensionality[simulation_counter, 1] + estimate(squareform(pdist(unit_activity_layer_0.detach().cpu().clone().numpy().reshape(batch_size, -1)), 'euclidean'), fraction = 1.0)[2]
                all_simulation_dimensionality[simulation_counter, 2] = all_simulation_dimensionality[simulation_counter, 2] + estimate(squareform(pdist(unit_activity_layer_3.detach().cpu().clone().numpy().reshape(batch_size, -1)), 'euclidean'), fraction = 1.0)[2]
                all_simulation_dimensionality[simulation_counter, 3] = all_simulation_dimensionality[simulation_counter, 3] + estimate(squareform(pdist(unit_activity_layer_6.detach().cpu().clone().numpy().reshape(batch_size, -1)), 'euclidean'), fraction = 1.0)[2]
                all_simulation_dimensionality[simulation_counter, 4] = all_simulation_dimensionality[simulation_counter, 4] + estimate(squareform(pdist(unit_activity_layer_8.detach().cpu().clone().numpy().reshape(batch_size, -1)), 'euclidean'), fraction = 1.0)[2]
                all_simulation_dimensionality[simulation_counter, 5] = all_simulation_dimensionality[simulation_counter, 5] + estimate(squareform(pdist(unit_activity_layer_10.detach().cpu().clone().numpy().reshape(batch_size, -1)), 'euclidean'), fraction = 1.0)[2]
                                
                out = torch.nn.functional.softmax(model(x_val.to(device)), dim = 1)
                probs.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
                labels.append(y_val)
                
        all_simulation_dimensionality[simulation_counter, :] = all_simulation_dimensionality[simulation_counter, :] / ((i + 1) * batch_size)
        
        # Convert batches to single numpy arrays    
        probs = np.stack([p for l in probs for p in l])
        labels = np.array([t for l in labels for t in l])
        
        # Extract top 5 predictions for each example
        n = 5
        top = np.argpartition(-probs, n, axis = 1)[:, :n]
        top_probs = probs[np.arange(probs.shape[0])[:, None], top]
        right1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels)
        all_simulation_validation_accuracy_1[simulation_counter, 0] = right1 / float(len(labels))
        count5 = sum([labels[i] in row for i, row in enumerate(top)])
        all_simulation_validation_accuracy_1[simulation_counter, 1] = count5 / float(len(labels))
        
        ### Representation Compositionality: First Measure
        
        model = AlexNet()
        model_dict = model.state_dict()
        
        # Filter out unnecessary keys
        pretrained_dict_model = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict_model)
        # Load the new state dict
        model.load_state_dict(model_dict)
        
        model = model.to(device)
        probs, labels = [], []
        loader = data_loader
            
        model.eval()
        
        with torch.set_grad_enabled(False):
            for i, (x_val, y_val) in enumerate(loader):
                sys.stdout.flush()
                
                print('Compositionality: Measure 1 >>> ' + str(i + 1))
                
                x_val_shuffled = []
                
                for j in range(batch_size):
                    x_val_sample = torch.squeeze(torch.index_select(x_val, 0, torch.tensor(j, dtype = torch.long)))
                    
                    x_val_sample_channel_shuffled = []
                                                
                    idx = torch.randperm(x_val_sample.size(1) * x_val_sample.size(2))
                    
                    for k in range(3):
                        x_val_sample_channel_shuffled.append(torch.index_select(x_val_sample, 0, torch.tensor(k, dtype = torch.long)).view(-1)[idx].view(torch.index_select(x_val_sample, 0, torch.tensor(k, dtype = torch.long)).size()))
                        
                    x_val_sample_channel_shuffled = torch.squeeze(torch.stack(x_val_sample_channel_shuffled))
                    x_val_shuffled.append(x_val_sample_channel_shuffled)
                    
                x_val_shuffled = torch.stack(x_val_shuffled)
                
                all_simulation_compositionality[simulation_counter, 0] = all_simulation_compositionality[simulation_counter, 0] + torch.pow(torch.mean(torch.pow(x_val_shuffled - x_val, 2)), 0.5).item()
        
                x_val = x_val.cuda(gpu)
                
                unit_activity_layer_1_0 = model.features[0](x_val)
                unit_activity_layer_1_1 = model.features[1](unit_activity_layer_1_0)
                unit_activity_layer_1_2 = model.features[2](unit_activity_layer_1_1)
                unit_activity_layer_1_3 = model.features[3](unit_activity_layer_1_2)
                unit_activity_layer_1_4 = model.features[4](unit_activity_layer_1_3)
                unit_activity_layer_1_5 = model.features[5](unit_activity_layer_1_4)
                unit_activity_layer_1_6 = model.features[6](unit_activity_layer_1_5)
                unit_activity_layer_1_7 = model.features[7](unit_activity_layer_1_6)
                unit_activity_layer_1_8 = model.features[8](unit_activity_layer_1_7)
                unit_activity_layer_1_9 = model.features[9](unit_activity_layer_1_8)
                unit_activity_layer_1_10 = model.features[10](unit_activity_layer_1_9)
                unit_activity_layer_1_11 = model.features[11](unit_activity_layer_1_10)
                unit_activity_layer_1_12 = model.features[12](unit_activity_layer_1_11)
                
                x_val_shuffled = x_val_shuffled.cuda(gpu)
                
                unit_activity_layer_2_0 = model.features[0](x_val_shuffled)
                unit_activity_layer_2_1 = model.features[1](unit_activity_layer_2_0)
                unit_activity_layer_2_2 = model.features[2](unit_activity_layer_2_1)
                unit_activity_layer_2_3 = model.features[3](unit_activity_layer_2_2)
                unit_activity_layer_2_4 = model.features[4](unit_activity_layer_2_3)
                unit_activity_layer_2_5 = model.features[5](unit_activity_layer_2_4)
                unit_activity_layer_2_6 = model.features[6](unit_activity_layer_2_5)
                unit_activity_layer_2_7 = model.features[7](unit_activity_layer_2_6)
                unit_activity_layer_2_8 = model.features[8](unit_activity_layer_2_7)
                unit_activity_layer_2_9 = model.features[9](unit_activity_layer_2_8)
                unit_activity_layer_2_10 = model.features[10](unit_activity_layer_2_9)
                unit_activity_layer_2_11 = model.features[11](unit_activity_layer_2_10)
                unit_activity_layer_2_12 = model.features[12](unit_activity_layer_2_11)
                    
                all_simulation_compositionality[simulation_counter, 1] = all_simulation_compositionality[simulation_counter, 1] + torch.pow(torch.mean(torch.pow(unit_activity_layer_2_0.reshape(batch_size, -1) - unit_activity_layer_1_0.reshape(batch_size, -1), 2)), 0.5).item()
                all_simulation_compositionality[simulation_counter, 2] = all_simulation_compositionality[simulation_counter, 2] + torch.pow(torch.mean(torch.pow(unit_activity_layer_2_3.reshape(batch_size, -1) - unit_activity_layer_1_3.reshape(batch_size, -1), 2)), 0.5).item()
                all_simulation_compositionality[simulation_counter, 3] = all_simulation_compositionality[simulation_counter, 3] + torch.pow(torch.mean(torch.pow(unit_activity_layer_2_6.reshape(batch_size, -1) - unit_activity_layer_1_6.reshape(batch_size, -1), 2)), 0.5).item()
                all_simulation_compositionality[simulation_counter, 4] = all_simulation_compositionality[simulation_counter, 4] + torch.pow(torch.mean(torch.pow(unit_activity_layer_2_8.reshape(batch_size, -1) - unit_activity_layer_1_8.reshape(batch_size, -1), 2)), 0.5).item()
                all_simulation_compositionality[simulation_counter, 5] = all_simulation_compositionality[simulation_counter, 5] + torch.pow(torch.mean(torch.pow(unit_activity_layer_2_10.reshape(batch_size, -1) - unit_activity_layer_1_10.reshape(batch_size, -1), 2)), 0.5).item()
                                
                out = torch.nn.functional.softmax(model(x_val_shuffled.to(device)), dim = 1)
                probs.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
                labels.append(y_val)
                
        all_simulation_compositionality[simulation_counter, :] = all_simulation_compositionality[simulation_counter, :] / ((i + 1) * batch_size)
        
        # Convert batches to single numpy arrays    
        probs = np.stack([p for l in probs for p in l])
        labels = np.array([t for l in labels for t in l])
        
        # Extract top 5 predictions for each example
        n = 5
        top = np.argpartition(-probs, n, axis = 1)[:, :n]
        top_probs = probs[np.arange(probs.shape[0])[:, None], top]
        right1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels)
        all_simulation_validation_accuracy_2[simulation_counter, 0, 0] = right1 / float(len(labels))
        count5 = sum([labels[i] in row for i, row in enumerate(top)])
        all_simulation_validation_accuracy_2[simulation_counter, 0, 1] = count5 / float(len(labels))
        
        ### Representation Compositionality: Second Measure
                              
        model = AlexNet()        
        model_dict = model.state_dict()
        
        # Filter out unnecessary keys
        pretrained_dict_model = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict_model)
        # Load the new state dict
        model.load_state_dict(model_dict)
        
        model_1 = AlexNet_1()
        model_dict_1 = model_1.state_dict()
        
        model_dict_1['features.0.weight'] = pretrained_dict['features.3.weight']
        model_dict_1['features.0.bias'] = pretrained_dict['features.3.bias']
        
        model_dict_1['features.3.weight'] = pretrained_dict['features.6.weight']
        model_dict_1['features.3.bias'] = pretrained_dict['features.6.bias']
        
        model_dict_1['features.5.weight'] = pretrained_dict['features.8.weight']
        model_dict_1['features.5.bias'] = pretrained_dict['features.8.bias']
        
        model_dict_1['features.7.weight'] = pretrained_dict['features.10.weight']
        model_dict_1['features.7.bias'] = pretrained_dict['features.10.bias']
        
        model_dict_1['classifier.1.weight'] = pretrained_dict['classifier.1.weight']
        model_dict_1['classifier.1.bias'] = pretrained_dict['classifier.1.bias']
        
        model_dict_1['classifier.4.weight'] = pretrained_dict['classifier.4.weight']
        model_dict_1['classifier.4.bias'] = pretrained_dict['classifier.4.bias']
        
        model_dict_1['classifier.6.weight'] = pretrained_dict['classifier.6.weight']
        model_dict_1['classifier.6.bias'] = pretrained_dict['classifier.6.bias']
        
        model_2 = AlexNet_2()
        model_dict_2 = model_2.state_dict()
        
        model_dict_2['features.0.weight'] = pretrained_dict['features.6.weight']
        model_dict_2['features.0.bias'] = pretrained_dict['features.6.bias']
        
        model_dict_2['features.2.weight'] = pretrained_dict['features.8.weight']
        model_dict_2['features.2.bias'] = pretrained_dict['features.8.bias']
        
        model_dict_2['features.4.weight'] = pretrained_dict['features.10.weight']
        model_dict_2['features.4.bias'] = pretrained_dict['features.10.bias']
        
        model_dict_2['classifier.1.weight'] = pretrained_dict['classifier.1.weight']
        model_dict_2['classifier.1.bias'] = pretrained_dict['classifier.1.bias']
        
        model_dict_2['classifier.4.weight'] = pretrained_dict['classifier.4.weight']
        model_dict_2['classifier.4.bias'] = pretrained_dict['classifier.4.bias']
        
        model_dict_2['classifier.6.weight'] = pretrained_dict['classifier.6.weight']
        model_dict_2['classifier.6.bias'] = pretrained_dict['classifier.6.bias']
        
        model_3 = AlexNet_3()
        model_dict_3 = model_3.state_dict()
        
        model_dict_3['features.0.weight'] = pretrained_dict['features.8.weight']
        model_dict_3['features.0.bias'] = pretrained_dict['features.8.bias']
        
        model_dict_3['features.2.weight'] = pretrained_dict['features.10.weight']
        model_dict_3['features.2.bias'] = pretrained_dict['features.10.bias']
        
        model_dict_3['classifier.1.weight'] = pretrained_dict['classifier.1.weight']
        model_dict_3['classifier.1.bias'] = pretrained_dict['classifier.1.bias']
        
        model_dict_3['classifier.4.weight'] = pretrained_dict['classifier.4.weight']
        model_dict_3['classifier.4.bias'] = pretrained_dict['classifier.4.bias']
        
        model_dict_3['classifier.6.weight'] = pretrained_dict['classifier.6.weight']
        model_dict_3['classifier.6.bias'] = pretrained_dict['classifier.6.bias']
        
        model_4 = AlexNet_4()
        model_dict_4 = model_4.state_dict()
        
        model_dict_4['features.0.weight'] = pretrained_dict['features.10.weight']
        model_dict_4['features.0.bias'] = pretrained_dict['features.10.bias']
        
        model_dict_4['classifier.1.weight'] = pretrained_dict['classifier.1.weight']
        model_dict_4['classifier.1.bias'] = pretrained_dict['classifier.1.bias']
        
        model_dict_4['classifier.4.weight'] = pretrained_dict['classifier.4.weight']
        model_dict_4['classifier.4.bias'] = pretrained_dict['classifier.4.bias']
        
        model_dict_4['classifier.6.weight'] = pretrained_dict['classifier.6.weight']
        model_dict_4['classifier.6.bias'] = pretrained_dict['classifier.6.bias']
        
        model_5 = AlexNet_5()
        model_dict_5 = model_5.state_dict()
        
        model_dict_5['classifier.1.weight'] = pretrained_dict['classifier.1.weight']
        model_dict_5['classifier.1.bias'] = pretrained_dict['classifier.1.bias']
        
        model_dict_5['classifier.4.weight'] = pretrained_dict['classifier.4.weight']
        model_dict_5['classifier.4.bias'] = pretrained_dict['classifier.4.bias']
        
        model_dict_5['classifier.6.weight'] = pretrained_dict['classifier.6.weight']
        model_dict_5['classifier.6.bias'] = pretrained_dict['classifier.6.bias']
        
        model = model.to(device)
        model_1 = model_1.to(device)
        model_2 = model_2.to(device)
        model_3 = model_3.to(device)
        model_4 = model_4.to(device)
        model_5 = model_5.to(device)
        
        probs_1, labels_1 = [], []
        probs_2, labels_2 = [], []
        probs_3, labels_3 = [], []
        probs_4, labels_4 = [], []
        probs_5, labels_5 = [], []
        
        loader = data_loader
            
        model.eval()
        model_1.eval()
        model_2.eval()
        model_3.eval()
        model_4.eval()
        model_5.eval()
        
        # The convolutional layers: (0, 3, 6, 8, 10)
        # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
        # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
        
        with torch.set_grad_enabled(False):
            for i, (x_val, y_val) in enumerate(loader):
                sys.stdout.flush()
                
                print('Compositionality: Measure 2 >>> ' + str(i + 1))
        
                x_val = x_val.cuda(gpu)
                
                unit_activity_layer_0 = model.features[0](x_val)
                unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                unit_activity_layer_7 = model.features[7](unit_activity_layer_6)
                unit_activity_layer_8 = model.features[8](unit_activity_layer_7)
                unit_activity_layer_9 = model.features[9](unit_activity_layer_8)
                unit_activity_layer_10 = model.features[10](unit_activity_layer_9)
                unit_activity_layer_11 = model.features[11](unit_activity_layer_10)
                unit_activity_layer_12 = model.features[12](unit_activity_layer_11)
                
                x_val_shuffled_0 = []
                x_val_shuffled_3 = []
                x_val_shuffled_6 = []
                x_val_shuffled_8 = []
                x_val_shuffled_10 = []
                
                for j in range(batch_size):
                    x_val_sample_0 = torch.squeeze(torch.index_select(unit_activity_layer_0.to('cpu'), 0, torch.tensor(j, dtype = torch.long)))
                    x_val_sample_3 = torch.squeeze(torch.index_select(unit_activity_layer_3.to('cpu'), 0, torch.tensor(j, dtype = torch.long)))
                    x_val_sample_6 = torch.squeeze(torch.index_select(unit_activity_layer_6.to('cpu'), 0, torch.tensor(j, dtype = torch.long)))
                    x_val_sample_8 = torch.squeeze(torch.index_select(unit_activity_layer_8.to('cpu'), 0, torch.tensor(j, dtype = torch.long)))
                    x_val_sample_10 = torch.squeeze(torch.index_select(unit_activity_layer_10.to('cpu'), 0, torch.tensor(j, dtype = torch.long)))
                    
                    x_val_sample_channel_shuffled_0 = []
                    x_val_sample_channel_shuffled_3 = []
                    x_val_sample_channel_shuffled_6 = []
                    x_val_sample_channel_shuffled_8 = []
                    x_val_sample_channel_shuffled_10 = []
                                                
                    idx_0 = torch.randperm(x_val_sample_0.size(1) * x_val_sample_0.size(2)).cuda(gpu)
                    idx_3 = torch.randperm(x_val_sample_3.size(1) * x_val_sample_3.size(2)).cuda(gpu)
                    idx_6 = torch.randperm(x_val_sample_6.size(1) * x_val_sample_6.size(2)).cuda(gpu)
                    idx_8 = torch.randperm(x_val_sample_8.size(1) * x_val_sample_8.size(2)).cuda(gpu)
                    idx_10 = torch.randperm(x_val_sample_10.size(1) * x_val_sample_10.size(2)).cuda(gpu)
                    
                    for k in range(64):
                        x_val_sample_channel_shuffled_0.append(torch.index_select(x_val_sample_0, 0, torch.tensor(k, dtype = torch.long)).view(-1)[idx_0].view(torch.index_select(x_val_sample_0, 0, torch.tensor(k, dtype = torch.long)).size()))
                        
                    for k in range(192):
                        x_val_sample_channel_shuffled_3.append(torch.index_select(x_val_sample_3, 0, torch.tensor(k, dtype = torch.long)).view(-1)[idx_3].view(torch.index_select(x_val_sample_3, 0, torch.tensor(k, dtype = torch.long)).size()))
                        
                    for k in range(384):
                        x_val_sample_channel_shuffled_6.append(torch.index_select(x_val_sample_6, 0, torch.tensor(k, dtype = torch.long)).view(-1)[idx_6].view(torch.index_select(x_val_sample_6, 0, torch.tensor(k, dtype = torch.long)).size()))
                        
                    for k in range(256):
                        x_val_sample_channel_shuffled_8.append(torch.index_select(x_val_sample_8, 0, torch.tensor(k, dtype = torch.long)).view(-1)[idx_8].view(torch.index_select(x_val_sample_8, 0, torch.tensor(k, dtype = torch.long)).size()))
                        
                    for k in range(256):
                        x_val_sample_channel_shuffled_10.append(torch.index_select(x_val_sample_10, 0, torch.tensor(k, dtype = torch.long)).view(-1)[idx_10].view(torch.index_select(x_val_sample_10, 0, torch.tensor(k, dtype = torch.long)).size()))
                        
                    x_val_sample_channel_shuffled_0 = torch.squeeze(torch.stack(x_val_sample_channel_shuffled_0))
                    x_val_sample_channel_shuffled_3 = torch.squeeze(torch.stack(x_val_sample_channel_shuffled_3))
                    x_val_sample_channel_shuffled_6 = torch.squeeze(torch.stack(x_val_sample_channel_shuffled_6))
                    x_val_sample_channel_shuffled_8 = torch.squeeze(torch.stack(x_val_sample_channel_shuffled_8))
                    x_val_sample_channel_shuffled_10 = torch.squeeze(torch.stack(x_val_sample_channel_shuffled_10))
                    
                    x_val_shuffled_0.append(x_val_sample_channel_shuffled_0)
                    x_val_shuffled_3.append(x_val_sample_channel_shuffled_3)
                    x_val_shuffled_6.append(x_val_sample_channel_shuffled_6)
                    x_val_shuffled_8.append(x_val_sample_channel_shuffled_8)
                    x_val_shuffled_10.append(x_val_sample_channel_shuffled_10)
                    
                x_val_shuffled_0 = torch.stack(x_val_shuffled_0)
                x_val_shuffled_3 = torch.stack(x_val_shuffled_3)
                x_val_shuffled_6 = torch.stack(x_val_shuffled_6)
                x_val_shuffled_8 = torch.stack(x_val_shuffled_8)
                x_val_shuffled_10 = torch.stack(x_val_shuffled_10)
                                
                out = torch.nn.functional.softmax(model_1(x_val_shuffled_0.to(device)), dim = 1)
                probs_1.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
                labels_1.append(y_val)
                
                out = torch.nn.functional.softmax(model_2(x_val_shuffled_3.to(device)), dim = 1)
                probs_2.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
                labels_2.append(y_val)
                
                out = torch.nn.functional.softmax(model_3(x_val_shuffled_6.to(device)), dim = 1)
                probs_3.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
                labels_3.append(y_val)
                
                out = torch.nn.functional.softmax(model_4(x_val_shuffled_8.to(device)), dim = 1)
                probs_4.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
                labels_4.append(y_val)
                
                out = torch.nn.functional.softmax(model_5(x_val_shuffled_10.to(device)), dim = 1)
                probs_5.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
                labels_5.append(y_val)
                
        # Convert batches to single numpy arrays    
        probs_1 = np.stack([p for l in probs_1 for p in l])
        labels_1 = np.array([t for l in labels_1 for t in l])
        
        probs_2 = np.stack([p for l in probs_2 for p in l])
        labels_2 = np.array([t for l in labels_2 for t in l])
        
        probs_3 = np.stack([p for l in probs_3 for p in l])
        labels_3 = np.array([t for l in labels_3 for t in l])
        
        probs_4 = np.stack([p for l in probs_4 for p in l])
        labels_4 = np.array([t for l in labels_4 for t in l])
        
        probs_5 = np.stack([p for l in probs_5 for p in l])
        labels_5 = np.array([t for l in labels_5 for t in l])
        
        # Extract top 5 predictions for each example
        n = 5
        
        top = np.argpartition(-probs_1, n, axis = 1)[:, :n]
        top_probs = probs_1[np.arange(probs_1.shape[0])[:, None], top]
        right1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels_1)
        all_simulation_validation_accuracy_2[simulation_counter, 1, 0] = right1 / float(len(labels_1))
        count5 = sum([labels_1[i] in row for i, row in enumerate(top)])
        all_simulation_validation_accuracy_2[simulation_counter, 1, 1] = count5 / float(len(labels_1))
        
        top = np.argpartition(-probs_2, n, axis = 1)[:, :n]
        top_probs = probs_2[np.arange(probs_2.shape[0])[:, None], top]
        right1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels_2)
        all_simulation_validation_accuracy_2[simulation_counter, 2, 0] = right1 / float(len(labels_2))
        count5 = sum([labels_2[i] in row for i, row in enumerate(top)])
        all_simulation_validation_accuracy_2[simulation_counter, 2, 1] = count5 / float(len(labels_2))
        
        top = np.argpartition(-probs_3, n, axis = 1)[:, :n]
        top_probs = probs_3[np.arange(probs_3.shape[0])[:, None], top]
        right1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels_3)
        all_simulation_validation_accuracy_2[simulation_counter, 3, 0] = right1 / float(len(labels_3))
        count5 = sum([labels_3[i] in row for i, row in enumerate(top)])
        all_simulation_validation_accuracy_2[simulation_counter, 3, 1] = count5 / float(len(labels_3))
        
        top = np.argpartition(-probs_4, n, axis = 1)[:, :n]
        top_probs = probs_4[np.arange(probs_4.shape[0])[:, None], top]
        right1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels_4)
        all_simulation_validation_accuracy_2[simulation_counter, 4, 0] = right1 / float(len(labels_4))
        count5 = sum([labels_4[i] in row for i, row in enumerate(top)])
        all_simulation_validation_accuracy_2[simulation_counter, 4, 1] = count5 / float(len(labels_4))
        
        top = np.argpartition(-probs_5, n, axis = 1)[:, :n]
        top_probs = probs_5[np.arange(probs_5.shape[0])[:, None], top]
        right1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels_5)
        all_simulation_validation_accuracy_2[simulation_counter, 5, 0] = right1 / float(len(labels_5))
        count5 = sum([labels_5[i] in row for i, row in enumerate(top)])
        all_simulation_validation_accuracy_2[simulation_counter, 5, 1] = count5 / float(len(labels_5))
                   
    ### Saving the main variables
   
    scipy.io.savemat(parent_folder + '/all_simulation_validation_accuracy_1.mat', mdict = {'all_simulation_validation_accuracy_1': all_simulation_validation_accuracy_1})
    scipy.io.savemat(parent_folder + '/all_simulation_dimensionality.mat', mdict = {'all_simulation_dimensionality': all_simulation_dimensionality})
    scipy.io.savemat(parent_folder + '/all_simulation_validation_accuracy_2.mat', mdict = {'all_simulation_validation_accuracy_2': all_simulation_validation_accuracy_2})
    scipy.io.savemat(parent_folder + '/all_simulation_compositionality.mat', mdict = {'all_simulation_compositionality': all_simulation_compositionality})
    
    # Dimensionality and compositionality across convolutional layers of AlexNet
    
    fig, axs = plt.subplots(1, 3, figsize = (1 * 8, 3 * 8))
    fig.suptitle('Dimensionality and Compositionality across Convolutional Layers of AlexNet', fontsize = 20)
    
    ax = axs[0]
   
    ax.set_title('Intrinsic Dimension', fontsize = 12)
    ax.set_ylabel('ID')
    
    ax.plot(range(0, 6), np.nanmean(all_simulation_dimensionality, axis = 0) , '-b')
    ax.fill_between(range(0, 6), np.nanmean(all_simulation_dimensionality, axis = 0) - np.nanstd(all_simulation_dimensionality, axis = 0) / number_simulation ** 0.5, np.nanmean(all_simulation_dimensionality, axis = 0) + np.nanstd(all_simulation_dimensionality, axis = 0) / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
    
    ax.legend(loc = 'upper right', fontsize = 'medium')
    ax.set_ylim((0, 20))
    ax.set_xticks(range(0, 6))
    ax.set_xticklabels(['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
    
    ax = axs[1]
   
    ax.set_title('Compositionality: Measure 1', fontsize = 12)
    ax.set_ylabel('Compositionality')
    
    ax.plot(range(0, 6), np.nanmean(all_simulation_compositionality, axis = 0) , '-b')
    ax.fill_between(range(0, 6), np.nanmean(all_simulation_compositionality, axis = 0) - np.nanstd(all_simulation_compositionality, axis = 0) / number_simulation ** 0.5, np.nanmean(all_simulation_compositionality, axis = 0) + np.nanstd(all_simulation_compositionality, axis = 0) / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
    
    ax.legend(loc = 'upper right', fontsize = 'medium')
    ax.set_ylim((0, 1))
    ax.set_xticks(range(0, 6))
    ax.set_xticklabels(['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
    
    ax = axs[2]
   
    ax.set_title('Compositionality: Measure 2', fontsize = 12)
    ax.set_ylabel('Compositionality')
    
    ax.plot(range(0, 6), np.nanmean(all_simulation_validation_accuracy_1, axis = 0)[0] - np.nanmean(all_simulation_validation_accuracy_2, axis = 0)[:, 0] , '-b', label = 'Top 1 Accuracy')
    ax.fill_between(range(0, 6), np.nanmean(all_simulation_validation_accuracy_1, axis = 0)[0] - np.nanmean(all_simulation_validation_accuracy_2, axis = 0)[:, 0] - np.nanstd(all_simulation_validation_accuracy_2, axis = 0)[:, 0] / number_simulation ** 0.5, np.nanmean(all_simulation_validation_accuracy_1, axis = 0)[0] - np.nanmean(all_simulation_validation_accuracy_2, axis = 0)[:, 0] + np.nanstd(all_simulation_validation_accuracy_2, axis = 0)[:, 0] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
    
    ax.plot(range(0, 6), np.nanmean(all_simulation_validation_accuracy_1, axis = 0)[1] - np.nanmean(all_simulation_validation_accuracy_2, axis = 0)[:, 1] , '-r', label = 'Top 5 Accuracy')
    ax.fill_between(range(0, 6), np.nanmean(all_simulation_validation_accuracy_1, axis = 0)[1] - np.nanmean(all_simulation_validation_accuracy_2, axis = 0)[:, 1] - np.nanstd(all_simulation_validation_accuracy_2, axis = 0)[:, 1] / number_simulation ** 0.5, np.nanmean(all_simulation_validation_accuracy_1, axis = 0)[1] - np.nanmean(all_simulation_validation_accuracy_2, axis = 0)[:, 1] + np.nanstd(all_simulation_validation_accuracy_2, axis = 0)[:, 1] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
    
    ax.legend(loc = 'upper right', fontsize = 'medium')
    ax.set_ylim((0, 100))
    ax.set_xticks(range(0, 6))
    ax.set_xticklabels(['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
                           
    fig.savefig(parent_folder + '/Dimensionality and Compositionality across Convolutional Layers of AlexNet.png')
  
if __name__ == '__main__':
    main()
