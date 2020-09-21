"""
Created on Thu Jul  9 11:29:28 2020

@author: satarydizaji
"""

import os
import copy
import gc
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import random
import scipy.io
from sklearn.decomposition import PCA
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from intrinsic_dimension import estimate
from scipy.spatial.distance import pdist, squareform

from pwcca import compute_pwcca
from cka import feature_space_linear_cka

# Initialize the weights of the convolutional layers of AlexNet
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}
pretrained_dict = load_state_dict_from_url(model_urls['alexnet'])

# The DNN model for VPL
class DNNforVPL(nn.Module):
    
    def __init__(self, num_classes = 2):
        
        super(DNNforVPL, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(64, 192, kernel_size = 5, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
             nn.Linear(256 * 6 * 6, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
def main():
    
    ### Initializing the main variables
    
    num_sample_artiphysiology = 1000
    x_sample_artiphysiology_index = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
    
    for i in range(0, num_sample_artiphysiology):
        x_sample_artiphysiology_index[i, 0] = random.randrange(1)
        x_sample_artiphysiology_index[i, 1] = random.randrange(20)
        x_sample_artiphysiology_index[i, 2] = random.randrange(180)
    
    best_acc1 = 0
    
    number_simulation = 3
    number_group = 4
    number_layer = 5
    number_layer_freeze = 6
    number_transfer_stimuli = 21
    
    all_simulation_training_accuracy = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_transfer_accuracy = np.zeros((number_simulation, number_group, number_layer_freeze, 10), dtype = np.float32)
    all_simulation_specificity_index = np.zeros((number_simulation, number_group, number_layer_freeze), dtype = np.float32)
    all_simulation_all_ID = np.zeros((number_simulation, number_group, number_layer, number_layer_freeze), dtype = np.float32)
    all_simulation_all_PWCCA = np.zeros((int(number_simulation * (number_simulation - 1) / 2), number_group, number_layer, number_layer_freeze), dtype = np.float32)
    all_simulation_all_CKA = np.zeros((int(number_simulation * (number_simulation - 1) / 2), number_group, number_layer, number_layer_freeze), dtype = np.float32)
    
    all_simulation_unit_activity_layer_1 = np.zeros((number_simulation, number_group, number_layer_freeze, number_transfer_stimuli, 64), dtype = np.float32)
    all_simulation_unit_activity_layer_2 = np.zeros((number_simulation, number_group, number_layer_freeze, number_transfer_stimuli, 192), dtype = np.float32)
    all_simulation_unit_activity_layer_3 = np.zeros((number_simulation, number_group, number_layer_freeze, number_transfer_stimuli, 384), dtype = np.float32)
    all_simulation_unit_activity_layer_4 = np.zeros((number_simulation, number_group, number_layer_freeze, number_transfer_stimuli, 256), dtype = np.float32)
    all_simulation_unit_activity_layer_5 = np.zeros((number_simulation, number_group, number_layer_freeze, number_transfer_stimuli, 256), dtype = np.float32)
    
    all_PCA_explained_variance_layer_1 = np.zeros((number_simulation, number_group, number_layer_freeze, 64), dtype = np.float32)
    all_PCA_explained_variance_layer_2 = np.zeros((number_simulation, number_group, number_layer_freeze, 192), dtype = np.float32)
    all_PCA_explained_variance_layer_3 = np.zeros((number_simulation, number_group, number_layer_freeze, 384), dtype = np.float32)
    all_PCA_explained_variance_layer_4 = np.zeros((number_simulation, number_group, number_layer_freeze, 256), dtype = np.float32)
    all_PCA_explained_variance_layer_5 = np.zeros((number_simulation, number_group, number_layer_freeze, 256), dtype = np.float32)
    
    all_simulation_weight_change_layer_1 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_weight_change_layer_2 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_weight_change_layer_3 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_weight_change_layer_4 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_weight_change_layer_5 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    
    all_simulation_layer_rotation_layer_1 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_layer_rotation_layer_2 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_layer_rotation_layer_3 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_layer_rotation_layer_4 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
    all_simulation_layer_rotation_layer_5 = np.zeros((number_simulation, number_group, number_layer_freeze, 180), dtype = np.float32)
        
    # Cosine distance definition
    CosSim = nn.CosineSimilarity(dim = 0, eps = 1e-10)
       
    parent_folder = 'New_Results_AlexNet_1Str_ArtPhy_FL_DR_LR_ID_PWCCA_CKA'
        
    os.mkdir(parent_folder)
    
    for simulation_counter in range(number_simulation):
        print('Simulation:   ', simulation_counter + 1)
        
        os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1))
                    
        group_counter = -1
        
        for group_training in ['group1', 'group2', 'group3', 'group4']:
            gc.collect()
            group_counter = group_counter + 1
            
            print('Group:   ', group_training)
            
            os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training)
            
            ### Training
            
            # The structure of image names in different groups                    
            if group_training == 'group1':
                SF_training = [170]
                Ori_training = [23325, 23350, 23375, 23400, 23425, 23450, 23475, 23500, 23525, 23550,
                                23650, 23675, 23700, 23725, 23750, 23775, 23800, 23825, 23850, 23875]
            
            elif group_training == 'group2':
                SF_training = [53, 170, 276]
                Ori_training = [23325, 23350, 23375, 23400, 23425, 23450, 23475, 23500, 23525, 23550,
                                23650, 23675, 23700, 23725, 23750, 23775, 23800, 23825, 23850, 23875]
                
            elif group_training == 'group3':
                SF_training = [170]
                Ori_training = [23075, 23100, 23125, 23150, 23175, 23200, 23225, 23250, 23275, 23300,
                                23900, 23925, 23950, 23975, 24000, 24025, 24050, 24075, 24100, 24125]
                    
            elif group_training == 'group4':
                SF_training = [53, 170, 276]
                Ori_training = [23075, 23100, 23125, 23150, 23175, 23200, 23225, 23250, 23275, 23300,
                                23900, 23925, 23950, 23975, 24000, 24025, 24050, 24075, 24100, 24125]
                
            # Reading all images                    
            if group_training == 'group1' or group_training == 'group2':
                file_name_paths = glob.glob('VPL Stimuli/Learning & Transfer_SF (32)/learning_group1&2/*.TIFF')
            elif group_training == 'group3' or group_training == 'group4':
                file_name_paths = glob.glob('VPL Stimuli/Learning & Transfer_SF (32)/learning_group3&4/*.TIFF')
            
            file_names = [os.path.basename(x) for x in file_name_paths]
            
            # Define the main variables
            x_val_training = np.zeros((len(SF_training) * len(Ori_training) * 180, 224, 224, 3), dtype = np.float32)
            y_val_training = np.zeros((len(SF_training) * len(Ori_training) * 180, 1), dtype = np.int64)
            z_val_training = np.zeros((len(SF_training), len(Ori_training), 180), dtype = np.int64)
            
            x_tensor_training = []
            y_tensor_training = []
            
            counter = -1
            
            for i in range(len(file_names)):                 
                # Construct the main descriptive variables
                name_digits = file_names[i].split('_')
                
                flag_image_name = False
                
                for j in range(len(SF_training)):
                    for k in range(len(Ori_training)):
                        SFplusOri = str(SF_training[j]) + str(Ori_training[k])
                        
                        if (SFplusOri) in name_digits[0]:
                            Phase = int(name_digits[0].replace(SFplusOri,''))
                            
                            if Phase % 2 == 1:
                                counter = counter + 1
                                flag_image_name = True
                                
                                if k <= int(len(Ori_training) / 2 - 1):
                                    y_val_training[counter] = 0
                                else:
                                    y_val_training[counter] = 1
                                    
                                z_val_training[j][k][((Phase + 1) // 2) - 1] = counter
                
                if flag_image_name:                      
                    # Load image
                    img = Image.open(file_name_paths[i]).convert('RGB')
                    
                    # Resize image
                    width, height = img.size
                    new_width = width * 256 // min(img.size)
                    new_height = height * 256 // min(img.size)
                    img = img.resize((new_width, new_height), Image.BILINEAR)
                    
                    # Center crop image
                    width, height = img.size
                    startx = width // 2 - (224 // 2)
                    starty = height // 2 - (224 // 2)
                    img = np.asarray(img).reshape(height, width, 3)
                    img = img[starty:starty + 224, startx:startx + 224]
                    assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)
                    
                    # Save image
                    x_val_training[counter, :, :, :] = img[:, :, :]
                    
                    # Convert image to tensor and normalize
                    x_temp = torch.from_numpy(np.transpose(x_val_training[counter, :, :, :], (2, 0, 1)))
                    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    x_tensor_training.append(normalize(x_temp))
                    
                    # Convert target to tensor
                    y_tensor_training.append(torch.from_numpy(y_val_training[counter]))
                
            x_tensor_training = torch.stack(x_tensor_training)
            y_tensor_training = torch.stack(y_tensor_training)
            print(x_tensor_training.shape, y_tensor_training.shape)
            
            # Save main variables
            np.save('x_val_training_' + group_training + '.npy', x_val_training)
            np.save('y_val_training_' + group_training + '.npy', y_val_training)
            np.save('z_val_training_' + group_training + '.npy', z_val_training)
            
            # Get five sample tensors of training/validation images and show them
            indices = torch.tensor(np.random.permutation(len(SF_training) * len(Ori_training) * 180), dtype = torch.long)
            x_sample = torch.index_select(x_tensor_training, 0, indices[:5])
            y_sample = torch.index_select(y_tensor_training, 0, indices[:5])
            x_sample = make_grid([x_sample[0], x_sample[1], x_sample[2], x_sample[3], x_sample[4]])
            y_sample = [str(y_sample[0].item()), str(y_sample[1].item()), str(y_sample[2].item()), str(y_sample[3].item()), str(y_sample[4].item())]
            imshow(x_sample, y_sample)
            
            ### SF Transfer
            
            # The structure of image names in different groups
            if group_training == 'group1':
                group_transfer = 'group1'
                SF_transfer = [96]
                Ori_transfer = [23325, 23350, 23375, 23400, 23425, 23450, 23475, 23500, 23525, 23550,
                                23650, 23675, 23700, 23725, 23750, 23775, 23800, 23825, 23850, 23875]
            
            elif group_training == 'group2':
                group_transfer = 'group2'
                SF_transfer= [96]
                Ori_transfer = [23325, 23350, 23375, 23400, 23425, 23450, 23475, 23500, 23525, 23550,
                                23650, 23675, 23700, 23725, 23750, 23775, 23800, 23825, 23850, 23875]
                
            elif group_training == 'group3':
                group_transfer = 'group3'
                SF_transfer = [96]
                Ori_transfer = [23075, 23100, 23125, 23150, 23175, 23200, 23225, 23250, 23275, 23300,
                                23900, 23925, 23950, 23975, 24000, 24025, 24050, 24075, 24100, 24125]
                    
            elif group_training == 'group4':
                group_transfer = 'group4'
                SF_transfer = [96]
                Ori_transfer = [23075, 23100, 23125, 23150, 23175, 23200, 23225, 23250, 23275, 23300,
                                23900, 23925, 23950, 23975, 24000, 24025, 24050, 24075, 24100, 24125]
            
            # Reading all images                   
            if group_transfer == 'group1' or group_transfer == 'group2':
                file_name_paths = glob.glob('VPL Stimuli/Learning & Transfer_SF (32)/transfer_SF_group1&2/*.TIFF')
            elif group_transfer == 'group3' or group_transfer == 'group4':
                file_name_paths = glob.glob('VPL Stimuli/Learning & Transfer_SF (32)/transfer_SF_group3&4/*.TIFF')
            
            file_names = [os.path.basename(x) for x in file_name_paths]
            
            # Define the main variables
            x_val_transfer = np.zeros((len(SF_transfer) * len(Ori_transfer) * 180, 224, 224, 3), dtype = np.float32)
            y_val_transfer = np.zeros((len(SF_transfer) * len(Ori_transfer) * 180, 1), dtype = np.int64)
            z_val_transfer = np.zeros((len(SF_transfer), len(Ori_transfer), 180), dtype = np.int64)
            
            x_tensor_transfer = []
            y_tensor_transfer = []
            
            counter = -1
            
            for i in range(len(file_names)):                
                # Construct the main descriptive variables
                name_digits = file_names[i].split('_')
                
                flag_image_name = False
                
                for j in range(len(SF_transfer)):
                    for k in range(len(Ori_transfer)):
                        SFplusOri = str(SF_transfer[j]) + str(Ori_transfer[k])
                        if (SFplusOri) in name_digits[0]:
                            
                            Phase = int(name_digits[0].replace(SFplusOri,''))
                            
                            if Phase % 2 == 1:
                                counter = counter + 1
                                flag_image_name = True
                                
                                if k <= int(len(Ori_transfer) / 2 - 1):
                                    y_val_transfer[counter] = 0
                                else:
                                    y_val_transfer[counter] = 1
                                    
                                z_val_transfer[j][k][((Phase + 1) // 2) - 1] = counter
                
                if flag_image_name:                     
                    # Load image
                    img = Image.open(file_name_paths[i]).convert('RGB')
                    
                    # Resize image
                    width, height = img.size
                    new_width = width * 256 // min(img.size)
                    new_height = height * 256 // min(img.size)
                    img = img.resize((new_width, new_height), Image.BILINEAR)
                    
                    # Center crop image
                    width, height = img.size
                    startx = width // 2 - (224 // 2)
                    starty = height // 2 - (224 // 2)
                    img = np.asarray(img).reshape(height, width, 3)
                    img = img[starty:starty + 224, startx:startx + 224]
                    assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)
                    
                    # Save image
                    x_val_transfer[counter, :, :, :] = img[:, :, :]
                    
                    # Convert image to tensor and normalize
                    x_temp = torch.from_numpy(np.transpose(x_val_transfer[counter, :, :, :], (2, 0, 1)))
                    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    x_tensor_transfer.append(normalize(x_temp))
                    
                    # Convert target to tensor
                    y_tensor_transfer.append(torch.from_numpy(y_val_transfer[counter]))
                
            x_tensor_transfer = torch.stack(x_tensor_transfer)
            y_tensor_transfer = torch.stack(y_tensor_transfer)
            print(x_tensor_transfer.shape, y_tensor_transfer.shape)
            
            # Save main variables
            np.save('x_val_transfer_' + group_transfer + '.npy', x_val_transfer)
            np.save('y_val_transfer_' + group_transfer + '.npy', y_val_transfer)
            np.save('z_val_transfer_' + group_transfer + '.npy', z_val_transfer)
            
            # Get five sample tensors of training/validation images and show them
            indices = torch.tensor(np.random.permutation(len(SF_transfer) * len(Ori_transfer) * 180), dtype = torch.long)
            x_sample = torch.index_select(x_tensor_transfer, 0, indices[:5])
            y_sample = torch.index_select(y_tensor_transfer, 0, indices[:5])
            x_sample = make_grid([x_sample[0], x_sample[1], x_sample[2], x_sample[3], x_sample[4]])
            y_sample = [str(y_sample[0].item()), str(y_sample[1].item()), str(y_sample[2].item()), str(y_sample[3].item()), str(y_sample[4].item())]
            imshow(x_sample, y_sample)
            
            ### Tuning
                                      
            if group_training == 'group1' or group_training == 'group2':
                SF_tuning = [33, 53, 140, 170, 340, 480]
                Ori_tuning = [23350, 23450, 23550,
                              23650, 23750, 23850]
            
            elif group_training == 'group3' or group_training == 'group4':
                SF_tuning = [33, 53, 140, 170, 340, 480]
                Ori_tuning = [23100, 23200, 23300,
                              23900, 24000, 24100]
            
            # Define the main variables
            x_val_tuning = np.zeros((len(SF_tuning) * len(Ori_tuning) * 360, 224, 224, 3), dtype = np.float32)
            y_val_tuning = np.zeros((len(SF_tuning) * len(Ori_tuning) * 360, 1), dtype = np.int64)
            z_val_tuning = np.zeros((len(SF_tuning), len(Ori_tuning), 360), dtype = np.int64)
            
            x_tensor_tuning = []
            y_tensor_tuning = []
            
            counter = -1
            
            for p in range(0, 10):
                # Reading all images
                if group_training == 'group1' or group_training == 'group2':
                    file_name_paths = glob.glob('VPL Stimuli/6 x 40 x 360 Stimuli (32)/group1&2/p' + str(p + 1) + '/*.TIFF')
                elif group_training == 'group3' or group_training == 'group4':
                    file_name_paths = glob.glob('VPL Stimuli/6 x 40 x 360 Stimuli (32)/group3&4/p' + str(p + 1) + '/*.TIFF')
                
                file_names = [os.path.basename(x) for x in file_name_paths]
                
                for i in range(len(file_names)):                     
                    # Construct the main descriptive variables
                    name_digits = file_names[i].split('_')
                    
                    flag_image_name = False
                    
                    for j in range(len(SF_tuning)):
                        for k in range(len(Ori_tuning)):
                            SFplusOri = str(SF_tuning[j]) + str(Ori_tuning[k])
                            SFplusOri = SFplusOri.replace('.0', '')
                            
                            if (SFplusOri) in name_digits[0]:
                                Phase = int(name_digits[0].replace(SFplusOri, ''))
                                counter = counter + 1
                                flag_image_name = True
                                
                                if k <= int(len(Ori_tuning) / 2 - 1):
                                    y_val_tuning[counter] = 0
                                else:
                                    y_val_tuning[counter] = 1
                                    
                                z_val_tuning[j][k][Phase - 1] = counter
                                        
                    if flag_image_name:                          
                        # Load image
                        img = Image.open(file_name_paths[i]).convert('RGB')
                        
                        # Resize image
                        width, height = img.size
                        new_width = width * 256 // min(img.size)
                        new_height = height * 256 // min(img.size)
                        img = img.resize((new_width, new_height), Image.BILINEAR)
                        
                        # Center crop image
                        width, height = img.size
                        startx = width // 2 - (224 // 2)
                        starty = height // 2 - (224 // 2)
                        img = np.asarray(img).reshape(height, width, 3)
                        img = img[starty:starty + 224, startx:startx + 224]
                        assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)
                        
                        # Save image
                        x_val_tuning[counter, :, :, :] = img[:, :, :]
                        
                        # Convert image to tensor and normalize
                        x_temp = torch.from_numpy(np.transpose(x_val_tuning[counter, :, :, :], (2, 0, 1)))
                        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                        x_tensor_tuning.append(normalize(x_temp))
                        
                        # Convert target to tensor
                        y_tensor_tuning.append(torch.from_numpy(y_val_tuning[counter]))
                
            x_tensor_tuning = torch.stack(x_tensor_tuning)
            y_tensor_tuning = torch.stack(y_tensor_tuning)
            print(x_tensor_tuning.shape, y_tensor_tuning.shape)
                
            layer_freeze_counter = -1
            
            for layer_freeze in [None, 0, 3, 6, 8, 10]:
                layer_freeze_counter = layer_freeze_counter + 1
                
                print('Freezed Layer:   ', layer_freeze)
                
                # Select GPU
                global device
                gpu = 0
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                print("Use GPU: {} for training".format(gpu))
                
                # Load the PyTorch model
                model = DNNforVPL()
                
                model_dict = model.state_dict()
                
                # Filter out unnecessary keys
                pretrained_dict_model = {k : v for k, v in pretrained_dict.items() if k in model_dict}
                # Overwrite entries in the existing state dict
                model_dict.update(pretrained_dict_model)
                # Load the new state dict
                model.load_state_dict(model_dict)
                
                # Initialize the weights of the fully-connected layer of the model
                nn.init.zeros_(model.classifier[0].weight)
                nn.init.zeros_(model.classifier[0].bias)
                
                # Set all the parameters of the model to be trained
                for param in model.parameters():
                    param.requires_grad = True
                    
                if layer_freeze != None:
                    model.features[layer_freeze].weight.requires_grad = False
                    model.features[layer_freeze].bias.requires_grad = False
                
                # Send the model to GPU/CPU
                model = model.to(device)
                
                # Model summary
                print(model)
                    
                cudnn.benchmark = True
                
                ### ’Artiphysiology’ reveals V4-like shape tuning in a deep network trained for image classification
                
                # The convolutional layers: (0, 3, 6, 8, 10)
                # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # The central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                
                # The target stimuli                
                if layer_freeze == None:
                    os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/before_training')
                    saving_folder = parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/before_training'
                    
                    feature_sample_artiphysiology = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
                    
                    all_unit_activity_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64, 55, 55), dtype = np.float32)
                    all_unit_activity_Conv2d_2 = np.zeros((num_sample_artiphysiology, 192, 27, 27), dtype = np.float32)
                    all_unit_activity_Conv2d_3 = np.zeros((num_sample_artiphysiology, 384, 13, 13), dtype = np.float32)
                    all_unit_activity_Conv2d_4 = np.zeros((num_sample_artiphysiology, 256, 13, 13), dtype = np.float32)
                    all_unit_activity_Conv2d_5 = np.zeros((num_sample_artiphysiology, 256, 13, 13), dtype = np.float32)
                            
                    for i in range(num_sample_artiphysiology):                    
                        feature_sample_artiphysiology[i, :] = [SF_transfer[x_sample_artiphysiology_index[i, 0]], Ori_transfer[x_sample_artiphysiology_index[i, 1]], x_sample_artiphysiology_index[i, 2]]
                        
                        index = torch.tensor(z_val_transfer[x_sample_artiphysiology_index[i, 0], x_sample_artiphysiology_index[i, 1], x_sample_artiphysiology_index[i, 2]], dtype = torch.long)
                        x_sample = torch.index_select(x_tensor_transfer, 0, index)
                        x_sample = x_sample.cuda(gpu)
                        
                        unit_activity_layer_0 = model.features[0](x_sample)
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
                        
                        all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_5[i, :] = unit_activity_layer_10[0].detach().cpu().clone().numpy()
                        
                    scipy.io.savemat(saving_folder + '/feature_sample_artiphysiology.mat', mdict = {'feature_sample_artiphysiology': feature_sample_artiphysiology})
                    
                    scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_1.mat', mdict = {'all_unit_activity_Conv2d_1': all_unit_activity_Conv2d_1})
                    scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_2.mat', mdict = {'all_unit_activity_Conv2d_2': all_unit_activity_Conv2d_2})
                    scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_3.mat', mdict = {'all_unit_activity_Conv2d_3': all_unit_activity_Conv2d_3})
                    scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_4.mat', mdict = {'all_unit_activity_Conv2d_4': all_unit_activity_Conv2d_4})
                    scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_5.mat', mdict = {'all_unit_activity_Conv2d_5': all_unit_activity_Conv2d_5})
                                  
                    # # Boxplotting the tuning curves of central units of three features of convolutional layers
                    # SF_box_central_unit_activity_Conv2d_1 = []
                    # SF_box_central_unit_activity_Conv2d_2 = []
                    # SF_box_central_unit_activity_Conv2d_3 = []
                    # SF_box_central_unit_activity_Conv2d_4 = []
                    # SF_box_central_unit_activity_Conv2d_5 = []
                    
                    # for i in range(len(SF_transfer)):                        
                    #     SF_box_central_unit_activity_Conv2d_1.append(np.mean(all_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                    #     SF_box_central_unit_activity_Conv2d_2.append(np.mean(all_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                    #     SF_box_central_unit_activity_Conv2d_3.append(np.mean(all_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                    #     SF_box_central_unit_activity_Conv2d_4.append(np.mean(all_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                    #     SF_box_central_unit_activity_Conv2d_5.append(np.mean(all_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                        
                    # Ori_box_central_unit_activity_Conv2d_1 = []
                    # Ori_box_central_unit_activity_Conv2d_2 = []
                    # Ori_box_central_unit_activity_Conv2d_3 = []
                    # Ori_box_central_unit_activity_Conv2d_4 = []
                    # Ori_box_central_unit_activity_Conv2d_5 = []
                    
                    # for i in range(len(Ori_transfer)):
                    #     Ori_box_central_unit_activity_Conv2d_1.append(np.mean(all_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                    #     Ori_box_central_unit_activity_Conv2d_2.append(np.mean(all_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                    #     Ori_box_central_unit_activity_Conv2d_3.append(np.mean(all_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                    #     Ori_box_central_unit_activity_Conv2d_4.append(np.mean(all_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                    #     Ori_box_central_unit_activity_Conv2d_5.append(np.mean(all_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                    
                    # Phase_box_central_unit_activity_Conv2d_1 = []
                    # Phase_box_central_unit_activity_Conv2d_2 = []
                    # Phase_box_central_unit_activity_Conv2d_3 = []
                    # Phase_box_central_unit_activity_Conv2d_4 = []
                    # Phase_box_central_unit_activity_Conv2d_5 = []
                    
                    # for i in range(180):
                    #     Phase_box_central_unit_activity_Conv2d_1.append(np.mean(all_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                    #     Phase_box_central_unit_activity_Conv2d_2.append(np.mean(all_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                    #     Phase_box_central_unit_activity_Conv2d_3.append(np.mean(all_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                    #     Phase_box_central_unit_activity_Conv2d_4.append(np.mean(all_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                    #     Phase_box_central_unit_activity_Conv2d_5.append(np.mean(all_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                        
                    # for feature in ['SF', 'Ori', 'Phase']:
                    #     for conv_layer_num in [1, 2, 3, 4, 5]:
                    #         plt.figure()
                    #         plt.title("%s Boxplot Tuning Curve of the Convolutional Layer %d" % (feature, conv_layer_num))
                    #         plt.xlabel(feature)
                    #         plt.ylabel("Average Units Activity")
                    #         variable_name = feature + '_box_average_unit_activity_Conv2d_' + str(conv_layer_num)
                    #         plt.boxplot(vars()[variable_name])
                    #         plt.show()
                    #         plt.savefig(saving_folder + '/' + feature + ' Boxplot Tuning Curve of the Convolutional Layer ' + str(conv_layer_num) + '.tif')
                    #         plt.close()
                
                # Define the main learning parameters
                lr = 0.0001
                momentum = 0.9
                weight_decay = 0.0001
                
                # Define the loss function (criterion) and optimizer
                criterion = nn.CrossEntropyLoss().cuda(gpu)
                optimizer = torch.optim.SGD(model.parameters(), lr, momentum = momentum, weight_decay = weight_decay)
                   
                # Save the initial weights of the convolutional layers of the model
                Conv2d_1_0 = copy.deepcopy(model.features[0].weight)
                Conv2d_2_0 = copy.deepcopy(model.features[3].weight)
                Conv2d_3_0 = copy.deepcopy(model.features[6].weight)
                Conv2d_4_0 = copy.deepcopy(model.features[8].weight)
                Conv2d_5_0 = copy.deepcopy(model.features[10].weight)
                   
                # Define the main training/validation parameters
                start_session = 0
                sessions = 1
                
                z_val_shuffle = copy.deepcopy(z_val_training)
                    
                for i in range(len(SF_training)):
                    for j in range(len(Ori_training)):
                        random.shuffle(z_val_shuffle[i, j, :])
                    
                for session in range(start_session, sessions):                    
                    # Adjust the learning rate
                    adjust_learning_rate(optimizer, session, lr)
                    
                    # Train on a training set        
                    epochs = 180
                            
                    for epoch in range(epochs):                        
                        z_val_shuffle_1D = np.unique(z_val_shuffle[:, :, epoch])
                        indices = torch.tensor(z_val_shuffle_1D, dtype = torch.long)
                        x_train = torch.index_select(x_tensor_training, 0, indices)
                        y_train = torch.index_select(y_tensor_training, 0, indices)
                        y_train = y_train.squeeze(1)
                        
                        batch_time = AverageMeter('Time', ':6.3f')
                        losses = AverageMeter('Loss', ':.4e')
                        top1 = AverageMeter('Accuracy', ':6.2f')
                        progress = ProgressMeter(epochs, [batch_time, losses, top1], prefix=("Training >>> Session:   " + str(session) + "   Epoch: [{}]").format(epoch))
                    
                        # Switch to training mode
                        model.train()
                        
                        with torch.set_grad_enabled(True):
                            end = time.time()
                    
                            x_train = x_train.cuda(gpu)
                            y_train = y_train.cuda(gpu)
                    
                            # Compute output
                            output = model(x_train)
                            loss = criterion(output, y_train)
                    
                            # Measure accuracy and record loss
                            acc1 = accuracy(output, y_train, topk = 1)
                            losses.update(loss.item(), x_train.size(0))
                            top1.update(acc1[0], x_train.size(0))
                    
                            # Compute gradient and do SGD step
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    
                            # Save the validation accuracy for plotting
                            all_simulation_training_accuracy[simulation_counter, group_counter, layer_freeze_counter, epoch] = acc1[0].item()
                            
                            # Measure elapsed time
                            batch_time.update(time.time() - end)
                    
                            progress.display(epoch)
                            
                        # Remember the best accuracy
                        is_best = all_simulation_training_accuracy[simulation_counter, group_counter, layer_freeze_counter, epoch] >= best_acc1
                        best_acc1 = max(all_simulation_training_accuracy[simulation_counter, group_counter, layer_freeze_counter, epoch], best_acc1)
                        
                        all_simulation_weight_change_layer_1[simulation_counter, group_counter, layer_freeze_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[0].weight - Conv2d_1_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[0].weight, 2)), 0.5)).item()
                        all_simulation_weight_change_layer_2[simulation_counter, group_counter, layer_freeze_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[3].weight - Conv2d_2_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[3].weight, 2)), 0.5)).item()
                        all_simulation_weight_change_layer_3[simulation_counter, group_counter, layer_freeze_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[6].weight - Conv2d_3_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[6].weight, 2)), 0.5)).item()
                        all_simulation_weight_change_layer_4[simulation_counter, group_counter, layer_freeze_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[8].weight - Conv2d_4_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[8].weight, 2)), 0.5)).item()
                        all_simulation_weight_change_layer_5[simulation_counter, group_counter, layer_freeze_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[10].weight - Conv2d_5_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[10].weight, 2)), 0.5)).item()
                        
                        all_simulation_layer_rotation_layer_1[simulation_counter, group_counter, layer_freeze_counter, epoch] = 1 - CosSim(torch.flatten(model.features[0].weight), torch.flatten(Conv2d_1_0)).item()
                        all_simulation_layer_rotation_layer_2[simulation_counter, group_counter, layer_freeze_counter, epoch] = 1 - CosSim(torch.flatten(model.features[3].weight), torch.flatten(Conv2d_2_0)).item()
                        all_simulation_layer_rotation_layer_3[simulation_counter, group_counter, layer_freeze_counter, epoch] = 1 - CosSim(torch.flatten(model.features[6].weight), torch.flatten(Conv2d_3_0)).item()
                        all_simulation_layer_rotation_layer_4[simulation_counter, group_counter, layer_freeze_counter, epoch] = 1 - CosSim(torch.flatten(model.features[8].weight), torch.flatten(Conv2d_4_0)).item()
                        all_simulation_layer_rotation_layer_5[simulation_counter, group_counter, layer_freeze_counter, epoch] = 1 - CosSim(torch.flatten(model.features[10].weight), torch.flatten(Conv2d_5_0)).item()
                
                # Save the checkpoint
                save_checkpoint({
                    'session': session + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, group_training, 'DNNforVPL_' + group_training + '.pth.tar')
                
                # Select GPU
                gpu = 0
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                print("Use GPU: {} for transfer".format(gpu))
                
                # Set all the parameters of the model to be trained
                for param in model.parameters():
                    param.requires_grad = False
                
                # Send the model to GPU/CPU
                model = model.to(device)
                
                # Model summary
                print(model)
                
                cudnn.benchmark = True
                
                # Define the main training/validation parameters
                start_session = 0
                sessions = 10
                    
                for session in range(start_session, sessions):                    
                    z_val_shuffle = copy.deepcopy(z_val_transfer)
                    
                    for j in range(len(SF_transfer)):
                        for k in range(len(Ori_transfer)):
                            random.shuffle(z_val_shuffle[j, k, :])
                
                    # Evaluate on a validation set
                    z_val_shuffle_1D = np.unique(z_val_shuffle[:, :, session])
                    indices = torch.tensor(z_val_shuffle_1D, dtype = torch.long)
                    x_valid = torch.index_select(x_tensor_transfer, 0, indices)
                    y_valid = torch.index_select(y_tensor_transfer, 0, indices)
                    y_valid = y_valid.squeeze(1)
                                               
                    batch_time = AverageMeter('Time', ':6.3f')
                    losses = AverageMeter('Loss', ':.4e')
                    top1 = AverageMeter('Accuracy', ':6.2f')
                    progress = ProgressMeter(1, [batch_time, losses, top1], prefix=("Transfer >>> Session:   " + str(session) + "   Epoch: [{}]").format(1))
                
                    # Switch to evaluating mode
                    model.eval()
                
                    with torch.no_grad():
                        end = time.time()
                        
                        x_valid = x_valid.cuda(gpu)
                        y_valid = y_valid.cuda(gpu)
            
                        # Compute output
                        output = model(x_valid)
                        loss = criterion(output, y_valid)
            
                        # Measure accuracy and record loss
                        acc1 = accuracy(output, y_valid, topk = 1)
                        losses.update(loss.item(), x_valid.size(0))
                        top1.update(acc1[0], x_valid.size(0))
                        
                        # Save the validation accuracy for plotting
                        all_simulation_transfer_accuracy[simulation_counter, group_counter, layer_freeze_counter, session - start_session] = acc1[0].item()
            
                        # Measure elapsed time
                        batch_time.update(time.time() - end)
            
                        progress.display(1)
                        
                    # Remember the best accuracy and save checkpoint
                    is_best = all_simulation_transfer_accuracy[simulation_counter, group_counter, layer_freeze_counter, session - start_session] >= best_acc1
                    best_acc1 = max(all_simulation_transfer_accuracy[simulation_counter, group_counter, layer_freeze_counter, session - start_session], best_acc1)
                                              
                ### ’Artiphysiology’ reveals V4-like shape tuning in a deep network trained for image classification
                
                # The convolutional layers: (0, 3, 6, 8, 10)
                # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # The central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                
                # The target stimuli
                os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/after_training_' + str(layer_freeze))
                saving_folder = parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/after_training_' + str(layer_freeze)
                
                feature_sample_artiphysiology = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
                
                all_unit_activity_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64, 55, 55), dtype = np.float32)
                all_unit_activity_Conv2d_2 = np.zeros((num_sample_artiphysiology, 192, 27, 27), dtype = np.float32)
                all_unit_activity_Conv2d_3 = np.zeros((num_sample_artiphysiology, 384, 13, 13), dtype = np.float32)
                all_unit_activity_Conv2d_4 = np.zeros((num_sample_artiphysiology, 256, 13, 13), dtype = np.float32)
                all_unit_activity_Conv2d_5 = np.zeros((num_sample_artiphysiology, 256, 13, 13), dtype = np.float32)
                        
                for i in range(num_sample_artiphysiology):                    
                    feature_sample_artiphysiology[i, :] = [SF_transfer[x_sample_artiphysiology_index[i, 0]], Ori_transfer[x_sample_artiphysiology_index[i, 1]], x_sample_artiphysiology_index[i, 2]]
                    
                    index = torch.tensor(z_val_transfer[x_sample_artiphysiology_index[i, 0], x_sample_artiphysiology_index[i, 1], x_sample_artiphysiology_index[i, 2]], dtype = torch.long)
                    x_sample = torch.index_select(x_tensor_transfer, 0, index)
                    x_sample = x_sample.cuda(gpu)
                    
                    unit_activity_layer_0 = model.features[0](x_sample)
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
                    
                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_5[i, :] = unit_activity_layer_10[0].detach().cpu().clone().numpy()
                    
                scipy.io.savemat(saving_folder + '/feature_sample_artiphysiology.mat', mdict = {'feature_sample_artiphysiology': feature_sample_artiphysiology})
                
                scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_1.mat', mdict = {'all_unit_activity_Conv2d_1': all_unit_activity_Conv2d_1})
                scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_2.mat', mdict = {'all_unit_activity_Conv2d_2': all_unit_activity_Conv2d_2})
                scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_3.mat', mdict = {'all_unit_activity_Conv2d_3': all_unit_activity_Conv2d_3})
                scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_4.mat', mdict = {'all_unit_activity_Conv2d_4': all_unit_activity_Conv2d_4})
                scipy.io.savemat(saving_folder + '/all_unit_activity_Conv2d_5.mat', mdict = {'all_unit_activity_Conv2d_5': all_unit_activity_Conv2d_5})
                              
                # # Boxplotting the tuning curves of central units of three features of convolutional layers
                # SF_box_central_unit_activity_Conv2d_1 = []
                # SF_box_central_unit_activity_Conv2d_2 = []
                # SF_box_central_unit_activity_Conv2d_3 = []
                # SF_box_central_unit_activity_Conv2d_4 = []
                # SF_box_central_unit_activity_Conv2d_5 = []
                
                # for i in range(len(SF_transfer)):                        
                #     SF_box_central_unit_activity_Conv2d_1.append(np.mean(all_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                #     SF_box_central_unit_activity_Conv2d_2.append(np.mean(all_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                #     SF_box_central_unit_activity_Conv2d_3.append(np.mean(all_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                #     SF_box_central_unit_activity_Conv2d_4.append(np.mean(all_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                #     SF_box_central_unit_activity_Conv2d_5.append(np.mean(all_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 0] == SF_transfer[i], :], axis = (1, 2, 3)))
                    
                # Ori_box_central_unit_activity_Conv2d_1 = []
                # Ori_box_central_unit_activity_Conv2d_2 = []
                # Ori_box_central_unit_activity_Conv2d_3 = []
                # Ori_box_central_unit_activity_Conv2d_4 = []
                # Ori_box_central_unit_activity_Conv2d_5 = []
                
                # for i in range(len(Ori_transfer)):
                #     Ori_box_central_unit_activity_Conv2d_1.append(np.mean(all_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                #     Ori_box_central_unit_activity_Conv2d_2.append(np.mean(all_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                #     Ori_box_central_unit_activity_Conv2d_3.append(np.mean(all_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                #     Ori_box_central_unit_activity_Conv2d_4.append(np.mean(all_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                #     Ori_box_central_unit_activity_Conv2d_5.append(np.mean(all_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 1] == Ori_transfer[i], :], axis = (1, 2, 3)))
                
                # Phase_box_central_unit_activity_Conv2d_1 = []
                # Phase_box_central_unit_activity_Conv2d_2 = []
                # Phase_box_central_unit_activity_Conv2d_3 = []
                # Phase_box_central_unit_activity_Conv2d_4 = []
                # Phase_box_central_unit_activity_Conv2d_5 = []
                
                # for i in range(180):
                #     Phase_box_central_unit_activity_Conv2d_1.append(np.mean(all_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                #     Phase_box_central_unit_activity_Conv2d_2.append(np.mean(all_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                #     Phase_box_central_unit_activity_Conv2d_3.append(np.mean(all_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                #     Phase_box_central_unit_activity_Conv2d_4.append(np.mean(all_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                #     Phase_box_central_unit_activity_Conv2d_5.append(np.mean(all_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = (1, 2, 3)))
                    
                # for feature in ['SF', 'Ori', 'Phase']:
                #     for conv_layer_num in [1, 2, 3, 4, 5]:
                #         plt.figure()
                #         plt.title("%s Boxplot Tuning Curve of the Convolutional Layer %d" % (feature, conv_layer_num))
                #         plt.xlabel(feature)
                #         plt.ylabel("Average Units Activity")
                #         variable_name = feature + '_box_average_unit_activity_Conv2d_' + str(conv_layer_num)
                #         plt.boxplot(vars()[variable_name])
                #         plt.show()
                #         plt.savefig(saving_folder + '/' + feature + ' Boxplot Tuning Curve of the Convolutional Layer ' + str(conv_layer_num) + '.tif')
                #         plt.close()
                
                ### Saving all units activity for all transfer stimuli 
                
                all_unit_activity_analysis_layer_1 = np.zeros((number_transfer_stimuli, 64, 55, 55), dtype = np.float32)
                all_unit_activity_analysis_layer_2 = np.zeros((number_transfer_stimuli, 192, 27, 27), dtype = np.float32)
                all_unit_activity_analysis_layer_3 = np.zeros((number_transfer_stimuli, 384, 13, 13), dtype = np.float32)
                all_unit_activity_analysis_layer_4 = np.zeros((number_transfer_stimuli, 256, 13, 13), dtype = np.float32)
                all_unit_activity_analysis_layer_5 = np.zeros((number_transfer_stimuli, 256, 13, 13), dtype = np.float32)
                
                for j in range(len(SF_transfer)):
                    for k in range(len(Ori_transfer)):
                        indices = np.intersect1d(np.where(feature_sample_artiphysiology[:, 0] == SF_transfer[j]), np.where(feature_sample_artiphysiology[:, 1] == Ori_transfer[k]))
                          
                        all_unit_activity_analysis_layer_1[j * len(SF_transfer) + k, :] = np.mean(all_unit_activity_Conv2d_1[indices, :], axis = 0)
                        all_unit_activity_analysis_layer_2[j * len(SF_transfer) + k, :] = np.mean(all_unit_activity_Conv2d_2[indices, :], axis = 0)
                        all_unit_activity_analysis_layer_3[j * len(SF_transfer) + k, :] = np.mean(all_unit_activity_Conv2d_3[indices, :], axis = 0)
                        all_unit_activity_analysis_layer_4[j * len(SF_transfer) + k, :] = np.mean(all_unit_activity_Conv2d_4[indices, :], axis = 0)
                        all_unit_activity_analysis_layer_5[j * len(SF_transfer) + k, :] = np.mean(all_unit_activity_Conv2d_5[indices, :], axis = 0)
                
                all_simulation_unit_activity_layer_1[simulation_counter, group_counter, layer_freeze_counter, :, :] = all_unit_activity_analysis_layer_1.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 64)
                all_simulation_unit_activity_layer_2[simulation_counter, group_counter, layer_freeze_counter, :, :] = all_unit_activity_analysis_layer_2.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 192)
                all_simulation_unit_activity_layer_3[simulation_counter, group_counter, layer_freeze_counter, :, :] = all_unit_activity_analysis_layer_3.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 384)
                all_simulation_unit_activity_layer_4[simulation_counter, group_counter, layer_freeze_counter, :, :] = all_unit_activity_analysis_layer_4.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 256)
                all_simulation_unit_activity_layer_5[simulation_counter, group_counter, layer_freeze_counter, :, :] = all_unit_activity_analysis_layer_5.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 256)
                
                ### Calculating the variance explained by PCA                
                
                PCA_layer_1 = PCA(n_components = 64).fit(all_unit_activity_Conv2d_1.mean(axis = (2, 3)).reshape(num_sample_artiphysiology, 64))
                PCA_layer_2 = PCA(n_components = 192).fit(all_unit_activity_Conv2d_2.mean(axis = (2, 3)).reshape(num_sample_artiphysiology, 192))
                PCA_layer_3 = PCA(n_components = 384).fit(all_unit_activity_Conv2d_3.mean(axis = (2, 3)).reshape(num_sample_artiphysiology, 384))
                PCA_layer_4 = PCA(n_components = 256).fit(all_unit_activity_Conv2d_4.mean(axis = (2, 3)).reshape(num_sample_artiphysiology, 256))
                PCA_layer_5 = PCA(n_components = 256).fit(all_unit_activity_Conv2d_5.mean(axis = (2, 3)).reshape(num_sample_artiphysiology, 256))
                
                all_PCA_explained_variance_layer_1[simulation_counter, group_counter, layer_freeze_counter, :] = PCA_layer_1.explained_variance_ratio_
                all_PCA_explained_variance_layer_2[simulation_counter, group_counter, layer_freeze_counter, :] = PCA_layer_2.explained_variance_ratio_
                all_PCA_explained_variance_layer_3[simulation_counter, group_counter, layer_freeze_counter, :] = PCA_layer_3.explained_variance_ratio_
                all_PCA_explained_variance_layer_4[simulation_counter, group_counter, layer_freeze_counter, :] = PCA_layer_4.explained_variance_ratio_
                all_PCA_explained_variance_layer_5[simulation_counter, group_counter, layer_freeze_counter, :] = PCA_layer_5.explained_variance_ratio_
                
                ### Calculating the intrinsic dimension
                
                all_simulation_all_ID[simulation_counter, group_counter, 0, layer_freeze_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'))[2]
                all_simulation_all_ID[simulation_counter, group_counter, 1, layer_freeze_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'))[2]
                all_simulation_all_ID[simulation_counter, group_counter, 2, layer_freeze_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'))[2]
                all_simulation_all_ID[simulation_counter, group_counter, 3, layer_freeze_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1)), 'euclidean'))[2]
                all_simulation_all_ID[simulation_counter, group_counter, 4, layer_freeze_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_5.reshape(num_sample_artiphysiology, -1)), 'euclidean'))[2]
    
    ### Specificity Index
    
    all_simulation_specificity_index = (all_simulation_training_accuracy[:, : , :, 179] - all_simulation_transfer_accuracy.mean(3)) / (all_simulation_training_accuracy[:, : , :, 179] - all_simulation_training_accuracy[:, : , :, 0])
        
    ### Saving the main variables
   
    scipy.io.savemat(parent_folder + '/all_simulation_training_accuracy.mat', mdict = {'all_simulation_training_accuracy': all_simulation_training_accuracy})
    scipy.io.savemat(parent_folder + '/all_simulation_transfer_accuracy.mat', mdict = {'all_simulation_transfer_accuracy': all_simulation_transfer_accuracy})
    scipy.io.savemat(parent_folder + '/all_simulation_specificity_index.mat', mdict = {'all_simulation_specificity_index': all_simulation_specificity_index})
    scipy.io.savemat(parent_folder + '/all_simulation_all_ID.mat', mdict = {'all_simulation_all_ID': all_simulation_all_ID})
            
    scipy.io.savemat(parent_folder + '/all_simulation_unit_activity_layer_1.mat', mdict = {'all_simulation_unit_activity_layer_1': all_simulation_unit_activity_layer_1})
    scipy.io.savemat(parent_folder + '/all_simulation_unit_activity_layer_2.mat', mdict = {'all_simulation_unit_activity_layer_2': all_simulation_unit_activity_layer_2})
    scipy.io.savemat(parent_folder + '/all_simulation_unit_activity_layer_3.mat', mdict = {'all_simulation_unit_activity_layer_3': all_simulation_unit_activity_layer_3})
    scipy.io.savemat(parent_folder + '/all_simulation_unit_activity_layer_4.mat', mdict = {'all_simulation_unit_activity_layer_4': all_simulation_unit_activity_layer_4})
    scipy.io.savemat(parent_folder + '/all_simulation_unit_activity_layer_5.mat', mdict = {'all_simulation_unit_activity_layer_5': all_simulation_unit_activity_layer_5})
    
    scipy.io.savemat(parent_folder + '/all_PCA_explained_variance_layer_1.mat', mdict = {'all_PCA_explained_variance_layer_1': all_PCA_explained_variance_layer_1})
    scipy.io.savemat(parent_folder + '/all_PCA_explained_variance_layer_2.mat', mdict = {'all_PCA_explained_variance_layer_2': all_PCA_explained_variance_layer_2})
    scipy.io.savemat(parent_folder + '/all_PCA_explained_variance_layer_3.mat', mdict = {'all_PCA_explained_variance_layer_3': all_PCA_explained_variance_layer_3})
    scipy.io.savemat(parent_folder + '/all_PCA_explained_variance_layer_4.mat', mdict = {'all_PCA_explained_variance_layer_4': all_PCA_explained_variance_layer_4})
    scipy.io.savemat(parent_folder + '/all_PCA_explained_variance_layer_5.mat', mdict = {'all_PCA_explained_variance_layer_5': all_PCA_explained_variance_layer_5})
    
    scipy.io.savemat(parent_folder + '/all_simulation_weight_change_layer_1.mat', mdict = {'all_simulation_weight_change_layer_1': all_simulation_weight_change_layer_1})
    scipy.io.savemat(parent_folder + '/all_simulation_weight_change_layer_2.mat', mdict = {'all_simulation_weight_change_layer_2': all_simulation_weight_change_layer_2})
    scipy.io.savemat(parent_folder + '/all_simulation_weight_change_layer_3.mat', mdict = {'all_simulation_weight_change_layer_3': all_simulation_weight_change_layer_3})
    scipy.io.savemat(parent_folder + '/all_simulation_weight_change_layer_4.mat', mdict = {'all_simulation_weight_change_layer_4': all_simulation_weight_change_layer_4})
    scipy.io.savemat(parent_folder + '/all_simulation_weight_change_layer_5.mat', mdict = {'all_simulation_weight_change_layer_5': all_simulation_weight_change_layer_5})
    
    scipy.io.savemat(parent_folder + '/all_simulation_layer_rotation_layer_1.mat', mdict = {'all_simulation_layer_rotation_layer_1': all_simulation_layer_rotation_layer_1})
    scipy.io.savemat(parent_folder + '/all_simulation_layer_rotation_layer_2.mat', mdict = {'all_simulation_layer_rotation_layer_2': all_simulation_layer_rotation_layer_2})
    scipy.io.savemat(parent_folder + '/all_simulation_layer_rotation_layer_3.mat', mdict = {'all_simulation_layer_rotation_layer_3': all_simulation_layer_rotation_layer_3})
    scipy.io.savemat(parent_folder + '/all_simulation_layer_rotation_layer_4.mat', mdict = {'all_simulation_layer_rotation_layer_4': all_simulation_layer_rotation_layer_4})
    scipy.io.savemat(parent_folder + '/all_simulation_layer_rotation_layer_5.mat', mdict = {'all_simulation_layer_rotation_layer_5': all_simulation_layer_rotation_layer_5})
    
    ### Training Accuracy
    
    fig, axs = plt.subplots(2, 3, figsize = (2 * 8, 3 * 6))
    fig.suptitle('Training Accuracy', fontsize = 20)
    
    for i in range(number_layer_freeze):
        if i <= 2:
            ax = axs[0, i]
        elif i > 2:
            ax = axs[1, i - 3]
        
        ax.set_title('Freezed Layer = ' + str(i), fontsize = 12)
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
                    
        ax.plot(range(0, 180), all_simulation_training_accuracy.mean(0)[0, i], "-b", label = "Group 1")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy.mean(0)[0, i] - all_simulation_training_accuracy.std(0)[0, i] / number_simulation ** 0.5, all_simulation_training_accuracy.mean(0)[0, i] + all_simulation_training_accuracy.std(0)[0, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy.mean(0)[1, i], "-g", label = "Group 2")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy.mean(0)[1, i] - all_simulation_training_accuracy.std(0)[1, i] / number_simulation ** 0.5, all_simulation_training_accuracy.mean(0)[1, i] + all_simulation_training_accuracy.std(0)[1, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy.mean(0)[2, i], "-r", label = "Group 3")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy.mean(0)[2, i] - all_simulation_training_accuracy.std(0)[2, i] / number_simulation ** 0.5, all_simulation_training_accuracy.mean(0)[2, i] + all_simulation_training_accuracy.std(0)[2, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy.mean(0)[3, i], "-c", label = "Group 4")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy.mean(0)[3, i] - all_simulation_training_accuracy.std(0)[3, i] / number_simulation ** 0.5, all_simulation_training_accuracy.mean(0)[3, i] + all_simulation_training_accuracy.std(0)[3, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                   
        ax.legend(loc = 'lower right', fontsize = 'medium')
        ax.set_ylim((0, 105))
        ax.set_xticks(np.arange(0, 180, 30.0))
        
    fig.savefig(parent_folder + '/Training Accuracy.tif')
    
    ### Transfer Accuracy
    
    fig, axs = plt.subplots(2, 3, figsize = (2 * 8, 3 * 6))
    fig.suptitle('Transfer Accuracy', fontsize = 20)
    
    for i in range(number_layer_freeze):
        if i <= 2:
            ax = axs[0, i]
        elif i > 2:
            ax = axs[1, i - 3]
        
        ax.set_title('Freezed Layer = ' + str(i), fontsize = 12)
        ax.set_ylabel('accuracy')
        
        bar_list = ax.bar(range(0, 4), [all_simulation_transfer_accuracy.mean(axis = (0, 3))[0, i], 
                                        all_simulation_transfer_accuracy.mean(axis = (0, 3))[1, i], 
                                        all_simulation_transfer_accuracy.mean(axis = (0, 3))[2, i], 
                                        all_simulation_transfer_accuracy.mean(axis = (0, 3))[3, i]],
                          yerr = [all_simulation_transfer_accuracy.std(axis = (0, 3))[0, i],
                                  all_simulation_transfer_accuracy.std(axis = (0, 3))[1, i],
                                  all_simulation_transfer_accuracy.std(axis = (0, 3))[2, i],
                                  all_simulation_transfer_accuracy.std(axis = (0, 3))[3, i]])
        
        bar_list[0].set_color('b')
        bar_list[1].set_color('g')
        bar_list[2].set_color('r')
        bar_list[3].set_color('c')
        
        ax.set_ylim((0, 105))
        ax.set_xticks(range(0, 4))
        ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3', 'Group 4'])
                
    fig.savefig(parent_folder + '/Transfer Accuracy.tif')
    
    ### Specificity Index
    
    fig, axs = plt.subplots(2, 3, figsize = (2 * 8, 3 * 6))
    fig.suptitle('Specificity Index', fontsize = 20)
    
    for i in range(number_layer_freeze):
        if i <= 2:
            ax = axs[0, i]
        elif i > 2:
            ax = axs[1, i - 3]
        
        ax.set_title('Freezed Layer = ' + str(i), fontsize = 12)
        ax.set_ylabel('index')
        
        bar_list = ax.bar(range(0, 4), [np.nanmean(all_simulation_specificity_index, axis = 0)[0, i], 
                                        np.nanmean(all_simulation_specificity_index, axis = 0)[1, i], 
                                        np.nanmean(all_simulation_specificity_index, axis = 0)[2, i], 
                                        np.nanmean(all_simulation_specificity_index, axis = 0)[3, i]],
                          yerr = [np.nanstd(all_simulation_specificity_index, axis = 0)[0, i],
                                  np.nanstd(all_simulation_specificity_index, axis = 0)[1, i],
                                  np.nanstd(all_simulation_specificity_index, axis = 0)[2, i],
                                  np.nanstd(all_simulation_specificity_index, axis = 0)[3, i]])
        
        bar_list[0].set_color('b')
        bar_list[1].set_color('g')
        bar_list[2].set_color('r')
        bar_list[3].set_color('c')
        
        ax.set_ylim((-0.2, 1.2))
        ax.set_xticks(range(0, 4))
        ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3', 'Group 4'])
                
    fig.savefig(parent_folder + '/Specificity Index.tif')
    
    ### Dimensionality reduction with PCA   
    
    resp_dict = {}
    
    for i in range(number_layer_freeze):
        for j in range(number_group):
            label = str(j + 1) + '1'
            resp_dict[label] = all_simulation_unit_activity_layer_1.mean(0)[j, i, :, :]
            
            label = str(j + 1) + '2'
            resp_dict[label] = all_simulation_unit_activity_layer_2.mean(0)[j, i, :, :]
            
            label = str(j + 1) + '3'
            resp_dict[label] = all_simulation_unit_activity_layer_3.mean(0)[j, i, :, :]
            
            label = str(j + 1) + '4'
            resp_dict[label] = all_simulation_unit_activity_layer_4.mean(0)[j, i, :, :]
            
            label = str(j + 1) + '5'
            resp_dict[label] = all_simulation_unit_activity_layer_5.mean(0)[j, i, :, :]
    
        plot_resp_lowd(resp_dict, i, number_group, 5, parent_folder)
    
    ### Variance Explained by PCA
    
    fig, axs = plt.subplots(number_layer, number_layer_freeze, figsize = (number_layer * 4, number_layer_freeze * 3.5))
    fig.suptitle('Variance Explained by PCA', fontsize = 20)
    
    for i in range(number_layer):
        for j in range(number_layer_freeze):
            ax = axs[i, j]
            
            if i == 0:
                ax.set_title('Freezed Layer = ' + str(j))
                n_pca_component = 64
            elif i == 1:
                n_pca_component = 192
            elif i == 2:
                n_pca_component = 384
            elif i == 3:
                n_pca_component = 256
            elif i == 4:
                ax.set_xlabel('components')
                n_pca_component = 256
            if j == 0:
                ax.set_ylabel('Layer ' + str(i + 1))
                
            variable_name = 'all_PCA_explained_variance_layer_' + str(i + 1)    
            PCA_explained_variance = vars()[variable_name]
                        
            ax.plot(range(0, n_pca_component), PCA_explained_variance.mean(0)[0, j], "-b", label = "Group 1")
            ax.fill_between(range(0, n_pca_component), PCA_explained_variance.mean(0)[0, j] - PCA_explained_variance.std(0)[0, j] / number_simulation ** 0.5, PCA_explained_variance.mean(0)[0, j] + PCA_explained_variance.std(0)[0, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
            
            ax.plot(range(0, n_pca_component), PCA_explained_variance.mean(0)[1, j], "-g", label = "Group 2")
            ax.fill_between(range(0, n_pca_component), PCA_explained_variance.mean(0)[1, j] - PCA_explained_variance.std(0)[1, j] / number_simulation ** 0.5, PCA_explained_variance.mean(0)[1, j] + PCA_explained_variance.std(0)[1, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
                        
            ax.plot(range(0, n_pca_component), PCA_explained_variance.mean(0)[2, j], "-r", label = "Group 3")
            ax.fill_between(range(0, n_pca_component), PCA_explained_variance.mean(0)[2, j] - PCA_explained_variance.std(0)[2, j] / number_simulation ** 0.5, PCA_explained_variance.mean(0)[2, j] + PCA_explained_variance.std(0)[2, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
            
            ax.plot(range(0, n_pca_component), PCA_explained_variance.mean(0)[3, j], "-c", label = "Group 4")
            ax.fill_between(range(0, n_pca_component), PCA_explained_variance.mean(0)[3, j] - PCA_explained_variance.std(0)[3, j] / number_simulation ** 0.5, PCA_explained_variance.mean(0)[3, j] + PCA_explained_variance.std(0)[3, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                                                
            ax.legend(loc = 'lower right', fontsize = 'x-small')
            ax.set_ylim((0, 1))
            
    fig.savefig(parent_folder + '/Variance Explained by PCA.tif')
    
    ### Weight Change
    
    fig, axs = plt.subplots(number_group, number_layer_freeze, figsize = (number_group * 4, number_layer_freeze * 3.5))
    fig.suptitle('Weight Change', fontsize = 20)
    
    for i in range(number_group):
        for j in range(number_layer_freeze):
            ax = axs[i, j]
            
            if i == 0:
                ax.set_title('Freezed Layer = ' + str(j))
            if i == 3:
                ax.set_xlabel('epoch')
            if j == 0:
                ax.set_ylabel('Group ' + str(i + 1))
                        
            ax.plot(range(0, 180), all_simulation_weight_change_layer_1.mean(0)[i, j], "-b", label = "Conv Layer 1")
            ax.fill_between(range(0, 180), all_simulation_weight_change_layer_1.mean(0)[i, j] - all_simulation_weight_change_layer_1.std(0)[i, j] / number_simulation ** 0.5, all_simulation_weight_change_layer_1.mean(0)[i, j] + all_simulation_weight_change_layer_1.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
            
            ax.plot(range(0, 180), all_simulation_weight_change_layer_2.mean(0)[i, j], "-g", label = "Conv Layer 2")
            ax.fill_between(range(0, 180), all_simulation_weight_change_layer_2.mean(0)[i, j] - all_simulation_weight_change_layer_2.std(0)[i, j] / number_simulation ** 0.5, all_simulation_weight_change_layer_2.mean(0)[i, j] + all_simulation_weight_change_layer_2.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
                        
            ax.plot(range(0, 180), all_simulation_weight_change_layer_3.mean(0)[i, j], "-r", label = "Conv Layer 3")
            ax.fill_between(range(0, 180), all_simulation_weight_change_layer_3.mean(0)[i, j] - all_simulation_weight_change_layer_3.std(0)[i, j] / number_simulation ** 0.5, all_simulation_weight_change_layer_3.mean(0)[i, j] + all_simulation_weight_change_layer_3.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
            
            ax.plot(range(0, 180), all_simulation_weight_change_layer_4.mean(0)[i, j], "-c", label = "Conv Layer 4")
            ax.fill_between(range(0, 180), all_simulation_weight_change_layer_4.mean(0)[i, j] - all_simulation_weight_change_layer_4.std(0)[i, j] / number_simulation ** 0.5, all_simulation_weight_change_layer_4.mean(0)[i, j] + all_simulation_weight_change_layer_4.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                        
            ax.plot(range(0, 180), all_simulation_weight_change_layer_5.mean(0)[i, j], "-m", label = "Conv Layer 5")
            ax.fill_between(range(0, 180), all_simulation_weight_change_layer_5.mean(0)[i, j] - all_simulation_weight_change_layer_5.std(0)[i, j] / number_simulation ** 0.5, all_simulation_weight_change_layer_5.mean(0)[i, j] + all_simulation_weight_change_layer_5.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'm', facecolor = 'm')
                        
            ax.legend(loc = 'upper left', fontsize = 'x-small')
            ax.set_ylim((0, 0.0018))
            ax.set_xticks(np.arange(0, 180, 30.0))
            
    fig.savefig(parent_folder + '/Weight Change.tif')
        
    ### Layer rotation: a surprisingly powerful indicator of generalization in deep networks?
        
    fig, axs = plt.subplots(number_group, number_layer_freeze, figsize = (number_group * 4, number_layer_freeze * 3.5))
    fig.suptitle('Layer Rotation', fontsize = 20)
    
    for i in range(number_group):
        for j in range(number_layer_freeze):
            ax = axs[i, j]
            
            if i == 0:
                ax.set_title('Freezed Layer = ' + str(j))
            if i == 3:
                ax.set_xlabel('epoch')
            if j == 0:
                ax.set_ylabel('Group ' + str(i + 1))
                        
            ax.plot(range(0, 180), all_simulation_layer_rotation_layer_1.mean(0)[i, j], "-b", label = "Conv Layer 1")
            ax.fill_between(range(0, 180), all_simulation_layer_rotation_layer_1.mean(0)[i, j] - all_simulation_layer_rotation_layer_1.std(0)[i, j] / number_simulation ** 0.5, all_simulation_layer_rotation_layer_1.mean(0)[i, j] + all_simulation_layer_rotation_layer_1.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
            
            ax.plot(range(0, 180), all_simulation_layer_rotation_layer_2.mean(0)[i, j], "-g", label = "Conv Layer 2")
            ax.fill_between(range(0, 180), all_simulation_layer_rotation_layer_2.mean(0)[i, j] - all_simulation_layer_rotation_layer_2.std(0)[i, j] / number_simulation ** 0.5, all_simulation_layer_rotation_layer_2.mean(0)[i, j] + all_simulation_layer_rotation_layer_2.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
                        
            ax.plot(range(0, 180), all_simulation_layer_rotation_layer_3.mean(0)[i, j], "-r", label = "Conv Layer 3")
            ax.fill_between(range(0, 180), all_simulation_layer_rotation_layer_3.mean(0)[i, j] - all_simulation_layer_rotation_layer_3.std(0)[i, j] / number_simulation ** 0.5, all_simulation_layer_rotation_layer_3.mean(0)[i, j] + all_simulation_layer_rotation_layer_3.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
            
            ax.plot(range(0, 180), all_simulation_layer_rotation_layer_4.mean(0)[i, j], "-c", label = "Conv Layer 4")
            ax.fill_between(range(0, 180), all_simulation_layer_rotation_layer_4.mean(0)[i, j] - all_simulation_layer_rotation_layer_4.std(0)[i, j] / number_simulation ** 0.5, all_simulation_layer_rotation_layer_4.mean(0)[i, j] + all_simulation_layer_rotation_layer_4.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                        
            ax.plot(range(0, 180), all_simulation_layer_rotation_layer_5.mean(0)[i, j], "-m", label = "Conv Layer 5")
            ax.fill_between(range(0, 180), all_simulation_layer_rotation_layer_5.mean(0)[i, j] - all_simulation_layer_rotation_layer_5.std(0)[i, j] / number_simulation ** 0.5, all_simulation_layer_rotation_layer_5.mean(0)[i, j] + all_simulation_layer_rotation_layer_5.std(0)[i, j] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'm', facecolor = 'm')
                        
            ax.legend(loc = 'upper left', fontsize = 'x-small')
            ax.set_ylim((-2.5 * 10 ** (-7), 10 * 10 ** (-7)))
            ax.set_xticks(np.arange(0, 180, 30.0))
            
    fig.savefig(parent_folder + '/Layer Rotation.tif')
    
    ### ID: Intrinsic dimension of data representations in deep neural networks
    
    fig, axs = plt.subplots(2, 3, figsize = (2 * 8, 3 * 6))
    fig.suptitle('Intrinsic Dimension', fontsize = 20)
    
    for i in range(number_layer_freeze):
        if i <= 2:
            ax = axs[0, i]
        elif i > 2:
            ax = axs[1, i - 3]
        
        ax.set_title('Freezed Layer = ' + str(i), fontsize = 12)
        ax.set_ylabel('ID')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_ID, axis = 0)[0, :, i], "-b", label = "Group 1")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_ID, axis = 0)[0, :, i] - np.nanstd(all_simulation_all_ID, axis = 0)[0, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_ID, axis = 0)[0, :, i] + np.nanstd(all_simulation_all_ID, axis = 0)[0, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_ID, axis = 0)[1, :, i], "-g", label = "Group 2")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_ID, axis = 0)[1, :, i] - np.nanstd(all_simulation_all_ID, axis = 0)[1, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_ID, axis = 0)[1, :, i] + np.nanstd(all_simulation_all_ID, axis = 0)[1, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_ID, axis = 0)[2, :, i], "-r", label = "Group 3")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_ID, axis = 0)[2, :, i] - np.nanstd(all_simulation_all_ID, axis = 0)[2, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_ID, axis = 0)[2, :, i] + np.nanstd(all_simulation_all_ID, axis = 0)[2, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_ID, axis = 0)[3, :, i], "-c", label = "Group 4")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_ID, axis = 0)[3, :, i] - np.nanstd(all_simulation_all_ID, axis = 0)[3, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_ID, axis = 0)[3, :, i] + np.nanstd(all_simulation_all_ID, axis = 0)[3, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
        ax.legend(loc = 'upper right', fontsize = 'medium')
        ax.set_ylim((0, 100))
        ax.set_xticks(range(0, 5))
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
                
    fig.savefig(parent_folder + '/Intrinsic Dimension.tif')
    
    ### PWCCA & CKA
    
    counter = -1
        
    for i in range(number_simulation):
        for j in range(i + 1, number_simulation):
            counter = counter + 1
            
            group_counter = -1
            
            for group_training in ['group1', 'group2', 'group3', 'group4']:
                group_counter = group_counter + 1
                
                layer_freeze_counter = -1
                
                for layer_freeze in [None, 0, 3, 6, 8, 10]: 
                    layer_freeze_counter = layer_freeze_counter + 1
                    
                    print(str(i) + '-' + str(j) + '-' + group_training + '-' + str(layer_freeze))
                    
                    loading_folder_i = parent_folder + '/Simulation_' + str(i + 1) + '/' + group_training + '/after_training_' + str(layer_freeze)
                    
                    all_unit_activity_layer_i_1 = np.transpose(scipy.io.loadmat(loading_folder_i + '/all_unit_activity_Conv2d_1.mat')['all_unit_activity_Conv2d_1'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 55 * 55, 64)
                    all_unit_activity_layer_i_2 = np.transpose(scipy.io.loadmat(loading_folder_i + '/all_unit_activity_Conv2d_2.mat')['all_unit_activity_Conv2d_2'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 27 * 27, 192)
                    all_unit_activity_layer_i_3 = np.transpose(scipy.io.loadmat(loading_folder_i + '/all_unit_activity_Conv2d_3.mat')['all_unit_activity_Conv2d_3'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 13 * 13, 384)
                    all_unit_activity_layer_i_4 = np.transpose(scipy.io.loadmat(loading_folder_i + '/all_unit_activity_Conv2d_4.mat')['all_unit_activity_Conv2d_4'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 13 * 13, 256)
                    all_unit_activity_layer_i_5 = np.transpose(scipy.io.loadmat(loading_folder_i + '/all_unit_activity_Conv2d_5.mat')['all_unit_activity_Conv2d_5'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 13 * 13, 256)

                    loading_folder_j = parent_folder + '/Simulation_' + str(j + 1) + '/' + group_training + '/after_training_' + str(layer_freeze)
                    
                    all_unit_activity_layer_j_1 = np.transpose(scipy.io.loadmat(loading_folder_j + '/all_unit_activity_Conv2d_1.mat')['all_unit_activity_Conv2d_1'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 55 * 55, 64)
                    all_unit_activity_layer_j_2 = np.transpose(scipy.io.loadmat(loading_folder_j + '/all_unit_activity_Conv2d_2.mat')['all_unit_activity_Conv2d_2'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 27 * 27, 192)
                    all_unit_activity_layer_j_3 = np.transpose(scipy.io.loadmat(loading_folder_j + '/all_unit_activity_Conv2d_3.mat')['all_unit_activity_Conv2d_3'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 13 * 13, 384)
                    all_unit_activity_layer_j_4 = np.transpose(scipy.io.loadmat(loading_folder_j + '/all_unit_activity_Conv2d_4.mat')['all_unit_activity_Conv2d_4'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 13 * 13, 256)
                    all_unit_activity_layer_j_5 = np.transpose(scipy.io.loadmat(loading_folder_j + '/all_unit_activity_Conv2d_5.mat')['all_unit_activity_Conv2d_5'], axes = (0, 2, 3, 1)).reshape(num_sample_artiphysiology * 13 * 13, 256)     
                    
                    all_simulation_all_PWCCA[counter, group_counter, 0, layer_freeze_counter] = 1 - compute_pwcca(all_unit_activity_layer_i_1.T, all_unit_activity_layer_j_1.T, epsilon = 1e-10)[0]
                    all_simulation_all_PWCCA[counter, group_counter, 1, layer_freeze_counter] = 1 - compute_pwcca(all_unit_activity_layer_i_2.T, all_unit_activity_layer_j_2.T, epsilon = 1e-10)[0]
                    all_simulation_all_PWCCA[counter, group_counter, 2, layer_freeze_counter] = 1 - compute_pwcca(all_unit_activity_layer_i_3.T, all_unit_activity_layer_j_3.T, epsilon = 1e-10)[0]
                    all_simulation_all_PWCCA[counter, group_counter, 3, layer_freeze_counter] = 1 - compute_pwcca(all_unit_activity_layer_i_4.T, all_unit_activity_layer_j_4.T, epsilon = 1e-10)[0]
                    all_simulation_all_PWCCA[counter, group_counter, 4, layer_freeze_counter] = 1 - compute_pwcca(all_unit_activity_layer_i_5.T, all_unit_activity_layer_j_5.T, epsilon = 1e-10)[0]
                    
                    all_simulation_all_CKA[counter, group_counter, 0, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_1, all_unit_activity_layer_j_1, debiased = True)
                    all_simulation_all_CKA[counter, group_counter, 1, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_2, all_unit_activity_layer_j_2, debiased = True)
                    all_simulation_all_CKA[counter, group_counter, 2, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_3, all_unit_activity_layer_j_3, debiased = True)
                    all_simulation_all_CKA[counter, group_counter, 3, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_4, all_unit_activity_layer_j_4, debiased = True)
                    all_simulation_all_CKA[counter, group_counter, 4, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_5, all_unit_activity_layer_j_5, debiased = True)
                    
    scipy.io.savemat(parent_folder + '/all_simulation_all_PWCCA.mat', mdict = {'all_simulation_all_PWCCA': all_simulation_all_PWCCA})
    scipy.io.savemat(parent_folder + '/all_simulation_all_CKA.mat', mdict = {'all_simulation_all_CKA': all_simulation_all_CKA})
    
    ### PWCCA: Insights on representational similarity in neural networks with canonical correlation
    
    fig, axs = plt.subplots(2, 3, figsize = (2 * 8, 3 * 6))
    fig.suptitle('Projection Weighted Canonical Correlation Analysis', fontsize = 20)
    
    for i in range(number_layer_freeze):
        if i <= 2:
            ax = axs[0, i]
        elif i > 2:
            ax = axs[1, i - 3]
        
        ax.set_title('Freezed Layer = ' + str(i), fontsize = 12)
        ax.set_ylabel('PWCCA distance')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_PWCCA, axis = 0)[0, :, i], "-b", label = "Group 1")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_PWCCA, axis = 0)[0, :, i] - np.nanstd(all_simulation_all_PWCCA, axis = 0)[0, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_PWCCA, axis = 0)[0, :, i] + np.nanstd(all_simulation_all_PWCCA, axis = 0)[0, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_PWCCA, axis = 0)[1, :, i], "-g", label = "Group 2")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_PWCCA, axis = 0)[1, :, i] - np.nanstd(all_simulation_all_PWCCA, axis = 0)[1, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_PWCCA, axis = 0)[1, :, i] + np.nanstd(all_simulation_all_PWCCA, axis = 0)[1, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_PWCCA, axis = 0)[2, :, i], "-r", label = "Group 3")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_PWCCA, axis = 0)[2, :, i] - np.nanstd(all_simulation_all_PWCCA, axis = 0)[2, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_PWCCA, axis = 0)[2, :, i] + np.nanstd(all_simulation_all_PWCCA, axis = 0)[2, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_PWCCA, axis = 0)[3, :, i], "-c", label = "Group 4")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_PWCCA, axis = 0)[3, :, i] - np.nanstd(all_simulation_all_PWCCA, axis = 0)[3, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_PWCCA, axis = 0)[3, :, i] + np.nanstd(all_simulation_all_PWCCA, axis = 0)[3, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
        ax.legend(loc = 'upper left', fontsize = 'medium')
        ax.set_ylim((0, 0.03))
        ax.set_xticks(range(0, 5))
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
                
    fig.savefig(parent_folder + '/PWCCA Distance.tif')
       
    ### CKA: Similarity of Neural Network Representations Revisited
    
    fig, axs = plt.subplots(2, 3, figsize = (2 * 8, 3 * 6))
    fig.suptitle('Centered Kernel Alignment', fontsize = 20)
    
    for i in range(number_layer_freeze):
        if i <= 2:
            ax = axs[0, i]
        elif i > 2:
            ax = axs[1, i - 3]
        
        ax.set_title('Freezed Layer = ' + str(i), fontsize = 12)
        ax.set_ylabel('CKA distance')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_CKA, axis = 0)[0, :, i], "-b", label = "Group 1")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_CKA, axis = 0)[0, :, i] - np.nanstd(all_simulation_all_CKA, axis = 0)[0, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_CKA, axis = 0)[0, :, i] + np.nanstd(all_simulation_all_CKA, axis = 0)[0, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_CKA, axis = 0)[1, :, i], "-g", label = "Group 2")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_CKA, axis = 0)[1, :, i] - np.nanstd(all_simulation_all_CKA, axis = 0)[1, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_CKA, axis = 0)[1, :, i] + np.nanstd(all_simulation_all_CKA, axis = 0)[1, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_CKA, axis = 0)[2, :, i], "-r", label = "Group 3")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_CKA, axis = 0)[2, :, i] - np.nanstd(all_simulation_all_CKA, axis = 0)[2, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_CKA, axis = 0)[2, :, i] + np.nanstd(all_simulation_all_CKA, axis = 0)[2, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, 5), np.nanmean(all_simulation_all_CKA, axis = 0)[3, :, i], "-c", label = "Group 4")
        ax.fill_between(range(0, 5), np.nanmean(all_simulation_all_CKA, axis = 0)[3, :, i] - np.nanstd(all_simulation_all_CKA, axis = 0)[3, :, i] / number_simulation ** 0.5, np.nanmean(all_simulation_all_CKA, axis = 0)[3, :, i] + np.nanstd(all_simulation_all_CKA, axis = 0)[3, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
        ax.legend(loc = 'upper left', fontsize = 'medium')
        ax.set_ylim((0, 0.005))
        ax.set_xticks(range(0, 5))
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
                
    fig.savefig(parent_folder + '/CKA Distance.tif')
    
def imshow(x_sample, title):
    """Imshow for Tensor"""
    
    x_sample = x_sample.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x_sample = (std * x_sample + mean) / 255.0
    x_sample = np.clip(x_sample, 0, 1)
    
    plt.figure()
    plt.imshow(x_sample)
    plt.title(title)
    plt.pause(0.01)
    plt.close()
    
def adjust_learning_rate(optimizer, session, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 1 session"""
    
    lr = lr * (0.5 ** (session))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk = 1):
    """Computes the accuracy over the top1 predictions"""
    
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].view(-1).float().sum(0, keepdim = True)
        res.append(correct_k.mul_(100.0 / batch_size))
        
        return res
    
def save_checkpoint(state, is_best, group, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'DNNforVPL_best_' + group + '.pth.tar')

def plot_resp_lowd(resp_dict, layer_freeze, num_group, num_layer, parent_folder):
    """Plot a low-dimensional representation of each dataset in resp_dict using PCA."""
    
    fig, axs = plt.subplots(num_group, num_layer, figsize = (num_group * 4, num_layer * 3.5))
    fig.suptitle('Dimensionality Reduction with PCA >>> Freezed Layer = ' + str(layer_freeze), fontsize = 20)
    
    for i, (label, resp) in enumerate(resp_dict.items()):
        row, column = np.unravel_index(i, (num_group, num_layer), order = 'C')
        ax = axs[row, column]
        
        if row == 0:
            ax.set_title('Layer = ' + str(column + 1))
        if column == 0:
            ax.set_ylabel('Group ' + str(row + 1))
    
        # Do PCA to reduce dimensionality to 2 dimensions
        resp_lowd = PCA(n_components = 2).fit_transform(resp)
    
        # Plot dimensionality-reduced population responses on 2D axes, with each point colored by stimulus orientation and ref
        x, y = resp_lowd[:, 0], resp_lowd[:, 1]
        
        point_label = np.zeros(len(x))
        point_label[0:10] = 0
        point_label[10:20] = 1
        classes = ['CW', 'CCW']
        colours = ListedColormap(['b','g'])
        
        scatter_legend = ax.scatter(x, y, c = point_label, cmap = colours)
        ax.legend(handles = scatter_legend.legend_elements()[0], labels = classes)
    
    fig.savefig(parent_folder + '/DL with PCA_FL = ' + str(layer_freeze) + '.tif')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self, name, fmt = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        
        if self.name == 'Accuracy':
            self.__dict__['val'] = self.val.item()
            self.__dict__['avg'] = self.avg.item()
            self.__dict__['sum'] = self.sum.item()            
            output = fmtstr.format(**self.__dict__)
        else:            
            output = fmtstr.format(**self.__dict__)
        
        return output

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
   
if __name__ == '__main__':
    main()