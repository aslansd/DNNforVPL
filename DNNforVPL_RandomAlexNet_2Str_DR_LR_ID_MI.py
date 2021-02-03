"""
Created on Thu Jul  9 11:29:28 2020

@author: satarydizaji
"""

import os
import copy
import gc
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pingouin as pg
from PIL import Image
import random
import scipy.io
import seaborn as sns
from scipy import stats
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
from mutual_info_EDGE import EDGE
from scipy.spatial.distance import pdist, squareform

# Initialize the weights of the convolutional layers of AlexNet
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}
pretrained_dict = load_state_dict_from_url(model_urls['alexnet'])

# The DNN model for VPL
class DNNforVPL(nn.Module):
    
    def __init__(self, num_classes = 1):
        
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
    
    def forward(self, x1, x2):
        x1 = self.features(x1)
        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.classifier(x1)
        
        x2 = self.features(x2)
        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.classifier(x2)
        
        SoftMax = nn.Softmax(dim = -1)
        input = torch.cat((x1, x2), -1)
        output = SoftMax(input)
        
        return output
    
def main():
    
    ### Initializing the main variables
    
    num_sample_artiphysiology = 1000
    x_sample_artiphysiology_index = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
    
    for i in range(0, num_sample_artiphysiology):
        x_sample_artiphysiology_index[i, 0] = random.randrange(1)
        x_sample_artiphysiology_index[i, 1] = random.randrange(20)
        x_sample_artiphysiology_index[i, 2] = random.randrange(180)
        
    number_model = 5
    number_simulation = 20
    number_group = 4
    number_transfer_stimuli = 20
    number_PCA_component = 20
    
    number_channel = [64, 192, 384, 256, 256]
        
    all_simulation_training_accuracy = []
    all_simulation_transfer_accuracy = []
    all_simulation_specificity_index = []
    all_simulation_all_MI_original = []
    all_simulation_all_MI_noise = []
    all_simulation_all_ID = []
    all_x_sample_ID = []
    
    all_simulation_training_accuracy_permuted = []
    all_simulation_all_ID_permuted = []
    
    all_simulation_unit_activity_layer = []
    all_PCA_explained_variance_layer = []
    all_simulation_weight_change_layer = []
    all_simulation_layer_rotation_layer = []
                
    for i in range(number_model):
        number_layer = number_model
            
        all_simulation_training_accuracy.append(np.zeros((number_simulation, number_group, 180), dtype = np.float32))
        all_simulation_transfer_accuracy.append(np.zeros((number_simulation, number_group, 10), dtype = np.float32))
        all_simulation_specificity_index.append(np.zeros((number_simulation, number_group), dtype = np.float32))
        all_simulation_all_MI_original.append(np.zeros((number_simulation, number_group, number_layer), dtype = np.float32))
        all_simulation_all_MI_noise.append(np.zeros((number_simulation, number_group, number_layer), dtype = np.float32))
        all_simulation_all_ID.append(np.zeros((number_simulation, number_group, number_layer, 19), dtype = np.float32))
        all_x_sample_ID.append(np.zeros((number_simulation, number_group), dtype = np.float32))
        
        all_simulation_training_accuracy_permuted.append(np.zeros((number_simulation, number_group, 180), dtype = np.float32))
        all_simulation_all_ID_permuted.append(np.zeros((number_simulation, number_group, number_layer, 19), dtype = np.float32))
        
        all_simulation_unit_activity_layer_temp = []
        all_PCA_explained_variance_layer_temp = []
        all_simulation_weight_change_layer_temp = []
        all_simulation_layer_rotation_layer_temp = []
        
        for j in range(number_model):
            all_simulation_unit_activity_layer_temp.append(np.zeros((number_simulation, number_group, number_transfer_stimuli, number_channel[j]), dtype = np.float32))
            all_PCA_explained_variance_layer_temp.append(np.zeros((number_simulation, number_group, number_PCA_component), dtype = np.float32))
            all_simulation_weight_change_layer_temp.append(np.zeros((number_simulation, number_group, 180), dtype = np.float32))
            all_simulation_layer_rotation_layer_temp.append(np.zeros((number_simulation, number_group, 180), dtype = np.float32))
            
        all_simulation_unit_activity_layer.append(all_simulation_unit_activity_layer_temp)
        all_PCA_explained_variance_layer.append(all_PCA_explained_variance_layer_temp)
        all_simulation_weight_change_layer.append(all_simulation_weight_change_layer_temp)
        all_simulation_layer_rotation_layer.append(all_simulation_layer_rotation_layer_temp)
            
    # Cosine distance definition
    CosSim = nn.CosineSimilarity(dim = 0, eps = 1e-10)
       
    parent_folder = 'New_Results_RandomAlexNet_2Str_DR_LR_ID_MI'
        
    os.mkdir(parent_folder)
    
    for simulation_counter in range(number_simulation):
        print('Simulation:   ', simulation_counter + 1)
        
        os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1))
            
        group_counter = -1
        
        for group_training in ['group1', 'group2', 'group3', 'group4']:
            gc.collect()
            group_counter = group_counter + 1
            
            best_acc1 = 0
            
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
            
            for p in range(10):
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
            
            for model_counter in range(number_model):                               
                print('DNN Model:   ' + str(model_counter + 1))
                
                # Reading the reference image
                file_name_path_ref = glob.glob('VPL Stimuli/Learning & Transfer_SF (32)/ReferenceStimulus.TIFF')
                
                # Define the main reference variables
                x_val_ref = np.zeros((224, 224, 3), dtype = np.float32)
                x_tensor_ref = []
                
                # Load image
                img = Image.open(file_name_path_ref[0]).convert('RGB')
                
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
                x_val_ref[:, :, :] = img[:, :, :]
                
                # Convert image to tensor and normalize and copy
                x_temp = torch.from_numpy(np.transpose(x_val_ref[:, :, :], (2, 0, 1)))
                normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                
                for i in range(len(SF_training) * len(Ori_training)):
                    x_tensor_ref.append(normalize(x_temp))
                    
                x_tensor_ref = torch.stack(x_tensor_ref)
                print(x_tensor_ref.shape)
                
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
                
                if model_counter == 0:
                    torch.nn.init.xavier_uniform_(model.features[3].weight)
                    torch.nn.init.xavier_uniform_(model.features[6].weight)
                    torch.nn.init.xavier_uniform_(model.features[8].weight)
                    torch.nn.init.xavier_uniform_(model.features[10].weight)
                    
                    torch.nn.init.zeros_(model.features[3].bias)
                    torch.nn.init.zeros_(model.features[6].bias)
                    torch.nn.init.zeros_(model.features[8].bias)
                    torch.nn.init.zeros_(model.features[10].bias)
                    
                elif model_counter == 1:
                    torch.nn.init.xavier_uniform_(model.features[6].weight)
                    torch.nn.init.xavier_uniform_(model.features[8].weight)
                    torch.nn.init.xavier_uniform_(model.features[10].weight)
      
                    torch.nn.init.zeros_(model.features[6].bias)
                    torch.nn.init.zeros_(model.features[8].bias)
                    torch.nn.init.zeros_(model.features[10].bias)
                    
                elif model_counter == 2:
                    torch.nn.init.xavier_uniform_(model.features[8].weight)
                    torch.nn.init.xavier_uniform_(model.features[10].weight)
                    
                    torch.nn.init.zeros_(model.features[8].bias)
                    torch.nn.init.zeros_(model.features[10].bias)
                    
                elif model_counter == 3:   
                    torch.nn.init.xavier_uniform_(model.features[10].weight)
                    
                    torch.nn.init.zeros_(model.features[10].bias)
                    
                elif model_counter == 4:
                    pass
                
                # Initialize the weights of the fully-connected layer of the model
                nn.init.zeros_(model.classifier[0].weight)
                nn.init.zeros_(model.classifier[0].bias)
                
                # Set all the parameters of the model to be trained
                for param in model.parameters():
                    param.requires_grad = True
                                    
                # Send the model to GPU/CPU
                model = model.to(device)
                
                # Model summary
                print(model)
                    
                cudnn.benchmark = True
                                
                # The convolutional layers: (0, 3, 6, 8, 10)
                # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # The central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                
                # The target stimuli
                os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/before_training')
                saving_folder = parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/before_training'
                
                feature_sample_artiphysiology = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
                
                all_x_sample = np.zeros((num_sample_artiphysiology, 3, 224, 224), dtype = np.float32)
                
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
                    
                    # Calculating the intrinsic dimension of stimuli
                    all_x_sample[i, :] = x_sample.detach().cpu().clone().numpy()                   
                                        
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
                    
                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_5[i, :] = unit_activity_layer_10[0].detach().cpu().clone().numpy()
                    
                all_x_sample_ID[model_counter][simulation_counter, group_counter] = estimate(squareform(pdist(all_x_sample.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                
                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 3, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 4, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_5.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                
                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0]
                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, 0]
                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 2, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, 0]
                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 3, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 3, 0]
                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 4, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 4, 0]
                                        
                scipy.io.savemat(saving_folder + '/feature_sample_artiphysiology.mat', mdict = {'feature_sample_artiphysiology': feature_sample_artiphysiology})                        
                
                # Define the main learning parameters
                lr = 0.00001
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
                    ID_counter = 0
                    
                    for epoch in range(epochs):                       
                        z_val_shuffle_1D = np.unique(z_val_shuffle[:, :, epoch])
                        indices = torch.tensor(z_val_shuffle_1D, dtype = torch.long)
                        x_train = torch.index_select(x_tensor_training, 0, indices)
                        y_train = torch.index_select(y_tensor_training, 0, indices)
                        y_train = y_train.squeeze(1)
                        
                        batch_time = AverageMeter('Time', ':6.3f')
                        losses = AverageMeter('Loss', ':.4e')
                        top1 = AverageMeter('Accuracy', ':6.2f')
                        progress = ProgressMeter(epochs, [batch_time, losses, top1], prefix = ("Training >>> Session:   " + str(session) + "   Epoch: [{}]").format(epoch))
                    
                        # Switch to training mode
                        model.train()
                        
                        with torch.set_grad_enabled(True):
                            end = time.time()
                    
                            x_ref = x_tensor_ref.cuda(gpu)
                            x_train = x_train.cuda(gpu)
                            y_train = y_train.cuda(gpu)
                    
                            # Compute output
                            output = model(x_train, x_ref)
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
                            all_simulation_training_accuracy[model_counter][simulation_counter, group_counter, epoch] = acc1[0].item()
                            
                            # Measure elapsed time
                            batch_time.update(time.time() - end)
                    
                            progress.display(epoch)
                            
                        # Remember the best accuracy
                        is_best = all_simulation_training_accuracy[model_counter][simulation_counter, group_counter, epoch] >= best_acc1
                        best_acc1 = max(all_simulation_training_accuracy[model_counter][simulation_counter, group_counter, epoch], best_acc1)
                        
                        all_simulation_weight_change_layer[model_counter][0][simulation_counter, group_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[0].weight - Conv2d_1_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[0].weight, 2)), 0.5)).item()
                        all_simulation_weight_change_layer[model_counter][1][simulation_counter, group_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[3].weight - Conv2d_2_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[3].weight, 2)), 0.5)).item()
                        all_simulation_weight_change_layer[model_counter][2][simulation_counter, group_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[6].weight - Conv2d_3_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[6].weight, 2)), 0.5)).item()
                        all_simulation_weight_change_layer[model_counter][3][simulation_counter, group_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[8].weight - Conv2d_4_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[8].weight, 2)), 0.5)).item()
                        all_simulation_weight_change_layer[model_counter][4][simulation_counter, group_counter, epoch] = (torch.pow(torch.sum(torch.pow(model.features[10].weight - Conv2d_5_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[10].weight, 2)), 0.5)).item()
                        
                        all_simulation_layer_rotation_layer[model_counter][0][simulation_counter, group_counter, epoch] = 1 - CosSim(torch.flatten(model.features[0].weight), torch.flatten(Conv2d_1_0)).item()
                        all_simulation_layer_rotation_layer[model_counter][1][simulation_counter, group_counter, epoch] = 1 - CosSim(torch.flatten(model.features[3].weight), torch.flatten(Conv2d_2_0)).item()
                        all_simulation_layer_rotation_layer[model_counter][2][simulation_counter, group_counter, epoch] = 1 - CosSim(torch.flatten(model.features[6].weight), torch.flatten(Conv2d_3_0)).item()
                        all_simulation_layer_rotation_layer[model_counter][3][simulation_counter, group_counter, epoch] = 1 - CosSim(torch.flatten(model.features[8].weight), torch.flatten(Conv2d_4_0)).item()
                        all_simulation_layer_rotation_layer[model_counter][4][simulation_counter, group_counter, epoch] = 1 - CosSim(torch.flatten(model.features[10].weight), torch.flatten(Conv2d_5_0)).item()
                        
                        if (epoch + 1) % 10 == 0:
                            ID_counter = ID_counter + 1
                            
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
                                
                                all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                                all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                                all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                                all_unit_activity_Conv2d_5[i, :] = unit_activity_layer_10[0].detach().cpu().clone().numpy()
                                
                            all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                            all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                            all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                            all_simulation_all_ID[model_counter][simulation_counter, group_counter, 3, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                            all_simulation_all_ID[model_counter][simulation_counter, group_counter, 4, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_5.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]                      
                    
                # Save the checkpoint
                save_checkpoint({
                    'session': session + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, group_training, 'DNNforVPL_' + group_training + '.pth.tar')
                
                # Reading the reference image
                file_name_path_ref = glob.glob('VPL Stimuli/Learning & Transfer_SF (32)/ReferenceStimulus.TIFF')
                        
                # Define the main reference variables
                x_val_ref = np.zeros((224, 224, 3), dtype = np.float32)
                x_tensor_ref = []
                
                # Load image
                img = Image.open(file_name_path_ref[0]).convert('RGB')
                
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
                x_val_ref[:, :, :] = img[:, :, :]
                
                # Convert image to tensor and normalize and copy
                x_temp = torch.from_numpy(np.transpose(x_val_ref[:, :, :], (2, 0, 1)))
                normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                
                for i in range(len(SF_transfer) * len(Ori_transfer)):
                    x_tensor_ref.append(normalize(x_temp))
                    
                x_tensor_ref = torch.stack(x_tensor_ref)
                print(x_tensor_ref.shape)
                
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
                    progress = ProgressMeter(1, [batch_time, losses, top1], prefix = ("Transfer >>> Session:   " + str(session) + "   Epoch: [{}]").format(1))
                
                    # Switch to evaluating mode
                    model.eval()
                
                    with torch.no_grad():
                        end = time.time()
                        
                        x_ref = x_tensor_ref.cuda(gpu)
                        x_valid = x_valid.cuda(gpu)
                        y_valid = y_valid.cuda(gpu)
            
                        # Compute output
                        output = model(x_valid, x_ref)
                        loss = criterion(output, y_valid)
            
                        # Measure accuracy and record loss
                        acc1 = accuracy(output, y_valid, topk = 1)
                        losses.update(loss.item(), x_valid.size(0))
                        top1.update(acc1[0], x_valid.size(0))
                        
                        # Save the validation accuracy for plotting
                        all_simulation_transfer_accuracy[model_counter][simulation_counter, group_counter, session - start_session] = acc1[0].item()
            
                        # Measure elapsed time
                        batch_time.update(time.time() - end)
            
                        progress.display(1)
                        
                    # Remember the best accuracy and save checkpoint
                    is_best = all_simulation_transfer_accuracy[model_counter][simulation_counter, group_counter, session - start_session] >= best_acc1
                    best_acc1 = max(all_simulation_transfer_accuracy[model_counter][simulation_counter, group_counter, session - start_session], best_acc1)
                
                # The convolutional layers: (0, 3, 6, 8, 10)
                # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # The central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                               
                # The target stimuli
                os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/after_training')
                saving_folder = parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/after_training'
                            
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
                    
                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                    all_unit_activity_Conv2d_5[i, :] = unit_activity_layer_10[0].detach().cpu().clone().numpy()
                                        
                scipy.io.savemat(saving_folder + '/feature_sample_artiphysiology.mat', mdict = {'feature_sample_artiphysiology': feature_sample_artiphysiology})         
                
                ### Saving all units activity for all transfer stimuli and calculating the variance explained by PCA
                
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
                                
                all_simulation_unit_activity_layer[model_counter][0][simulation_counter, group_counter, :, :] = all_unit_activity_analysis_layer_1.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 64)
                all_simulation_unit_activity_layer[model_counter][1][simulation_counter, group_counter, :, :] = all_unit_activity_analysis_layer_2.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 192)
                all_simulation_unit_activity_layer[model_counter][2][simulation_counter, group_counter, :, :] = all_unit_activity_analysis_layer_3.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 384)
                all_simulation_unit_activity_layer[model_counter][3][simulation_counter, group_counter, :, :] = all_unit_activity_analysis_layer_4.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 256)
                all_simulation_unit_activity_layer[model_counter][4][simulation_counter, group_counter, :, :] = all_unit_activity_analysis_layer_5.mean(axis = (2, 3)).reshape(number_transfer_stimuli, 256)
                
                PCA_layer_1 = PCA(n_components = number_PCA_component).fit(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1))
                PCA_layer_2 = PCA(n_components = number_PCA_component).fit(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1))
                PCA_layer_3 = PCA(n_components = number_PCA_component).fit(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1))
                PCA_layer_4 = PCA(n_components = number_PCA_component).fit(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1))
                PCA_layer_5 = PCA(n_components = number_PCA_component).fit(all_unit_activity_Conv2d_5.reshape(num_sample_artiphysiology, -1))
                
                all_PCA_explained_variance_layer[model_counter][0][simulation_counter, group_counter, :] = PCA_layer_1.explained_variance_ratio_
                all_PCA_explained_variance_layer[model_counter][1][simulation_counter, group_counter, :] = PCA_layer_2.explained_variance_ratio_
                all_PCA_explained_variance_layer[model_counter][2][simulation_counter, group_counter, :] = PCA_layer_3.explained_variance_ratio_
                all_PCA_explained_variance_layer[model_counter][3][simulation_counter, group_counter, :] = PCA_layer_4.explained_variance_ratio_
                all_PCA_explained_variance_layer[model_counter][4][simulation_counter, group_counter, :] = PCA_layer_5.explained_variance_ratio_
                    
                ### Emergence of Invariance and Disentanglement in Deep Representations
                
                # The convolutional layers: (0, 3, 6, 8, 10)
                # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # The central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                
                phase_count = 20                   
                counter = -1
                
                x_tensor_training_original = np.zeros((len(SF_training) * len(Ori_training) * phase_count, 3, 224, 224), dtype = np.float32)
                x_tensor_training_noise = np.zeros((len(SF_training) * len(Ori_training) * phase_count, 3, 224, 224), dtype = np.float32)
                
                all_unit_activity_MI_Conv2d_1 = np.zeros((len(SF_training) * len(Ori_training) * phase_count, 64, 55, 55), dtype = np.float32)
                all_unit_activity_MI_Conv2d_2 = np.zeros((len(SF_training) * len(Ori_training) * phase_count, 192, 27, 27), dtype = np.float32)
                all_unit_activity_MI_Conv2d_3 = np.zeros((len(SF_training) * len(Ori_training) * phase_count, 384, 13, 13), dtype = np.float32)
                all_unit_activity_MI_Conv2d_4 = np.zeros((len(SF_training) * len(Ori_training) * phase_count, 256, 13, 13), dtype = np.float32)
                all_unit_activity_MI_Conv2d_5 = np.zeros((len(SF_training) * len(Ori_training) * phase_count, 256, 13, 13), dtype = np.float32)
                
                for i in range(len(SF_training)):
                    for j in range(len(Ori_training)):
                        phase = np.random.permutation(180)[:phase_count]
                        
                        for k in range(phase_count):
                            counter = counter + 1
                            
                            indices_training_1 = torch.tensor(z_val_training[i, j, phase[k]], dtype = torch.long)
                            indices_training_2 = torch.tensor(z_val_training[int(len(SF_training) / 2 + 0.5) - 1, j, phase[k]], dtype = torch.long)
                            
                            x_tensor_training_original[counter, :] = torch.index_select(x_tensor_training, 0, indices_training_1).detach().cpu().clone().numpy()
                            x_tensor_training_noise[counter, :] = (torch.index_select(x_tensor_training, 0, indices_training_1) - torch.index_select(x_tensor_training, 0, indices_training_2)).cuda(gpu)[0].detach().cpu().clone().numpy()
                            
                            x_sample = torch.index_select(x_tensor_training, 0, indices_training_1)
                            x_sample = x_sample.cuda(gpu)
                                                       
                            ### Calculating the mutual information between the original stimuli and layers activities, and the mutual information between the nuisance stimuli and layers activities
                            
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
                            
                            all_unit_activity_MI_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                            all_unit_activity_MI_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                            all_unit_activity_MI_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                            all_unit_activity_MI_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                            all_unit_activity_MI_Conv2d_5[i, :] = unit_activity_layer_10[0].detach().cpu().clone().numpy()
            
                all_simulation_all_MI_original[model_counter][simulation_counter, group_counter, 0] = EDGE(x_tensor_training_original.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_1.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), 
                                                                                                           U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                all_simulation_all_MI_original[model_counter][simulation_counter, group_counter, 1] = EDGE(x_tensor_training_original.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_2.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1),
                                                                                                           U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                all_simulation_all_MI_original[model_counter][simulation_counter, group_counter, 2] = EDGE(x_tensor_training_original.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_3.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1),
                                                                                                           U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                all_simulation_all_MI_original[model_counter][simulation_counter, group_counter, 3] = EDGE(x_tensor_training_original.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_4.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1),
                                                                                                           U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                all_simulation_all_MI_original[model_counter][simulation_counter, group_counter, 4] = EDGE(x_tensor_training_original.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_5.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1),
                                                                                                           U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                                
                all_simulation_all_MI_noise[model_counter][simulation_counter, group_counter, 0] = EDGE(x_tensor_training_noise.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_1.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), 
                                                                                                        U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                all_simulation_all_MI_noise[model_counter][simulation_counter, group_counter, 1] = EDGE(x_tensor_training_noise.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_2.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1),
                                                                                                        U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                all_simulation_all_MI_noise[model_counter][simulation_counter, group_counter, 2] = EDGE(x_tensor_training_noise.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_3.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1),
                                                                                                        U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                all_simulation_all_MI_noise[model_counter][simulation_counter, group_counter, 3] = EDGE(x_tensor_training_noise.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_4.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1),
                                                                                                        U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
                all_simulation_all_MI_noise[model_counter][simulation_counter, group_counter, 4] = EDGE(x_tensor_training_noise.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1), all_unit_activity_MI_Conv2d_5.mean(axis = 1).reshape(len(SF_training) * len(Ori_training) * phase_count, -1),
                                                                                                        U = 10, gamma = [1, 1], epsilon_vector = 'range', eps_range_factor = 0.1, normalize_epsilon = False, ensemble_estimation = 'median', L_ensemble = 5, hashing = 'p-stable', stochastic = False)
    
                ### Training with Permuted Labels
                
                print('Training with Permuted Labels')
                
                # Load the PyTorch model
                model = DNNforVPL()
                                    
                model_dict = model.state_dict()
                
                # Filter out unnecessary keys
                pretrained_dict_model = {k : v for k, v in pretrained_dict.items() if k in model_dict}
                # Overwrite entries in the existing state dict
                model_dict.update(pretrained_dict_model)
                # Load the new state dict
                model.load_state_dict(model_dict)
                
                if model_counter == 0:
                    torch.nn.init.xavier_uniform_(model.features[3].weight)
                    torch.nn.init.xavier_uniform_(model.features[6].weight)
                    torch.nn.init.xavier_uniform_(model.features[8].weight)
                    torch.nn.init.xavier_uniform_(model.features[10].weight)
                    
                    torch.nn.init.zeros_(model.features[3].bias)
                    torch.nn.init.zeros_(model.features[6].bias)
                    torch.nn.init.zeros_(model.features[8].bias)
                    torch.nn.init.zeros_(model.features[10].bias)
                    
                elif model_counter == 1:
                    torch.nn.init.xavier_uniform_(model.features[6].weight)
                    torch.nn.init.xavier_uniform_(model.features[8].weight)
                    torch.nn.init.xavier_uniform_(model.features[10].weight)
      
                    torch.nn.init.zeros_(model.features[6].bias)
                    torch.nn.init.zeros_(model.features[8].bias)
                    torch.nn.init.zeros_(model.features[10].bias)
                    
                elif model_counter == 2:
                    torch.nn.init.xavier_uniform_(model.features[8].weight)
                    torch.nn.init.xavier_uniform_(model.features[10].weight)
                    
                    torch.nn.init.zeros_(model.features[8].bias)
                    torch.nn.init.zeros_(model.features[10].bias)
                    
                elif model_counter == 3:   
                    torch.nn.init.xavier_uniform_(model.features[10].weight)
                    
                    torch.nn.init.zeros_(model.features[10].bias)
                    
                elif model_counter == 4:
                    pass
                
                # Initialize the weights of the fully-connected layer of the model
                nn.init.zeros_(model.classifier[0].weight)
                nn.init.zeros_(model.classifier[0].bias)
                
                # Set all the parameters of the model to be trained
                for param in model.parameters():
                    param.requires_grad = True
                                    
                # Send the model to GPU/CPU
                model = model.to(device)
                    
                cudnn.benchmark = True
                
                # Define the main learning parameters
                lr = 0.00001
                momentum = 0.9
                weight_decay = 0.0001
                
                # Define the loss function (criterion) and optimizer
                criterion = nn.CrossEntropyLoss().cuda(gpu)
                optimizer = torch.optim.SGD(model.parameters(), lr, momentum = momentum, weight_decay = weight_decay)
                    
                # Define the main training/validation parameters
                start_session = 0
                sessions = 1
                
                # Random permutation of labels
                y_tensor_training_permuted = copy.deepcopy(y_tensor_training)
                idx = torch.randperm(y_tensor_training_permuted.nelement())
                y_tensor_training_permuted = y_tensor_training_permuted.view(-1)[idx].view(y_tensor_training_permuted.size())
                    
                for session in range(start_session, sessions):                   
                    # Adjust the learning rate
                    adjust_learning_rate(optimizer, session, lr)
                    
                    # Train on a training set        
                    epochs = 180
                    ID_counter = 0
                    
                    for epoch in range(epochs):                       
                        z_val_shuffle_1D = np.unique(z_val_shuffle[:, :, epoch])
                        indices = torch.tensor(z_val_shuffle_1D, dtype = torch.long)
                        x_train = torch.index_select(x_tensor_training, 0, indices)
                        y_train = torch.index_select(y_tensor_training_permuted, 0, indices)
                        y_train = y_train.squeeze(1)
                        
                        batch_time = AverageMeter('Time', ':6.3f')
                        losses = AverageMeter('Loss', ':.4e')
                        top1 = AverageMeter('Accuracy', ':6.2f')
                        progress = ProgressMeter(epochs, [batch_time, losses, top1], prefix = ("Training >>> Session:   " + str(session) + "   Epoch: [{}]").format(epoch))
                    
                        # Switch to training mode
                        model.train()
                        
                        with torch.set_grad_enabled(True):
                            end = time.time()
                    
                            x_ref = x_tensor_ref.cuda(gpu)
                            x_train = x_train.cuda(gpu)
                            y_train = y_train.cuda(gpu)
                    
                            # Compute output
                            output = model(x_train, x_ref)
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
                            all_simulation_training_accuracy_permuted[model_counter][simulation_counter, group_counter, epoch] = acc1[0].item()
                            
                            # Measure elapsed time
                            batch_time.update(time.time() - end)
                    
                            progress.display(epoch)
                            
                        # Remember the best accuracy
                        is_best = all_simulation_training_accuracy_permuted[model_counter][simulation_counter, group_counter, epoch] >= best_acc1
                        best_acc1 = max(all_simulation_training_accuracy_permuted[model_counter][simulation_counter, group_counter, epoch], best_acc1)
                        
                        if (epoch + 1) % 10 == 0:
                            ID_counter = ID_counter + 1
                            
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
                                
                                all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                                all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                                all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                                all_unit_activity_Conv2d_5[i, :] = unit_activity_layer_10[0].detach().cpu().clone().numpy()
                                
                            all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                            all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                            all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 2, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                            all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 3, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                            all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 4, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_5.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2] 

    ### Specificity Index
    
    for i in range(number_model):
        all_simulation_specificity_index = (all_simulation_training_accuracy[i][:, :, 179] - all_simulation_transfer_accuracy[i].mean(2)) / (all_simulation_training_accuracy[i][:, :, 179] - all_simulation_training_accuracy[i].mean(2))
    
    ### Saving the main variables
   
    scipy.io.savemat(parent_folder + '/all_simulation_training_accuracy.mat', mdict = {'all_simulation_training_accuracy': all_simulation_training_accuracy})
    scipy.io.savemat(parent_folder + '/all_simulation_transfer_accuracy.mat', mdict = {'all_simulation_transfer_accuracy': all_simulation_transfer_accuracy})
    scipy.io.savemat(parent_folder + '/all_simulation_specificity_index.mat', mdict = {'all_simulation_specificity_index': all_simulation_specificity_index})
    scipy.io.savemat(parent_folder + '/all_simulation_all_MI_original.mat', mdict = {'all_simulation_all_MI_original': all_simulation_all_MI_original})
    scipy.io.savemat(parent_folder + '/all_simulation_all_MI_noise.mat', mdict = {'all_simulation_all_MI_noise': all_simulation_all_MI_noise})
    scipy.io.savemat(parent_folder + '/all_simulation_all_ID.mat', mdict = {'all_simulation_all_ID': all_simulation_all_ID})
    scipy.io.savemat(parent_folder + '/all_x_sample_ID.mat', mdict = {'all_x_sample_ID': all_x_sample_ID})
    
    scipy.io.savemat(parent_folder + '/all_simulation_training_accuracy_permuted.mat', mdict = {'all_simulation_training_accuracy_permuted': all_simulation_training_accuracy_permuted})
    scipy.io.savemat(parent_folder + '/all_simulation_all_ID_permuted.mat', mdict = {'all_simulation_all_ID_permuted': all_simulation_all_ID_permuted})
            
    scipy.io.savemat(parent_folder + '/all_simulation_unit_activity_layer.mat', mdict = {'all_simulation_unit_activity_layer': all_simulation_unit_activity_layer})
    scipy.io.savemat(parent_folder + '/all_PCA_explained_variance_layer.mat', mdict = {'all_PCA_explained_variance_layer': all_PCA_explained_variance_layer})
    scipy.io.savemat(parent_folder + '/all_simulation_weight_change_layer.mat', mdict = {'all_simulation_weight_change_layer': all_simulation_weight_change_layer})
    scipy.io.savemat(parent_folder + '/all_simulation_layer_rotation_layer.mat', mdict = {'all_simulation_layer_rotation_layer': all_simulation_layer_rotation_layer})
       
    ### Training Accuracy with Correct Labels
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('Training Accuracy with Correct Labels', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
        
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('% Accuracy')
                    
        ax.plot(range(0, 180), all_simulation_training_accuracy[i].mean(0)[0], "-b", label = "Group 1")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy[i].mean(0)[0] - all_simulation_training_accuracy[i].std(0)[0] / number_simulation ** 0.5, all_simulation_training_accuracy[i].mean(0)[0] + all_simulation_training_accuracy[i].std(0)[0] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy[i].mean(0)[1], "-g", label = "Group 2")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy[i].mean(0)[1] - all_simulation_training_accuracy[i].std(0)[1] / number_simulation ** 0.5, all_simulation_training_accuracy[i].mean(0)[1] + all_simulation_training_accuracy[i].std(0)[1] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy[i].mean(0)[2], "-r", label = "Group 3")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy[i].mean(0)[2] - all_simulation_training_accuracy[i].std(0)[2] / number_simulation ** 0.5, all_simulation_training_accuracy[i].mean(0)[2] + all_simulation_training_accuracy[i].std(0)[2] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy[i].mean(0)[3], "-c", label = "Group 4")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy[i].mean(0)[3] - all_simulation_training_accuracy[i].std(0)[3] / number_simulation ** 0.5, all_simulation_training_accuracy[i].mean(0)[3] + all_simulation_training_accuracy[i].std(0)[3] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                   
        ax.legend(loc = 'lower right', fontsize = 'medium')
        ax.set_ylim((0, 105))
        ax.set_xticks(np.arange(0, 180, 30.0))
        
    fig.savefig(parent_folder + '/Training Accuracy with Correct Labels.png')
    
    ### Training Accuracy with Permuted Labels
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('Training Accuracy with Permuted Labels', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
        
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('% Accuracy')
                    
        ax.plot(range(0, 180), all_simulation_training_accuracy_permuted[i].mean(0)[0], "-b", label = "Group 1")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy_permuted[i].mean(0)[0] - all_simulation_training_accuracy_permuted[i].std(0)[0] / number_simulation ** 0.5, all_simulation_training_accuracy_permuted[i].mean(0)[0] + all_simulation_training_accuracy_permuted[i].std(0)[0] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy_permuted[i].mean(0)[1], "-g", label = "Group 2")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy_permuted[i].mean(0)[1] - all_simulation_training_accuracy_permuted[i].std(0)[1] / number_simulation ** 0.5, all_simulation_training_accuracy_permuted[i].mean(0)[1] + all_simulation_training_accuracy_permuted[i].std(0)[1] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy_permuted[i].mean(0)[2], "-r", label = "Group 3")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy_permuted[i].mean(0)[2] - all_simulation_training_accuracy_permuted[i].std(0)[2] / number_simulation ** 0.5, all_simulation_training_accuracy_permuted[i].mean(0)[2] + all_simulation_training_accuracy_permuted[i].std(0)[2] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, 180), all_simulation_training_accuracy_permuted[i].mean(0)[3], "-c", label = "Group 4")
        ax.fill_between(range(0, 180), all_simulation_training_accuracy_permuted[i].mean(0)[3] - all_simulation_training_accuracy_permuted[i].std(0)[3] / number_simulation ** 0.5, all_simulation_training_accuracy_permuted[i].mean(0)[3] + all_simulation_training_accuracy_permuted[i].std(0)[3] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                   
        ax.legend(loc = 'lower right', fontsize = 'medium')
        ax.set_ylim((0, 105))
        ax.set_xticks(np.arange(0, 180, 30.0))
        
    fig.savefig(parent_folder + '/Training Accuracy with Permuted Labels.png')
    
    ### Transfer Accuracy
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('Transfer Accuracy', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
        
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_ylabel('% Accuracy')
        
        bar_list = ax.bar(range(0, number_group), [all_simulation_transfer_accuracy[i].mean(axis = (0, 2))[0], 
                                        all_simulation_transfer_accuracy[i].mean(axis = (0, 2))[1], 
                                        all_simulation_transfer_accuracy[i].mean(axis = (0, 2))[2], 
                                        all_simulation_transfer_accuracy[i].mean(axis = (0, 2))[3]],
                          yerr = [all_simulation_transfer_accuracy[i].std(axis = (0, 2))[0],
                                  all_simulation_transfer_accuracy[i].std(axis = (0, 2))[1],
                                  all_simulation_transfer_accuracy[i].std(axis = (0, 2))[2],
                                  all_simulation_transfer_accuracy[i].std(axis = (0, 2))[3]])
        
        bar_list[0].set_color('b')
        bar_list[1].set_color('g')
        bar_list[2].set_color('r')
        bar_list[3].set_color('c')
        
        ax.set_ylim((0, 105))
        ax.set_xticks(range(0, number_group))
        ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3', 'Group 4'])
        
        t_stat_1, p_value_1 = stats.ttest_ind(all_simulation_transfer_accuracy[i, :, 0, :].flatten(), all_simulation_transfer_accuracy[i, :, 1, :].flatten(), equal_var = True, nan_policy = 'omit')
        t_stat_2, p_value_2 = stats.ttest_ind(all_simulation_transfer_accuracy[i, :, 2, :].flatten(), all_simulation_transfer_accuracy[i, :, 3, :].flatten(), equal_var = True, nan_policy = 'omit')
        
        ax.text(bar_list[1].get_x() + bar_list[1].get_width() * 0.5, 1.15 * bar_list[1].get_height(), 'p = {:.4e}'.format(p_value_1), ha = 'center', va = 'bottom', color = 'g')
        ax.text(bar_list[3].get_x() + bar_list[3].get_width() * 0.5, 1.15 * bar_list[3].get_height(), 'p = {:.4e}'.format(p_value_2), ha = 'center', va = 'bottom', color = 'c')
            
    fig.savefig(parent_folder + '/Transfer Accuracy.png')
    
    ### Specificity Index
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('Specificity Index', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
        
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_ylabel('Index')
        
        bar_list = ax.bar(range(0, number_group), [np.nanmean(all_simulation_specificity_index[i], axis = 0)[0], 
                                        np.nanmean(all_simulation_specificity_index[i], axis = 0)[1], 
                                        np.nanmean(all_simulation_specificity_index[i], axis = 0)[2], 
                                        np.nanmean(all_simulation_specificity_index[i], axis = 0)[3]],
                          yerr = [np.nanstd(all_simulation_specificity_index[i], axis = 0)[0],
                                  np.nanstd(all_simulation_specificity_index[i], axis = 0)[1],
                                  np.nanstd(all_simulation_specificity_index[i], axis = 0)[2],
                                  np.nanstd(all_simulation_specificity_index[i], axis = 0)[3]])
        
        bar_list[0].set_color('b')
        bar_list[1].set_color('g')
        bar_list[2].set_color('r')
        bar_list[3].set_color('c')
        
        ax.set_ylim((-0.2, 1.2))
        ax.set_xticks(range(0, number_group))
        ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3', 'Group 4'])

        t_stat_1, p_value_1 = stats.ttest_ind(all_simulation_specificity_index[i, :, 0].flatten(), all_simulation_specificity_index[i, :, 1].flatten(), equal_var = True, nan_policy = 'omit')
        t_stat_2, p_value_2 = stats.ttest_ind(all_simulation_specificity_index[i, :, 2].flatten(), all_simulation_specificity_index[i, :, 3].flatten(), equal_var = True, nan_policy = 'omit')
        
        ax.text(bar_list[1].get_x() + bar_list[1].get_width() * 0.5, 1.25 * bar_list[1].get_height(), 'p = {:.4e}'.format(p_value_1), ha = 'center', va = 'bottom', color = 'g')
        ax.text(bar_list[3].get_x() + bar_list[3].get_width() * 0.5, 1.25 * bar_list[3].get_height(), 'p = {:.4e}'.format(p_value_2), ha = 'center', va = 'bottom', color = 'c')
                
    fig.savefig(parent_folder + '/Specificity Index.png')
    
    ### Dimensionality reduction with PCA   
    
    resp_dict = {}
    
    for i in range(number_model):
        for j in range(number_group):
            for k in range(i):                
                label = str(j + 1) + str(k + 1)
                resp_dict[label] = all_simulation_unit_activity_layer[i][k].mean(0)[j, :, :]
    
        plot_resp_lowd(resp_dict, i, number_group, i + 1, parent_folder)
        
    ### Variance Explained by PCA
    
    fig, axs = plt.subplots(number_model, number_model, figsize = (number_model * 4, number_model * 4))
    fig.suptitle('Variance Explained by PCA', fontsize = 20)
    
    for j in range(number_model):
        for i in range(number_model):
            ax = axs[i, j]
            
            if i == 0:
                ax.set_title('DNN Model = ' + str(number_model - j), fontsize = 12)
            elif i == number_model:
                ax.set_xlabel('Components')
            if j == 0:
                ax.set_ylabel('Layer ' + str(i + 1))
                                         
            ax.plot(range(0, number_PCA_component), all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[0], "-b", label = "Group 1")
            ax.fill_between(range(0, number_PCA_component), all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[0] - all_PCA_explained_variance_layer[number_model - j - 1][i].std(0)[0] / number_simulation ** 0.5, all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[0] + all_PCA_explained_variance_layer[number_model - j - 1][i].std(0)[0] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
            
            ax.plot(range(0, number_PCA_component), all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[1], "-g", label = "Group 2")
            ax.fill_between(range(0, number_PCA_component), all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[1] - all_PCA_explained_variance_layer[number_model - j - 1][i].std(0)[1] / number_simulation ** 0.5, all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[1] + all_PCA_explained_variance_layer[number_model - j - 1][i].std(0)[1] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
                        
            ax.plot(range(0, number_PCA_component), all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[2], "-r", label = "Group 3")
            ax.fill_between(range(0, number_PCA_component), all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[2] - all_PCA_explained_variance_layer[number_model - j - 1][i].std(0)[2] / number_simulation ** 0.5, all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[2] + all_PCA_explained_variance_layer[number_model - j - 1][i].std(0)[2] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
            
            ax.plot(range(0, number_PCA_component), all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[3], "-c", label = "Group 4")
            ax.fill_between(range(0, number_PCA_component), all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[3] - all_PCA_explained_variance_layer[number_model - j - 1][i].std(0)[3] / number_simulation ** 0.5, all_PCA_explained_variance_layer[number_model - j - 1][i].mean(0)[3] + all_PCA_explained_variance_layer[number_model - j - 1][i].std(0)[3] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                                                
            ax.legend(loc = 'upper right', fontsize = 'x-small')
            ax.set_ylim((0, 1))
            ax.set_xlim((0, 5))
            
    fig.savefig(parent_folder + '/Variance Explained by PCA.png')
    
    ### Weight Change

    fig, axs = plt.subplots(number_group, number_model, figsize = (number_group * 4, number_model * 4))
    fig.suptitle('Weight Change', fontsize = 20)
    
    for i in range(number_group):
        for j in range(number_model):
            ax = axs[i, j]
            
            if i == 0:
                ax.set_title('DNN Model = ' + str(j + 1), fontsize = 12)
            if i == number_group - 1:
                ax.set_xlabel('Epoch')
            if j == 0:
                ax.set_ylabel('Group ' + str(i + 1))
                
            color_map = ['b', 'g', 'r', 'c', 'm']
                
            for k in range(number_model):                                  
                ax.plot(range(0, 180), all_simulation_weight_change_layer[j, k].mean(0)[i], color_map[k], label = "Conv Layer " + str(k + 1))
                ax.fill_between(range(0, 180), all_simulation_weight_change_layer[j, k].mean(0)[i] - all_simulation_weight_change_layer[j, k].std(0)[i] / number_simulation ** 0.5, all_simulation_weight_change_layer[j, k].mean(0)[i] + all_simulation_weight_change_layer[j, k].std(0)[i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = color_map[k], facecolor = color_map[k])
                
            ax.legend(loc = 'upper left', fontsize = 'x-small')
            ax.set_ylim((0, 0.0018))
            ax.set_xticks(np.arange(0, 180, 30.0))
            
            if i in [1, 3]:
                t_stat, p_value = stats.ttest_ind(all_simulation_weight_change_layer[j, 0, :, i].flatten(), all_simulation_weight_change_layer[j, 0, :, i - 1].flatten(), equal_var = True, nan_policy = 'omit')
                                
                label = 'p = {:.4e}'.format(p_value)
                ax.annotate(label, (179, all_simulation_weight_change_layer[j, 0].mean(0)[i, 179]), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'b')
            
    fig.savefig(parent_folder + '/Weight Change.png')
        
    ### Layer rotation: a surprisingly powerful indicator of generalization in deep networks?
        
    fig, axs = plt.subplots(number_group, number_model, figsize = (number_group * 4, number_model * 4))
    fig.suptitle('Layer Rotation', fontsize = 20)
    
    for i in range(number_group):
        for j in range(number_model):
            ax = axs[i, j]
            
            if i == 0:
                ax.set_title('DNN Model = ' + str(j + 1), fontsize = 12)
            if i == number_group - 1:
                ax.set_xlabel('Epoch')
            if j == 0:
                ax.set_ylabel('Group ' + str(i + 1))
                
            color_map = ['b', 'g', 'r', 'c', 'm']
            
            for k in range(number_model):                                  
                ax.plot(range(0, 180), all_simulation_layer_rotation_layer[j, k].mean(0)[i], color_map[k], label = "Conv Layer " + str(k + 1))
                ax.fill_between(range(0, 180), all_simulation_layer_rotation_layer[j, k].mean(0)[i] - all_simulation_layer_rotation_layer[j, k].std(0)[i] / number_simulation ** 0.5, all_simulation_layer_rotation_layer[j, k].mean(0)[i] + all_simulation_layer_rotation_layer[j, k].std(0)[i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = color_map[k], facecolor = color_map[k])
                        
            ax.legend(loc = 'upper left', fontsize = 'x-small')
            ax.set_ylim((-2.5 * 10 ** (-7), 10 * 10 ** (-7)))
            ax.set_xticks(np.arange(0, 180, 30.0))
            
            if i in [1, 3]:
                t_stat, p_value = stats.ttest_ind(all_simulation_layer_rotation_layer[j, 0, :, i].flatten(), all_simulation_layer_rotation_layer[j, 0, :, i - 1].flatten(), equal_var = True, nan_policy = 'omit')
                                
                label = 'p = {:.4e}'.format(p_value)
                ax.annotate(label, (179, all_simulation_layer_rotation_layer[j, 0].mean(0)[i, 179]), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'b')
            
    fig.savefig(parent_folder + '/Layer Rotation.png')
    
    ### Emergence of Invariance and Disentanglement in Deep Representations
    
    # Mutual information between the original stimuli and layers activities
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('Mutual Information between the Original Stimuli and Layers Activities', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
       
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_ylabel('MI')
        
        ax.plot(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[0, :], "-b", label = "Group 1")
        ax.fill_between(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[0, :] - np.nanstd(all_simulation_all_MI_original[i], axis = 0)[0, :] / number_simulation ** 0.5, np.nanmean(all_simulation_all_MI_original[i], axis = 0)[0, :] + np.nanstd(all_simulation_all_MI_original[i], axis = 0)[0, :] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[1, :], "-g", label = "Group 2")
        ax.fill_between(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[1, :] - np.nanstd(all_simulation_all_MI_original[i], axis = 0)[1, :] / number_simulation ** 0.5, np.nanmean(all_simulation_all_MI_original[i], axis = 0)[1, :] + np.nanstd(all_simulation_all_MI_original[i], axis = 0)[1, :] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[2, :], "-r", label = "Group 3")
        ax.fill_between(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[2, :] - np.nanstd(all_simulation_all_MI_original[i], axis = 0)[2, :] / number_simulation ** 0.5, np.nanmean(all_simulation_all_MI_original[i], axis = 0)[2, :] + np.nanstd(all_simulation_all_MI_original[i], axis = 0)[2, :] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[3, :], "-c", label = "Group 4")
        ax.fill_between(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[3, :] - np.nanstd(all_simulation_all_MI_original[i], axis = 0)[3, :] / number_simulation ** 0.5, np.nanmean(all_simulation_all_MI_original[i], axis = 0)[3, :] + np.nanstd(all_simulation_all_MI_original[i], axis = 0)[3, :] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
        ax.legend(loc = 'upper right', fontsize = 'medium')
        ax.set_ylim((0, 10))
        ax.set_xticks(range(0, i + 1))
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
        
        t_stat_lp = np.zeros(i + 1)
        t_stat_hp = np.zeros(i + 1)
        
        p_value_lp = np.zeros(i + 1)
        p_value_hp = np.zeros(i + 1)
        
        for j in range(i + 1):
            t_stat_lp[j], p_value_lp[j] = stats.ttest_ind(all_simulation_all_MI_original[i, :, 0, j], all_simulation_all_MI_original[i, :, 1, j], equal_var = True, nan_policy = 'omit')
            t_stat_hp[j], p_value_hp[j] = stats.ttest_ind(all_simulation_all_MI_original[i, :, 2, j], all_simulation_all_MI_original[i, :, 3, j], equal_var = True, nan_policy = 'omit')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[1, :])):
            label = 'p = {:.4e}'.format(p_value_lp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'g')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_MI_original[i], axis = 0)[3, :])):
            label = 'p = {:.4e}'.format(p_value_hp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, -10), ha = 'center', color = 'c')
            
        all_simulation_group_layer_MI = np.array([])
        
        for j in range(number_group):
            all_simulation_group_layer_MI_temp = np.array([])
            layer_map = []
            
            for k in range(number_model):
                all_simulation_group_layer_MI_temp = np.concatenate(all_simulation_group_layer_MI_temp, all_simulation_all_MI_original[i, :, j, k].flatten() - all_simulation_all_MI_original[i, :, j, k + 1].flatten())
                layer_map.append('Layer ' + str(k + 1) + str(k + 2))
            
            all_simulation_group_layer_MI = np.concatenate(all_simulation_group_layer_MI, all_simulation_group_layer_MI_temp)
        
        df = pd.DataFrame({'MI': all_simulation_group_layer_MI,
                           'Simulation': np.tile(np.arange(4 * number_simulation), 4),
                           'Layer': np.tile(np.repeat(layer_map, number_simulation), number_group),
                           'Group': np.concatenate((np.tile(['Group 1'], 4 * number_simulation), np.tile(['Group 2'], 4 * number_simulation), np.tile(['Group 3'], 4 * number_simulation), np.tile(['Group 4'], 4 * number_simulation)))})
        
        aov = pg.mixed_anova(dv = 'MI', within = 'Group', between = 'Layer', subject = 'Simulation', data = df)
        pg.print_table(aov)
        
        posthocs = pg.pairwise_ttests(dv = 'MI', within = 'Group', between = 'Layer', subject = 'Simulation', data = df)
        pg.print_table(posthocs)
                
    fig.savefig(parent_folder + '/Mutual Information between the Original Stimuli and Layers Activities.png')
    
    # Mutual information between the nuisance stimuli and layers activities
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('Mutual Information between the Nuisance Stimuli and Layers Activities', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
       
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_ylabel('MI')
        
        ax.plot(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[0, :], "-b", label = "Group 1")
        ax.fill_between(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[0, :] - np.nanstd(all_simulation_all_MI_noise[i], axis = 0)[0, :] / number_simulation ** 0.5, np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[0, :] + np.nanstd(all_simulation_all_MI_noise[i], axis = 0)[0, :] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[1, :], "-g", label = "Group 2")
        ax.fill_between(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[1, :] - np.nanstd(all_simulation_all_MI_noise[i], axis = 0)[1, :] / number_simulation ** 0.5, np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[1, :] + np.nanstd(all_simulation_all_MI_noise[i], axis = 0)[1, :] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[2, :], "-r", label = "Group 3")
        ax.fill_between(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[2, :] - np.nanstd(all_simulation_all_MI_noise[i], axis = 0)[2, :] / number_simulation ** 0.5, np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[2, :] + np.nanstd(all_simulation_all_MI_noise[i], axis = 0)[2, :] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[3, :], "-c", label = "Group 4")
        ax.fill_between(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[3, :] - np.nanstd(all_simulation_all_MI_noise[i], axis = 0)[3, :] / number_simulation ** 0.5, np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[3, :] + np.nanstd(all_simulation_all_MI_noise[i], axis = 0)[3, :] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
        ax.legend(loc = 'upper right', fontsize = 'medium')
        ax.set_ylim((0, 10))
        ax.set_xticks(range(0, i + 1))
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
        
        t_stat_lp = np.zeros(i + 1)
        t_stat_hp = np.zeros(i + 1)
        
        p_value_lp = np.zeros(i + 1)
        p_value_hp = np.zeros(i + 1)
        
        for j in range(i + 1):
            t_stat_lp[j], p_value_lp[j] = stats.ttest_ind(all_simulation_all_MI_noise[i, :, 0, j], all_simulation_all_MI_noise[i, :, 1, j], equal_var = True, nan_policy = 'omit')
            t_stat_hp[j], p_value_hp[j] = stats.ttest_ind(all_simulation_all_MI_noise[i, :, 2, j], all_simulation_all_MI_noise[i, :, 3, j], equal_var = True, nan_policy = 'omit')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[1, :])):
            label = 'p = {:.4e}'.format(p_value_lp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'g')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_MI_noise[i], axis = 0)[3, :])):
            label = 'p = {:.4e}'.format(p_value_hp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, -10), ha = 'center', color = 'c')
            
        all_simulation_group_layer_MI = np.array([])
        
        for j in range(number_group):
            all_simulation_group_layer_MI_temp = np.array([])
            layer_map = []
            
            for k in range(number_model):
                all_simulation_group_layer_MI_temp = np.concatenate(all_simulation_group_layer_MI_temp, all_simulation_all_MI_noise[i, :, j, k].flatten() - all_simulation_all_MI_noise[i, :, j, k + 1].flatten())
                layer_map.append('Layer ' + str(k + 1) + str(k + 2))
            
            all_simulation_group_layer_MI = np.concatenate(all_simulation_group_layer_MI, all_simulation_group_layer_MI_temp)
        
        df = pd.DataFrame({'MI': all_simulation_group_layer_MI,
                           'Simulation': np.tile(np.arange(4 * number_simulation), 4),
                           'Layer': np.tile(np.repeat(layer_map, number_simulation), number_group),
                           'Group': np.concatenate((np.tile(['Group 1'], 4 * number_simulation), np.tile(['Group 2'], 4 * number_simulation), np.tile(['Group 3'], 4 * number_simulation), np.tile(['Group 4'], 4 * number_simulation)))})
        
        aov = pg.mixed_anova(dv = 'MI', within = 'Group', between = 'Layer', subject = 'Simulation', data = df)
        pg.print_table(aov)
        
        posthocs = pg.pairwise_ttests(dv = 'MI', within = 'Group', between = 'Layer', subject = 'Simulation', data = df)
        pg.print_table(posthocs)
                
    fig.savefig(parent_folder + '/Mutual Information between the Nuisance Stimuli and Layers Activities.png')
    
    ### ID: Intrinsic dimension of data representations in deep neural networks
    
    # ID across layers and groups for correct labels
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('Intrinsic Dimension with Correct Labels', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
        
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_ylabel('ID')
        
        ax.plot(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[0], np.nanmean(all_simulation_all_ID[i], axis = 0)[0, :, -1]) , "-b", label = "Group 1")
        ax.fill_between(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[0], np.nanmean(all_simulation_all_ID[i], axis = 0)[0, :, -1] - np.nanstd(all_simulation_all_ID[i], axis = 0)[0, :, -1] / number_simulation ** 0.5), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[0], np.nanmean(all_simulation_all_ID[i], axis = 0)[0, :, -1] + np.nanstd(all_simulation_all_ID[i], axis = 0)[0, :, -1] / number_simulation ** 0.5), alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[1], np.nanmean(all_simulation_all_ID[i], axis = 0)[1, :, -1]) , "-g", label = "Group 2")
        ax.fill_between(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[1], np.nanmean(all_simulation_all_ID[i], axis = 0)[1, :, -1] - np.nanstd(all_simulation_all_ID[i], axis = 0)[1, :, -1] / number_simulation ** 0.5), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[1], np.nanmean(all_simulation_all_ID[i], axis = 0)[1, :, -1] + np.nanstd(all_simulation_all_ID[i], axis = 0)[1, :, -1] / number_simulation ** 0.5), alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[2], np.nanmean(all_simulation_all_ID[i], axis = 0)[2, :, -1]) , "-r", label = "Group 3")
        ax.fill_between(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[2], np.nanmean(all_simulation_all_ID[i], axis = 0)[2, :, -1] - np.nanstd(all_simulation_all_ID[i], axis = 0)[2, :, -1] / number_simulation ** 0.5), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[2], np.nanmean(all_simulation_all_ID[i], axis = 0)[2, :, -1] + np.nanstd(all_simulation_all_ID[i], axis = 0)[2, :, -1] / number_simulation ** 0.5), alpha = 0.5, edgecolor = 'c', facecolor = 'r')
        
        ax.plot(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[3], np.nanmean(all_simulation_all_ID[i], axis = 0)[3, :, -1]) , "-c", label = "Group 4")
        ax.fill_between(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[3], np.nanmean(all_simulation_all_ID[i], axis = 0)[3, :, -1] - np.nanstd(all_simulation_all_ID[i], axis = 0)[3, :, -1] / number_simulation ** 0.5), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[3], np.nanmean(all_simulation_all_ID[i], axis = 0)[3, :, -1] + np.nanstd(all_simulation_all_ID[i], axis = 0)[3, :, -1] / number_simulation ** 0.5), alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
        ax.legend(loc = 'upper right', fontsize = 'medium')
        ax.set_ylim((2, 4))
        ax.set_xticks(range(0, number_model + 1))
        ax.set_xticklabels(['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
        
        t_stat_lp = np.zeros(number_model)
        t_stat_hp = np.zeros(number_model)
        
        p_value_lp = np.zeros(number_model)
        p_value_hp = np.zeros(number_model)
        
        for j in range(number_model):
            t_stat_lp[j], p_value_lp[j] = stats.ttest_ind(all_simulation_all_ID[i, :, 0, j, -1], all_simulation_all_ID[i, :, 1, j, -1], equal_var = True, nan_policy = 'omit')
            t_stat_hp[j], p_value_hp[j] = stats.ttest_ind(all_simulation_all_ID[i, :, 2, j, -1], all_simulation_all_ID[i, :, 3, j, -1], equal_var = True, nan_policy = 'omit')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_ID[i], axis = 0)[1, :, -1])):
            label = 'p = {:.4e}'.format(p_value_lp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'g')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_ID[i], axis = 0)[3, :, -1])):
            label = 'p = {:.4e}'.format(p_value_hp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'c')
            
        all_simulation_group_layer_ID = np.array([])
        
        for j in range(number_group):
            all_simulation_group_layer_ID_temp = np.array([])
            layer_map = []
            
            for k in range(number_model):
                all_simulation_group_layer_ID_temp = np.concatenate(all_simulation_group_layer_ID_temp, all_simulation_all_ID[i, :, j, k, -1].flatten())
                layer_map.append('Layer ' + str(k + 1))
            
            all_simulation_group_layer_ID = np.concatenate(all_simulation_group_layer_ID, all_simulation_group_layer_ID_temp)

        df = pd.DataFrame({'ID': all_simulation_group_layer_ID,
                           'Simulation': np.concatenate((np.tile(np.arange(number_simulation), 2), number_simulation + np.tile(np.arange(number_simulation), 2), 2 * number_simulation + np.tile(np.arange(number_simulation), 2), 3 * number_simulation + np.tile(np.arange(number_simulation), 2))),
                           'Layer': np.tile(np.repeat(layer_map, number_simulation), number_group),
                           'Group': np.concatenate((np.tile(['Group 1'], 2 * number_simulation), np.tile(['Group 2'], 2 * number_simulation), np.tile(['Group 3'], 2 * number_simulation), np.tile(['Group 4'], 2 * number_simulation)))})
        
        aov = pg.mixed_anova(dv = 'ID', within = 'Layer', between = 'Group', subject = 'Simulation', data = df)
        pg.print_table(aov)
        
        posthocs = pg.pairwise_ttests(dv = 'ID', within = 'Layer', between = 'Group', subject = 'Simulation', data = df)
        pg.print_table(posthocs)
                
    fig.savefig(parent_folder + '/Intrinsic Dimension with Correct Labels.png')
    
    # ID across layers and groups for permuted labels
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('Intrinsic Dimension with Permuted Labels', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
        
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_ylabel('ID')
        
        ax.plot(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[0], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[0, :, -1]) , "-b", label = "Group 1")
        ax.fill_between(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[0], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[0, :, -1] - np.nanstd(all_simulation_all_ID_permuted[i], axis = 0)[0, :, -1] / number_simulation ** 0.5), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[0], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[0, :, -1] + np.nanstd(all_simulation_all_ID_permuted[i], axis = 0)[0, :, -1] / number_simulation ** 0.5), alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[1], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[1, :, -1]) , "-g", label = "Group 2")
        ax.fill_between(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[1], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[1, :, -1] - np.nanstd(all_simulation_all_ID_permuted[i], axis = 0)[1, :, -1] / number_simulation ** 0.5), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[1], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[1, :, -1] + np.nanstd(all_simulation_all_ID_permuted[i], axis = 0)[1, :, -1] / number_simulation ** 0.5), alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[2], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[2, :, -1]) , "-r", label = "Group 3")
        ax.fill_between(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[2], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[2, :, -1] - np.nanstd(all_simulation_all_ID_permuted[i], axis = 0)[2, :, -1] / number_simulation ** 0.5), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[2], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[2, :, -1] + np.nanstd(all_simulation_all_ID_permuted[i], axis = 0)[2, :, -1] / number_simulation ** 0.5), alpha = 0.5, edgecolor = 'c', facecolor = 'r')
        
        ax.plot(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[3], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[3, :, -1]) , "-c", label = "Group 4")
        ax.fill_between(range(0, number_model + 1), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[3], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[3, :, -1] - np.nanstd(all_simulation_all_ID_permuted[i], axis = 0)[3, :, -1] / number_simulation ** 0.5), np.append(np.nanmean(all_x_sample_ID[i], axis = 0)[3], np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[3, :, -1] + np.nanstd(all_simulation_all_ID_permuted[i], axis = 0)[3, :, -1] / number_simulation ** 0.5), alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
        ax.legend(loc = 'upper right', fontsize = 'medium')
        ax.set_ylim((2, 4))
        ax.set_xticks(range(number_model + 1))
        ax.set_xticklabels(['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
        
        t_stat_1 = np.zeros(number_model)
        t_stat_2 = np.zeros(number_model)
        t_stat_3 = np.zeros(number_model)
        t_stat_4 = np.zeros(number_model)
        
        p_value_1 = np.zeros(number_model)
        p_value_2 = np.zeros(number_model)
        p_value_3 = np.zeros(number_model)
        p_value_4 = np.zeros(number_model)
        
        for j in range(number_model):
            t_stat_lp[j], p_value_lp[j] = stats.ttest_ind(all_simulation_all_ID[i, :, 0, j, -1], all_simulation_all_ID_permuted[i, :, 0, j, -1], equal_var = True, nan_policy = 'omit')
            t_stat_hp[j], p_value_hp[j] = stats.ttest_ind(all_simulation_all_ID[i, :, 1, j, -1], all_simulation_all_ID_permuted[i, :, 1, j, -1], equal_var = True, nan_policy = 'omit')
            t_stat_lp[j], p_value_lp[j] = stats.ttest_ind(all_simulation_all_ID[i, :, 2, j, -1], all_simulation_all_ID_permuted[i, :, 2, j, -1], equal_var = True, nan_policy = 'omit')
            t_stat_hp[j], p_value_hp[j] = stats.ttest_ind(all_simulation_all_ID[i, :, 3, j, -1], all_simulation_all_ID_permuted[i, :, 3, j, -1], equal_var = True, nan_policy = 'omit')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[0, :, -1])):
            label = 'p = {:.4e}'.format(p_value_lp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'b')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[1, :, -1])):
            label = 'p = {:.4e}'.format(p_value_hp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'g')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[2, :, -1])):
            label = 'p = {:.4e}'.format(p_value_lp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'r')
            
        for j, (x, y) in enumerate(zip(range(0, i + 1), np.nanmean(all_simulation_all_ID_permuted[i], axis = 0)[3, :, -1])):
            label = 'p = {:.4e}'.format(p_value_hp[j])
            ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'c')
            
        all_simulation_group_layer_ID = np.array([])
        
        for j in range(number_group):
            all_simulation_group_layer_ID_temp = np.array([])
            layer_map = []
            
            for k in range(number_model):
                all_simulation_group_layer_ID_temp = np.concatenate(all_simulation_group_layer_ID_temp, all_simulation_all_ID_permuted[i, :, j, k, -1].flatten())
                layer_map.append('Layer ' + str(k + 1))
            
            all_simulation_group_layer_ID = np.concatenate(all_simulation_group_layer_ID, all_simulation_group_layer_ID_temp)

        df = pd.DataFrame({'ID': all_simulation_group_layer_ID,
                           'Simulation': np.concatenate((np.tile(np.arange(number_simulation), 2), number_simulation + np.tile(np.arange(number_simulation), 2), 2 * number_simulation + np.tile(np.arange(number_simulation), 2), 3 * number_simulation + np.tile(np.arange(number_simulation), 2))),
                           'Layer': np.tile(np.repeat(layer_map, number_simulation), number_group),
                           'Group': np.concatenate((np.tile(['Group 1'], 2 * number_simulation), np.tile(['Group 2'], 2 * number_simulation), np.tile(['Group 3'], 2 * number_simulation), np.tile(['Group 4'], 2 * number_simulation)))})
        
        aov = pg.mixed_anova(dv = 'ID', within = 'Layer', between = 'Group', subject = 'Simulation', data = df)
        pg.print_table(aov)
        
        posthocs = pg.pairwise_ttests(dv = 'ID', within = 'Layer', between = 'Group', subject = 'Simulation', data = df)
        pg.print_table(posthocs)
                
    fig.savefig(parent_folder + '/Intrinsic Dimension with Permuted Labels.png')
    
    # Scatter plot of ID in the first layer versus transfer accuracy
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('ID in the First Layer versus Transfer Accuracy', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
        
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_xlabel('% Transfer Accuracy')
        ax.set_ylabel('ID')
        
        x = np.zeros(number_group * number_simulation)
        y = np.zeros(number_group * number_simulation)
        point_label = np.zeros(number_group * number_simulation)
        
        for j in range(number_group):
            x[j * number_simulation:(j + 1) * number_simulation] = all_simulation_transfer_accuracy[i].mean(2)[:, j]
            y[j * number_simulation:(j + 1) * number_simulation] = all_simulation_all_ID[i, :, j, 0, -1]
            point_label[j * number_simulation:(j + 1) * number_simulation] = j
        
        colours = ListedColormap(['b', 'g', 'r', 'c'])
        scatter_legend = ax.scatter(x, y, c = point_label, cmap = colours)
        
        classes = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
        ax.legend(handles = scatter_legend.legend_elements()[0], labels = classes)
        
        slope_12, intercept_12, r_value_12, p_value_12, std_err_12 = stats.linregress(x[0:2 * number_simulation], y[0:2 * number_simulation])
        slope_34, intercept_34, r_value_34, p_value_34, std_err_34 = stats.linregress(x[2 * number_simulation:4 * number_simulation], y[2 * number_simulation:4 * number_simulation])
        
        ax.plot(x[0:2 * number_simulation], intercept_12 + slope_12 * x[0:2 * number_simulation], color = 'g')
        ax.plot(x[2 * number_simulation:4 * number_simulation], intercept_34 + slope_34 * x[2 * number_simulation:4 * number_simulation], color = 'c')
        
        ax.annotate('Group 1&2: Correlation = {:.4e}'.format(r_value_12) + ', p-value = {:.4e}'.format(p_value_12),
                    (x[number_simulation], intercept_12 + slope_12 * x[number_simulation]),
                    textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'g')
        
        ax.annotate('Group 3&4: Correlation = {:.4e}'.format(r_value_34) + ', p-value = {:.4e}'.format(p_value_34),
                    (x[3 * number_simulation], intercept_34 + slope_34 * x[3 * number_simulation]),
                    textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'c')
               
        ax.set_xlim((45, 105))
        ax.set_ylim((3, 3.75))
    
    fig.savefig(parent_folder + '/ID in the First Layer versus Transfer Accuracy.png')
    
    # Scatter plot of ID in the last layer versus transfer accuracy
    
    fig, axs = plt.subplots(1, number_model, figsize = (1 * 8, number_model * 6))
    fig.suptitle('ID in the Last Layer versus Transfer Accuracy', fontsize = 20)
    
    for i in range(number_model):
        ax = axs[0, i]
        
        ax.set_title('DNN Model = ' + str(i + 1), fontsize = 12)
        ax.set_xlabel('% Transfer Accuracy')
        ax.set_ylabel('ID')
        
        x = np.zeros(number_group * number_simulation)
        y = np.zeros(number_group * number_simulation)
        point_label = np.zeros(number_group * number_simulation)
        
        for j in range(number_group):
            x[j * number_simulation:(j + 1) * number_simulation] = all_simulation_transfer_accuracy[i].mean(2)[:, j]
            y[j * number_simulation:(j + 1) * number_simulation] = all_simulation_all_ID[i, :, j, -1, -1]
            point_label[j * number_simulation:(j + 1) * number_simulation] = j
              
        colours = ListedColormap(['b', 'g', 'r', 'c'])
        scatter_legend = ax.scatter(x, y, c = point_label, cmap = colours)
        
        classes = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
        ax.legend(handles = scatter_legend.legend_elements()[0], labels = classes)
        
        slope_12, intercept_12, r_value_12, p_value_12, std_err_12 = stats.linregress(x[0:2 * number_simulation], y[0:2 * number_simulation])
        slope_34, intercept_34, r_value_34, p_value_34, std_err_34 = stats.linregress(x[2 * number_simulation:4 * number_simulation], y[2 * number_simulation:4 * number_simulation])
        
        ax.plot(x[0:2 * number_simulation], intercept_12 + slope_12 * x[0:2 * number_simulation], color = 'g')
        ax.plot(x[2 * number_simulation:4 * number_simulation], intercept_34 + slope_34 * x[2 * number_simulation:4 * number_simulation], color = 'c')
        
        ax.annotate('Group 1&2: Correlation = {:.4e}'.format(r_value_12) + ', p-value = {:.4e}'.format(p_value_12),
                    (x[number_simulation], intercept_12 + slope_12 * x[number_simulation]),
                    textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'g')
        
        ax.annotate('Group 3&4: Correlation = {:.4e}'.format(r_value_34) + ', p-value = {:.4e}'.format(p_value_34),
                    (x[3 * number_simulation], intercept_34 + slope_34 * x[3 * number_simulation]),
                    textcoords = "offset points", xytext = (0, 10), ha = 'center', color = 'c')
        
        ax.set_xlim((45, 105))
        ax.set_ylim((3, 3.75))
    
    fig.savefig(parent_folder + '/ID in the Last Layer versus Transfer Accuracy.png')
    
    # ID across layers and epochs for correct labels for DNN model = 5
    
    DNN_model = 5
    
    fig, axs = plt.subplots(2, 2, figsize = (2 * 8, 2 * 6))
    fig.suptitle('ID versus Layers across Epochs for Correct Labels', fontsize = 20)
    
    for i in range(number_group):
        if i <= 1:
            ax = axs[0, i]
        elif i > 1:
            ax = axs[1, i - 2]
        
        ax.set_title('Group = ' + str(i + 1), fontsize = 12)
        ax.set_ylabel('ID')
        ax.set_ylim((3, 3.75))
        
        ax.set_xticks(range(0, DNN_model))
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
        
        n_lines = 19        
        parameters = np.arange(0, n_lines)
        norm = matplotlib.colors.Normalize(vmin = np.min(parameters), vmax = np.max(parameters))
        
        c_m = matplotlib.cm.cool
        s_m = matplotlib.cm.ScalarMappable(cmap = c_m, norm = norm)
        s_m.set_array([])
        
        for j in range(n_lines):
            x = np.arange(0, DNN_model)
            y = np.nanmean(all_simulation_all_ID[DNN_model - 1], axis = 0)[i, :, j]
            ax.plot(x, y, color = s_m.to_rgba(j))
        
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(s_m, cax = cbar_ax)
    cbar.set_label('number of epochs')
    cbar.set_ticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    cbar.set_ticklabels(['0', '20', '40', '60', '80', '100', '120', '140', '160', '180'])
    
    fig.savefig(parent_folder + '/ID versus Layers across Epochs for Correct Labels.png')
    
    # ID across layers and epochs for permuted labels for DNN Model = 5
    
    DNN_model = 5
    
    fig, axs = plt.subplots(2, 2, figsize = (2 * 8, 2 * 6))
    fig.suptitle('ID versus Layers across Epochs for Permuted Labels', fontsize = 20)
    
    for i in range(number_group):
        if i <= 1:
            ax = axs[0, i]
        elif i > 1:
            ax = axs[1, i - 2]
        
        ax.set_title('Group = ' + str(i + 1), fontsize = 12)
        ax.set_ylabel('ID')
        ax.set_ylim((2.8, 3.75))
        
        ax.set_xticks(range(0, DNN_model))
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
        
        n_lines = 19
        parameters = np.arange(0, n_lines)
        norm = matplotlib.colors.Normalize(vmin = np.min(parameters), vmax = np.max(parameters))
        
        c_m = matplotlib.cm.cool
        s_m = matplotlib.cm.ScalarMappable(cmap = c_m, norm = norm)
        s_m.set_array([])
        
        for j in range(n_lines):
            x = np.arange(0, DNN_model)
            y = np.nanmean(all_simulation_all_ID_permuted[DNN_model - 1], axis = 0)[i, :, j]
            ax.plot(x, y, color = s_m.to_rgba(j))
        
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(s_m, cax = cbar_ax)
    cbar.set_label('number of epochs')
    cbar.set_ticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    cbar.set_ticklabels(['0', '20', '40', '60', '80', '100', '120', '140', '160', '180'])
    
    fig.savefig(parent_folder + '/ID versus Layers across Epochs for Permuted Labels.png')
            
    # ID in the first layer versus training accuracy across epochs for correct labels for DNN Model = 5
    
    DNN_model = 5
    
    fig, axs = plt.subplots(2, 2, figsize = (2 * 8, 2 * 6))
    fig.suptitle('ID in the First Layer versus Training Accuracy across Epochs for Correct Labels', fontsize = 20)
    
    for i in range(number_group):
        if i <= 1:
            ax = axs[0, i]
        elif i > 1:
            ax = axs[1, i - 2]
        
        ax.set_title('Group = ' + str(i + 1), fontsize = 12)
        
        ax.set_xlabel('% Error')
        ax.set_xlim((-5, 55))
        
        ax.set_ylabel('ID')
        ax.set_ylim((3, 3.75))
        
        n_points = 18
        
        x = np.zeros(n_points)
        for j in range(n_points):
            x[j] = 100 - all_simulation_training_accuracy[DNN_model - 1].mean(0)[i, 10 * (j + 1) - 1]
        y = np.nanmean(all_simulation_all_ID[DNN_model - 1], axis = 0)[i, 0, 1:]
        
        color_idx = np.linspace(0, 1, n_points)
        
        cmap = sns.cubehelix_palette(as_cmap = True)
        points = ax.scatter(x, y, c = color_idx, cmap = cmap)
    
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(points, cax = cbar_ax)
    cbar.set_label('number of epochs')
    cbar.set_ticks([0, 2 / 18, 4 / 18, 6 / 18, 8 / 18, 10 / 18, 12 / 18, 14 / 18, 16 / 18, 1])
    cbar.set_ticklabels(['0', '20', '40', '60', '80', '100', '120', '140', '160', '180'])
    
    fig.savefig(parent_folder + '/ID in the First Layer versus Training Accuracy across Epochs for Correct Labels.png')
    
    # ID in the last layer versus training accuracy across epochs for correct labels for DNN Model = 5
    
    DNN_model = 5
    
    fig, axs = plt.subplots(2, 2, figsize = (2 * 8, 2 * 6))
    fig.suptitle('ID in the Last Layer versus Training Accuracy across Epochs for Correct Labels', fontsize = 20)
    
    for i in range(number_group):
        if i <= 1:
            ax = axs[0, i]
        elif i > 1:
            ax = axs[1, i - 2]
        
        ax.set_title('Group = ' + str(i + 1), fontsize = 12)
        
        ax.set_xlabel('% Error')
        ax.set_xlim((-5, 55))
        
        ax.set_ylabel('ID')
        ax.set_ylim((3, 3.75))
        
        n_points = 18
        
        x = np.zeros(n_points)
        for j in range(n_points):
            x[j] = 100 - all_simulation_training_accuracy[DNN_model - 1].mean(0)[i, 10 * (j + 1) - 1]
        y = np.nanmean(all_simulation_all_ID[DNN_model - 1], axis = 0)[i, -1, 1:]
        
        color_idx = np.linspace(0, 1, n_points)
        
        cmap = sns.cubehelix_palette(as_cmap = True)
        points = ax.scatter(x, y, c = color_idx, cmap = cmap)
    
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(points, cax = cbar_ax)
    cbar.set_label('number of epochs')
    cbar.set_ticks([0, 2 / 18, 4 / 18, 6 / 18, 8 / 18, 10 / 18, 12 / 18, 14 / 18, 16 / 18, 1])
    cbar.set_ticklabels(['0', '20', '40', '60', '80', '100', '120', '140', '160', '180'])
    
    fig.savefig(parent_folder + '/ID in the Last Layer versus Training Accuracy across Epochs for Correct Labels.png')
    
    # ID in the first layer versus training accuracy across epochs for permuted labels for DNN Model = 5
    
    DNN_model = 5
    
    fig, axs = plt.subplots(2, 2, figsize = (2 * 8, 2 * 6))
    fig.suptitle('ID in the First Layer versus Training Accuracy across Epochs for Permuted Labels', fontsize = 20)
    
    for i in range(number_group):
        if i <= 1:
            ax = axs[0, i]
        elif i > 1:
            ax = axs[1, i - 2]
        
        ax.set_title('Group = ' + str(i + 1), fontsize = 12)
        
        ax.set_xlabel('% Error')
        ax.set_xlim((-5, 55))
        
        ax.set_ylabel('ID')
        ax.set_ylim((2.8, 3.75))
        
        n_points = 18
        
        x = np.zeros(n_points)
        for j in range(n_points):
            x[j] = 100 - all_simulation_training_accuracy_permuted[DNN_model - 1].mean(0)[i, 10 * (j + 1) - 1]
        y = np.nanmean(all_simulation_all_ID_permuted[DNN_model - 1], axis = 0)[i, 0, 1:]
        
        color_idx = np.linspace(0, 1, n_points)
        
        cmap = sns.cubehelix_palette(as_cmap = True)
        points = ax.scatter(x, y, c = color_idx, cmap = cmap)
    
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(points, cax = cbar_ax)
    cbar.set_label('number of epochs')
    cbar.set_ticks([0, 2 / 18, 4 / 18, 6 / 18, 8 / 18, 10 / 18, 12 / 18, 14 / 18, 16 / 18, 1])
    cbar.set_ticklabels(['0', '20', '40', '60', '80', '100', '120', '140', '160', '180'])
    
    fig.savefig(parent_folder + '/ID in the First Layer versus Training Accuracy across Epochs for Permuted Labels.png')
    
    # ID in the last layer versus training accuracy across epochs for permuted labels for DNN Model = 5
    
    DNN_model = 5
    
    fig, axs = plt.subplots(2, 2, figsize = (2 * 8, 2 * 6))
    fig.suptitle('ID in the Last Layer versus Training Accuracy across Epochs for Permuted Labels', fontsize = 20)
    
    for i in range(number_group):
        if i <= 1:
            ax = axs[0, i]
        elif i > 1:
            ax = axs[1, i - 2]
        
        ax.set_title('Group = ' + str(i + 1), fontsize = 12)
        
        ax.set_xlabel('% Error')
        ax.set_xlim((-5, 55))
        
        ax.set_ylabel('ID')
        ax.set_ylim((2.8, 3.75))
        
        n_points = 18
        
        x = np.zeros(n_points)
        for j in range(n_points):
            x[j] = 100 - all_simulation_training_accuracy_permuted[DNN_model - 1].mean(0)[i, 10 * (j + 1) - 1]
        y = np.nanmean(all_simulation_all_ID_permuted[DNN_model - 1], axis = 0)[i, -1, 1:]
        
        color_idx = np.linspace(0, 1, n_points)
        
        cmap = sns.cubehelix_palette(as_cmap = True)
        points = ax.scatter(x, y, c = color_idx, cmap = cmap)
    
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(points, cax = cbar_ax)
    cbar.set_label('number of epochs')
    cbar.set_ticks([0, 2 / 18, 4 / 18, 6 / 18, 8 / 18, 10 / 18, 12 / 18, 14 / 18, 16 / 18, 1])
    cbar.set_ticklabels(['0', '20', '40', '60', '80', '100', '120', '140', '160', '180'])
    
    fig.savefig(parent_folder + '/ID in the Last Layer versus Training Accuracy across Epochs for Permuted Labels.png')
       
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

def plot_resp_lowd(resp_dict, DNN_model, num_group, num_layer, parent_folder):
    """Plot a low-dimensional representation of each dataset in resp_dict using PCA."""
    
    fig, axs = plt.subplots(num_group, num_layer, figsize = (num_group * 4, num_layer * 4))
    fig.suptitle('Dimensionality Reduction with PCA >>> DNN Model = ' + str(DNN_model + 1), fontsize = 20)
    
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
    
    fig.savefig(parent_folder + '/DR with PCA for DNN Model = ' + str(DNN_model + 1) + '.png')

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