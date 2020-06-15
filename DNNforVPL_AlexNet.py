"""
Created on Tue Oct 22 13:18:11 2019

@author: satarydizaji
"""

import os
import copy
import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import scipy.io
import shutil
import time

import torch
from torch.autograd import grad
import torch.backends.cudnn as cudnn
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid

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
    
    best_acc1 = 0
        
    os.mkdir('New_Results_AlexNet')
    
    for num_simulation in range(0, 50):
        print('Simulation:   ', num_simulation + 1)
        
        os.mkdir('New_Results_AlexNet/Simulation_' + str(num_simulation + 1))
            
        num_sample_artiphysiology = 1000
        x_sample_artiphysiology_index = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
        
        for i in range(0, num_sample_artiphysiology):
            x_sample_artiphysiology_index[i, 0] = random.randrange(6)
            x_sample_artiphysiology_index[i, 1] = random.randrange(6)
            x_sample_artiphysiology_index[i, 2] = random.randrange(360)
        
        for group_training in ['group1', 'group2', 'group3', 'group4']:
            gc.collect()
            
            print('Group:   ', group_training)
            
            os.mkdir('New_Results_AlexNet/Simulation_' + str(num_simulation + 1) + '/' + group_training)
            
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
                Ori_tuning = [23325, 23425, 23525,
                              23650, 23750, 23850]
            
            elif group_training == 'group3' or group_training == 'group4':
                SF_tuning = [33, 53, 140, 170, 340, 480]
                Ori_tuning = [23075, 23175, 23275,
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
                
            for layer_freeze in [None]: # [0, 3, 6, 8, 10]
                print('Freezed Layer:   ', layer_freeze)
                
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
                
                # Initialize the weights of the fully-connected layer of the model
                nn.init.xavier_uniform_(model.classifier[0].weight)
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
                
                # ### ’Artiphysiology’ reveals V4-like shape tuning in a deep network trained for image classification
                
                # # The Convolutional layers: (0, 3, 6, 8, 10)
                # # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # # The central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                
                # if layer_freeze == None:
                #     os.mkdir('New_Results_AlexNet/Simulation_' + str(num_simulation + 1) + '/' + group_training + '/before_training')
                #     saving_folder = 'New_Results_AlexNet/Simulation_' + str(num_simulation + 1) + '/' + group_training + '/before_training'
                    
                #     feature_sample_artiphysiology = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
                    
                #     all_central_unit_activity_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64), dtype = np.float32)
                #     all_central_unit_activity_Conv2d_2 = np.zeros((num_sample_artiphysiology, 192), dtype = np.float32)
                #     all_central_unit_activity_Conv2d_3 = np.zeros((num_sample_artiphysiology, 384), dtype = np.float32)
                #     all_central_unit_activity_Conv2d_4 = np.zeros((num_sample_artiphysiology, 256), dtype = np.float32)
                #     all_central_unit_activity_Conv2d_5 = np.zeros((num_sample_artiphysiology, 256), dtype = np.float32)
                            
                #     for i in range(0, num_sample_artiphysiology):                    
                #         feature_sample_artiphysiology[i, :] = [SF_tuning[x_sample_artiphysiology_index[i, 0]], Ori_tuning[x_sample_artiphysiology_index[i, 1]], x_sample_artiphysiology_index[i, 2]]
                        
                #         index = torch.tensor(z_val_tuning[x_sample_artiphysiology_index[i, 0], x_sample_artiphysiology_index[i, 1], x_sample_artiphysiology_index[i, 2]], dtype = torch.long)
                #         x_sample = torch.index_select(x_tensor_tuning, 0, index)
                #         x_sample = x_sample.cuda(gpu)
                        
                #         unit_activity_layer_0 = model.features[0](x_sample)
                #         unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                #         unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                #         unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                #         unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                #         unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                #         unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                #         unit_activity_layer_7 = model.features[7](unit_activity_layer_6)
                #         unit_activity_layer_8 = model.features[8](unit_activity_layer_7)
                #         unit_activity_layer_9 = model.features[9](unit_activity_layer_8)
                #         unit_activity_layer_10 = model.features[10](unit_activity_layer_9)
                #         unit_activity_layer_11 = model.features[11](unit_activity_layer_10)
                #         unit_activity_layer_12 = model.features[12](unit_activity_layer_11)
                        
                #         all_central_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0, :, 27, 27].detach().cpu().clone().numpy()
                #         all_central_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0, :, 13, 13].detach().cpu().clone().numpy()
                #         all_central_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0, :, 6, 6].detach().cpu().clone().numpy()
                #         all_central_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0, :, 6, 6].detach().cpu().clone().numpy()
                #         all_central_unit_activity_Conv2d_5[i, :] = unit_activity_layer_10[0, :, 6, 6].detach().cpu().clone().numpy()
                        
                #     scipy.io.savemat(saving_folder + '/feature_sample_artiphysiology.mat', mdict = {'feature_sample_artiphysiology': feature_sample_artiphysiology})
                    
                #     scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_1.mat', mdict = {'all_central_unit_activity_Conv2d_1': all_central_unit_activity_Conv2d_1})
                #     scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_2.mat', mdict = {'all_central_unit_activity_Conv2d_2': all_central_unit_activity_Conv2d_2})
                #     scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_3.mat', mdict = {'all_central_unit_activity_Conv2d_3': all_central_unit_activity_Conv2d_3})
                #     scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_4.mat', mdict = {'all_central_unit_activity_Conv2d_4': all_central_unit_activity_Conv2d_4})
                #     scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_5.mat', mdict = {'all_central_unit_activity_Conv2d_5': all_central_unit_activity_Conv2d_5})
                                  
                #     # Boxplotting the tuning curves of central units of three features of convolutional layers
                #     SF_box_central_unit_activity_Conv2d_1 = []
                #     SF_box_central_unit_activity_Conv2d_2 = []
                #     SF_box_central_unit_activity_Conv2d_3 = []
                #     SF_box_central_unit_activity_Conv2d_4 = []
                #     SF_box_central_unit_activity_Conv2d_5 = []
                    
                #     for i in range(0, len(SF_tuning)):                        
                #         SF_box_central_unit_activity_Conv2d_1.append(np.mean(all_central_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #         SF_box_central_unit_activity_Conv2d_2.append(np.mean(all_central_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #         SF_box_central_unit_activity_Conv2d_3.append(np.mean(all_central_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #         SF_box_central_unit_activity_Conv2d_4.append(np.mean(all_central_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #         SF_box_central_unit_activity_Conv2d_5.append(np.mean(all_central_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                        
                #     Ori_box_central_unit_activity_Conv2d_1 = []
                #     Ori_box_central_unit_activity_Conv2d_2 = []
                #     Ori_box_central_unit_activity_Conv2d_3 = []
                #     Ori_box_central_unit_activity_Conv2d_4 = []
                #     Ori_box_central_unit_activity_Conv2d_5 = []
                    
                #     for i in range(0, len(Ori_tuning)):
                #         Ori_box_central_unit_activity_Conv2d_1.append(np.mean(all_central_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #         Ori_box_central_unit_activity_Conv2d_2.append(np.mean(all_central_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #         Ori_box_central_unit_activity_Conv2d_3.append(np.mean(all_central_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #         Ori_box_central_unit_activity_Conv2d_4.append(np.mean(all_central_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #         Ori_box_central_unit_activity_Conv2d_5.append(np.mean(all_central_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                    
                #     Phase_box_central_unit_activity_Conv2d_1 = []
                #     Phase_box_central_unit_activity_Conv2d_2 = []
                #     Phase_box_central_unit_activity_Conv2d_3 = []
                #     Phase_box_central_unit_activity_Conv2d_4 = []
                #     Phase_box_central_unit_activity_Conv2d_5 = []
                    
                #     for i in range(0, 360):
                #         Phase_box_central_unit_activity_Conv2d_1.append(np.mean(all_central_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #         Phase_box_central_unit_activity_Conv2d_2.append(np.mean(all_central_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #         Phase_box_central_unit_activity_Conv2d_3.append(np.mean(all_central_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #         Phase_box_central_unit_activity_Conv2d_4.append(np.mean(all_central_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #         Phase_box_central_unit_activity_Conv2d_5.append(np.mean(all_central_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                        
                #     for feature in ['SF', 'Ori', 'Phase']:
                #         for conv_layer_num in [1, 2, 3, 4, 5]:
                #             plt.figure()
                #             plt.title("%s Boxplot Tuning Curve of the Convolutional Layer %d" % (feature, conv_layer_num))
                #             plt.xlabel(feature)
                #             plt.ylabel("Central Unit Activity")
                #             variable_name = feature + '_box_central_unit_activity_Conv2d_' + str(conv_layer_num)
                #             plt.boxplot(vars()[variable_name])
                #             plt.show()
                #             plt.savefig(saving_folder + '/' + feature + ' Boxplot Tuning Curve of the Convolutional Layer ' + str(conv_layer_num) + '.tif')
                #             plt.close()
                        
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
                    
                    validation_accuracy = np.zeros(epochs, dtype=np.float32)
                    
                    weight_change_1 = np.zeros(epochs, dtype=np.float32)
                    weight_change_2 = np.zeros(epochs, dtype=np.float32)
                    weight_change_3 = np.zeros(epochs, dtype=np.float32)
                    weight_change_4 = np.zeros(epochs, dtype=np.float32)
                    weight_change_5 = np.zeros(epochs, dtype=np.float32)
                            
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
                            validation_accuracy[epoch] = acc1[0].item()
                            
                            # Measure elapsed time
                            batch_time.update(time.time() - end)
                    
                            progress.display(epoch)
                            
                        # Remember the best accuracy
                        is_best = validation_accuracy[epoch] >= best_acc1
                        best_acc1 = max(validation_accuracy[epoch], best_acc1)
                        
                        weight_change_1[epoch] = (torch.pow(torch.sum(torch.pow(model.features[0].weight - Conv2d_1_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[0].weight, 2)), 0.5)).item()
                        weight_change_2[epoch] = (torch.pow(torch.sum(torch.pow(model.features[3].weight - Conv2d_2_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[3].weight, 2)), 0.5)).item()
                        weight_change_3[epoch] = (torch.pow(torch.sum(torch.pow(model.features[6].weight - Conv2d_3_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[6].weight, 2)), 0.5)).item()
                        weight_change_4[epoch] = (torch.pow(torch.sum(torch.pow(model.features[8].weight - Conv2d_4_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[8].weight, 2)), 0.5)).item()
                        weight_change_5[epoch] = (torch.pow(torch.sum(torch.pow(model.features[10].weight - Conv2d_5_0, 2)), 0.5) / torch.pow(torch.sum(torch.pow(model.features[10].weight, 2)), 0.5)).item()       
                           
                os.mkdir('New_Results_AlexNet/Simulation_' + str(num_simulation + 1) + '/' + group_training + '/after_training_' + str(layer_freeze))
                saving_folder = 'New_Results_AlexNet/Simulation_' + str(num_simulation + 1) + '/' + group_training + '/after_training_' + str(layer_freeze)
                
                np.savetxt(saving_folder + '/Training_Accuracy.txt', validation_accuracy, fmt = '%d')
            
                # Plot the validation accuracy vs. number of training sessions
                plt.figure()
                plt.title("Training Accuracy vs. Number of Training Sessions")
                plt.xlabel("Training Sessions")
                plt.ylabel("Training Accuracy")
                plt.plot(range(0, epochs), validation_accuracy)
                plt.ylim((0, 105.))
                plt.xticks(np.arange(-1, epochs + 1, 1.0))
                plt.show()
                plt.savefig(saving_folder + '/Training_Accuracy.tif')
                plt.close()
                
                # Plot the weight change in convolutional layers vs. number of training sessions
                plt.figure()
                plt.title("Weight Change in Convolutional Layers vs. Number of Training Sessions")
                plt.xlabel("Training Sessions")
                plt.ylabel("Weight Change")
                plt.plot(range(0, epochs), weight_change_1, "-b", label = "Conv Layer 1")
                plt.plot(range(0, epochs), weight_change_2, "-g", label = "Conv Layer 2")
                plt.plot(range(0, epochs), weight_change_3, "-r", label = "Conv Layer 3")
                plt.plot(range(0, epochs), weight_change_4, "-c", label = "Conv Layer 4")
                plt.plot(range(0, epochs), weight_change_5, "-m", label = "Conv Layer 5")
                plt.legend(loc="upper left")
                plt.ylim((0, 0.0012))
                plt.xticks(np.arange(-1, epochs + 1, 1.0))
                plt.show()
                plt.savefig(saving_folder + '/Weight_Change.tif')
                plt.close()
                
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
                validation_accuracy = np.zeros((sessions - start_session), dtype = np.float32)
                    
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
                        validation_accuracy[session - start_session] = acc1[0].item()
            
                        # Measure elapsed time
                        batch_time.update(time.time() - end)
            
                        progress.display(1)
                        
                    # Remember the best accuracy and save checkpoint
                    is_best = acc1[0].item() >= best_acc1
                    best_acc1 = max(acc1[0].item(), best_acc1)
                    
                np.savetxt(saving_folder + '/Transfer_Accuracy.txt', validation_accuracy, fmt = '%d')
                                              
                # ### ’Artiphysiology’ reveals V4-like shape tuning in a deep network trained for image classification
                
                # # The Convolutional layers: (0, 3, 6, 8, 10)
                # # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # # The central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                
                # feature_sample_artiphysiology = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
                
                # all_central_unit_activity_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64), dtype = np.float32)
                # all_central_unit_activity_Conv2d_2 = np.zeros((num_sample_artiphysiology, 192), dtype = np.float32)
                # all_central_unit_activity_Conv2d_3 = np.zeros((num_sample_artiphysiology, 384), dtype = np.float32)
                # all_central_unit_activity_Conv2d_4 = np.zeros((num_sample_artiphysiology, 256), dtype = np.float32)
                # all_central_unit_activity_Conv2d_5 = np.zeros((num_sample_artiphysiology, 256), dtype = np.float32)
                        
                # for i in range(0, num_sample_artiphysiology):
                #     feature_sample_artiphysiology[i, :] = [SF_tuning[x_sample_artiphysiology_index[i, 0]], Ori_tuning[x_sample_artiphysiology_index[i, 1]], x_sample_artiphysiology_index[i, 2]]
                    
                #     index = torch.tensor(z_val_tuning[x_sample_artiphysiology_index[i, 0], x_sample_artiphysiology_index[i, 1], x_sample_artiphysiology_index[i, 2]], dtype = torch.long)
                #     x_sample = torch.index_select(x_tensor_tuning, 0, index)
                #     x_sample = x_sample.cuda(gpu)
                    
                #     unit_activity_layer_0 = model.features[0](x_sample)
                #     unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                #     unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                #     unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                #     unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                #     unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                #     unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                #     unit_activity_layer_7 = model.features[7](unit_activity_layer_6)
                #     unit_activity_layer_8 = model.features[8](unit_activity_layer_7)
                #     unit_activity_layer_9 = model.features[9](unit_activity_layer_8)
                #     unit_activity_layer_10 = model.features[10](unit_activity_layer_9)
                #     unit_activity_layer_11 = model.features[11](unit_activity_layer_10)
                #     unit_activity_layer_12 = model.features[12](unit_activity_layer_11)
                    
                #     all_central_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0, :, 27, 27].detach().cpu().clone().numpy()
                #     all_central_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0, :, 13, 13].detach().cpu().clone().numpy()
                #     all_central_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0, :, 6, 6].detach().cpu().clone().numpy()
                #     all_central_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0, :, 6, 6].detach().cpu().clone().numpy()
                #     all_central_unit_activity_Conv2d_5[i, :] = unit_activity_layer_10[0, :, 6, 6].detach().cpu().clone().numpy()
                    
                # scipy.io.savemat(saving_folder + '/feature_sample_artiphysiology.mat', mdict={'feature_sample_artiphysiology': feature_sample_artiphysiology})
                
                # scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_1.mat', mdict = {'all_central_unit_activity_Conv2d_1': all_central_unit_activity_Conv2d_1})
                # scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_2.mat', mdict = {'all_central_unit_activity_Conv2d_2': all_central_unit_activity_Conv2d_2})
                # scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_3.mat', mdict = {'all_central_unit_activity_Conv2d_3': all_central_unit_activity_Conv2d_3})
                # scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_4.mat', mdict = {'all_central_unit_activity_Conv2d_4': all_central_unit_activity_Conv2d_4})
                # scipy.io.savemat(saving_folder + '/all_central_unit_activity_Conv2d_5.mat', mdict = {'all_central_unit_activity_Conv2d_5': all_central_unit_activity_Conv2d_5})
                              
                # # Boxplotting the tuning curves of central units of three features of convolutional layers
                # SF_box_central_unit_activity_Conv2d_1 = []
                # SF_box_central_unit_activity_Conv2d_2 = []
                # SF_box_central_unit_activity_Conv2d_3 = []
                # SF_box_central_unit_activity_Conv2d_4 = []
                # SF_box_central_unit_activity_Conv2d_5 = []
                
                # for i in range(0, len(SF_tuning)):                        
                #     SF_box_central_unit_activity_Conv2d_1.append(np.mean(all_central_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #     SF_box_central_unit_activity_Conv2d_2.append(np.mean(all_central_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #     SF_box_central_unit_activity_Conv2d_3.append(np.mean(all_central_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #     SF_box_central_unit_activity_Conv2d_4.append(np.mean(all_central_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #     SF_box_central_unit_activity_Conv2d_5.append(np.mean(all_central_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                    
                # Ori_box_central_unit_activity_Conv2d_1 = []
                # Ori_box_central_unit_activity_Conv2d_2 = []
                # Ori_box_central_unit_activity_Conv2d_3 = []
                # Ori_box_central_unit_activity_Conv2d_4 = []
                # Ori_box_central_unit_activity_Conv2d_5 = []
                
                # for i in range(0, len(Ori_tuning)):
                #     Ori_box_central_unit_activity_Conv2d_1.append(np.mean(all_central_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #     Ori_box_central_unit_activity_Conv2d_2.append(np.mean(all_central_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #     Ori_box_central_unit_activity_Conv2d_3.append(np.mean(all_central_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #     Ori_box_central_unit_activity_Conv2d_4.append(np.mean(all_central_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #     Ori_box_central_unit_activity_Conv2d_5.append(np.mean(all_central_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                
                # Phase_box_central_unit_activity_Conv2d_1 = []
                # Phase_box_central_unit_activity_Conv2d_2 = []
                # Phase_box_central_unit_activity_Conv2d_3 = []
                # Phase_box_central_unit_activity_Conv2d_4 = []
                # Phase_box_central_unit_activity_Conv2d_5 = []
                
                # for i in range(0, 360):
                #     Phase_box_central_unit_activity_Conv2d_1.append(np.mean(all_central_unit_activity_Conv2d_1[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #     Phase_box_central_unit_activity_Conv2d_2.append(np.mean(all_central_unit_activity_Conv2d_2[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #     Phase_box_central_unit_activity_Conv2d_3.append(np.mean(all_central_unit_activity_Conv2d_3[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #     Phase_box_central_unit_activity_Conv2d_4.append(np.mean(all_central_unit_activity_Conv2d_4[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #     Phase_box_central_unit_activity_Conv2d_5.append(np.mean(all_central_unit_activity_Conv2d_5[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                    
                # for feature in ['SF', 'Ori', 'Phase']:
                #     for conv_layer_num in [1, 2, 3, 4, 5]:
                #         plt.figure()
                #         plt.title("%s Boxplot Tuning Curve of the Convolutional Layer %d" % (feature, conv_layer_num))
                #         plt.xlabel(feature)
                #         plt.ylabel("Central Unit Activity")
                #         variable_name = feature + '_box_central_unit_activity_Conv2d_' + str(conv_layer_num)
                #         plt.boxplot(vars()[variable_name])
                #         plt.show()
                #         plt.savefig(saving_folder + '/' + feature + ' Boxplot Tuning Curve of the Convolutional Layer ' + str(conv_layer_num) + '.tif')
                #         plt.close()
                    
                ### Axiomatic Attribution for Deep Networks
                ### How Important Is a Neuron?
                ### From deep learning to mechanistic understanding in neuroscience: the structure of retinal prediction
                
                # The Convolutional layers: (0, 3, 6, 8, 10)
                # The size of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # The central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                
                feature_sample_artiphysiology = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
                
                all_channel_importance_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64, 55, 55), dtype = np.float32)
                all_channel_importance_Conv2d_2 = np.zeros((num_sample_artiphysiology, 192, 27, 27), dtype = np.float32)
                all_channel_importance_Conv2d_3 = np.zeros((num_sample_artiphysiology, 384, 13, 13), dtype = np.float32)
                all_channel_importance_Conv2d_4 = np.zeros((num_sample_artiphysiology, 256, 13, 13), dtype = np.float32)
                all_channel_importance_Conv2d_5 = np.zeros((num_sample_artiphysiology, 256, 13, 13), dtype = np.float32)
                        
                for i in range(0, num_sample_artiphysiology):
                    torch.cuda.empty_cache()
                    
                    feature_sample_artiphysiology[i, :] = [SF_tuning[x_sample_artiphysiology_index[i, 0]], Ori_tuning[x_sample_artiphysiology_index[i, 1]], x_sample_artiphysiology_index[i, 2]]
                    
                    index = torch.tensor(z_val_tuning[x_sample_artiphysiology_index[i, 0], x_sample_artiphysiology_index[i, 1], x_sample_artiphysiology_index[i, 2]], dtype = torch.long)
                    x_sample = torch.index_select(x_tensor_tuning, 0, index)
                    x_sample = x_sample.squeeze(0).cuda(gpu)
                    
                    x_baseline = torch.index_select(x_tensor_ref, 0, torch.tensor(0, dtype = torch.long))
                    x_baseline = x_baseline.squeeze(0).cuda(gpu)
                    
                    bin_size = 50
                    
                    x_alpha = [x_baseline + (float(j) / bin_size) * (x_sample - x_baseline) for j in range(0, bin_size + 1)]
                    x_alpha = torch.stack(x_alpha)
                    x_alpha.requires_grad = True
                    
                    unit_activity_layer_0 = model.features[0](x_alpha)
                    unit_activity_layer_0.retain_grad()
                    
                    unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                    unit_activity_layer_1.retain_grad()
                    
                    unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                    unit_activity_layer_2.retain_grad()
                    
                    unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                    unit_activity_layer_3.retain_grad()
                    
                    unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                    unit_activity_layer_4.retain_grad()
                    
                    unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                    unit_activity_layer_5.retain_grad()
                    
                    unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                    unit_activity_layer_6.retain_grad()
                    
                    unit_activity_layer_7 = model.features[7](unit_activity_layer_6)
                    unit_activity_layer_7.retain_grad()
                    
                    unit_activity_layer_8 = model.features[8](unit_activity_layer_7)
                    unit_activity_layer_8.retain_grad()
                    
                    unit_activity_layer_9 = model.features[9](unit_activity_layer_8)
                    unit_activity_layer_9.retain_grad()
                    
                    unit_activity_layer_10 = model.features[10](unit_activity_layer_9)
                    unit_activity_layer_10.retain_grad()
                    
                    unit_activity_layer_11 = model.features[11](unit_activity_layer_10)
                    unit_activity_layer_11.retain_grad()
                    
                    unit_activity_layer_12 = model.features[12](unit_activity_layer_11)
                    unit_activity_layer_12.retain_grad()
                    
                    unit_activity_layer_13 = model.avgpool(unit_activity_layer_12)
                    unit_activity_layer_13.retain_grad()
                    
                    unit_activity_layer_14 = torch.flatten(unit_activity_layer_13, 1)
                    unit_activity_layer_14.retain_grad()
                    
                    unit_activity_layer_15 = model.classifier(unit_activity_layer_14)
                    unit_activity_layer_15.retain_grad()
                    
                    for j in range(0, 5):
                        torch.cuda.empty_cache()
                        
                        layers = [0, 3, 6, 8, 10]
                        y_variable_name = 'unit_activity_layer_' + str(layers[j])
                        conv_variable_name = 'all_channel_importance_Conv2d_' + str(j + 1)
                                                   
                        dF_dy = grad(outputs = unit_activity_layer_15, inputs = vars()[y_variable_name], grad_outputs = torch.ones_like(unit_activity_layer_15), retain_graph = True)
                        dy_dx = grad(outputs = vars()[y_variable_name], inputs = x_alpha, grad_outputs = torch.ones_like(vars()[y_variable_name]), retain_graph = True)
                                                    
                        for k in range(1, bin_size + 1):
                            base_1 = torch.sum((x_sample - x_baseline) * torch.index_select(dy_dx[0], 0, torch.tensor(k - 1, dtype = torch.long).cuda(gpu))) * torch.index_select(dF_dy[0], 0, torch.tensor(k - 1, dtype = torch.long).cuda(gpu))
                            base_2 = torch.sum((x_sample - x_baseline) * torch.index_select(dy_dx[0], 0, torch.tensor(k, dtype = torch.long).cuda(gpu))) * torch.index_select(dF_dy[0], 0, torch.tensor(k, dtype = torch.long).cuda(gpu))
                            vars()[conv_variable_name][i] = vars()[conv_variable_name][i] + 1 / bin_size * ((base_1 + base_2) / 2).detach().cpu().clone().numpy()
                    
                scipy.io.savemat(saving_folder + '/feature_sample_artiphysiology.mat', mdict = {'feature_sample_artiphysiology': feature_sample_artiphysiology})
                
                scipy.io.savemat(saving_folder + '/all_channel_importance_Conv2d_1.mat', mdict = {'all_channel_importance_Conv2d_1': all_channel_importance_Conv2d_1})
                scipy.io.savemat(saving_folder + '/all_channel_importance_Conv2d_2.mat', mdict = {'all_channel_importance_Conv2d_2': all_channel_importance_Conv2d_2})
                scipy.io.savemat(saving_folder + '/all_channel_importance_Conv2d_3.mat', mdict = {'all_channel_importance_Conv2d_3': all_channel_importance_Conv2d_3})
                scipy.io.savemat(saving_folder + '/all_channel_importance_Conv2d_4.mat', mdict = {'all_channel_importance_Conv2d_4': all_channel_importance_Conv2d_4})
                scipy.io.savemat(saving_folder + '/all_channel_importance_Conv2d_5.mat', mdict = {'all_channel_importance_Conv2d_5': all_channel_importance_Conv2d_5})
                              
                # # Boxplotting the channel importance of three features of convolutional layer
                # SF_box_channel_importance_Conv2d_1 = []
                # SF_box_channel_importance_Conv2d_2 = []
                # SF_box_channel_importance_Conv2d_3 = []
                # SF_box_channel_importance_Conv2d_4 = []
                # SF_box_channel_importance_Conv2d_5 = []
                
                # for i in range(0, len(SF_tuning)):                        
                #     SF_box_channel_importance_Conv2d_1.append(np.mean(all_channel_importance_Conv2d_1[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #     SF_box_channel_importance_Conv2d_2.append(np.mean(all_channel_importance_Conv2d_2[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #     SF_box_channel_importance_Conv2d_3.append(np.mean(all_channel_importance_Conv2d_3[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #     SF_box_channel_importance_Conv2d_4.append(np.mean(all_channel_importance_Conv2d_4[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                #     SF_box_channel_importance_Conv2d_5.append(np.mean(all_channel_importance_Conv2d_5[feature_sample_artiphysiology[:, 0] == SF_tuning[i], :], axis = 1))
                    
                # Ori_box_channel_importance_Conv2d_1 = []
                # Ori_box_channel_importance_Conv2d_2 = []
                # Ori_box_channel_importance_Conv2d_3 = []
                # Ori_box_channel_importance_Conv2d_4 = []
                # Ori_box_channel_importance_Conv2d_5 = []
                
                # for i in range(0, len(Ori_tuning)):
                #     Ori_box_channel_importance_Conv2d_1.append(np.mean(all_channel_importance_Conv2d_1[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #     Ori_box_channel_importance_Conv2d_2.append(np.mean(all_channel_importance_Conv2d_2[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #     Ori_box_channel_importance_Conv2d_3.append(np.mean(all_channel_importance_Conv2d_3[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #     Ori_box_channel_importance_Conv2d_4.append(np.mean(all_channel_importance_Conv2d_4[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                #     Ori_box_channel_importance_Conv2d_5.append(np.mean(all_channel_importance_Conv2d_5[feature_sample_artiphysiology[:, 1] == Ori_tuning[i], :], axis = 1))
                
                # Phase_box_channel_importance_Conv2d_1 = []
                # Phase_box_channel_importance_Conv2d_2 = []
                # Phase_box_channel_importance_Conv2d_3 = []
                # Phase_box_channel_importance_Conv2d_4 = []
                # Phase_box_channel_importance_Conv2d_5 = []
                
                # for i in range(0, 360):
                #     Phase_box_channel_importance_Conv2d_1.append(np.mean(all_channel_importance_Conv2d_1[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #     Phase_box_channel_importance_Conv2d_2.append(np.mean(all_channel_importance_Conv2d_2[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #     Phase_box_channel_importance_Conv2d_3.append(np.mean(all_channel_importance_Conv2d_3[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #     Phase_box_channel_importance_Conv2d_4.append(np.mean(all_channel_importance_Conv2d_4[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                #     Phase_box_channel_importance_Conv2d_5.append(np.mean(all_channel_importance_Conv2d_5[feature_sample_artiphysiology[:, 2] == i + 1, :], axis = 1))
                    
                # for feature in ['SF', 'Ori', 'Phase']:
                #     for conv_layer_num in [1, 2, 3, 4, 5]:
                #         plt.figure()
                #         plt.title("%s Boxplot Channel Importance of the Convolutional Layer %d" % (feature, conv_layer_num))
                #         plt.xlabel(feature)
                #         plt.ylabel("Channel Importance")
                #         variable_name = feature + '_box_channel_importance_Conv2d_' + str(conv_layer_num)
                #         plt.boxplot(vars()[variable_name])
                #         plt.show()
                #         plt.savefig(saving_folder + '/' + feature + ' Boxplot Channel Importance of the Convolutional Layer ' + str(conv_layer_num) + '.tif')
                #         plt.close()
                        
                # ### Visualizing and Understanding Convolutional Networks
                
                # # Reading the grey background image
                # file_name_path_ref = glob.glob('VPL Stimuli/greybackground.TIFF')
                # img = Image.open(file_name_path_ref[0]).convert('RGB')
                
                # x_val_greybackground = np.zeros((224, 224, 3), dtype = np.float32)
                # x_tensor_greybackground = []
                
                # width, height = img.size
                # new_width = width * 256 // min(img.size)
                # new_height = height * 256 // min(img.size)
                # img = img.resize((new_width, new_height), Image.BILINEAR)
                
                # width, height = img.size
                # startx = width // 2 - (224 // 2)
                # starty = height // 2 - (224 // 2)
                # img = np.asarray(img).reshape(height, width, 3)
                # img = img[starty:starty + 224, startx:startx + 224]
                # assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)
                
                # x_val_greybackground[:, :, :] = img[:, :, :]
                # x_temp = torch.from_numpy(np.transpose(x_val_greybackground[:, :, :], (2, 0, 1)))
                # normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                # x_tensor_greybackground.append(normalize(x_temp))
                # x_tensor_greybackground = torch.stack(x_tensor_greybackground)
                
                # print(x_tensor_greybackground.shape)
                
                # x_img_difference_SF = np.zeros((1000, 5), dtype = np.float32)
                # if group_training == 'group2' or group_training == 'group4':
                #     x_img_difference_Ori = np.zeros((1000, 5), dtype = np.float32)
                    
                # for i in range(0, 1000):
                #     # Get two sample tensors of training/validation images with the same phase and spatial frequency but different orientation to visualize and quantify the convolutional networks
                #     SF_index = random.randrange(len(SF_training))
                #     Ori_index_1 = random.randrange(int(len(Ori_training) / 2))
                #     Ori_index_2 = random.randrange(int(len(Ori_training) / 2), len(Ori_training))
                #     Phase_index = random.randrange(180)
                    
                #     indices = torch.tensor(np.array([z_val_training[SF_index, Ori_index_1, Phase_index], z_val_training[SF_index, Ori_index_2, Phase_index]]), dtype = torch.long)
                #     x_sample = torch.index_select(x_tensor_training, 0, indices)
                #     y_title = ['SF = ' + str(SF_training[SF_index]) + ' ***** ' + 'Ori = ' + str(Ori_training[Ori_index_1]) + ' ***** ' + 'Ph = ' + str(Phase_index),
                #               'SF = ' + str(SF_training[SF_index]) + ' ***** ' + 'Ori = ' + str(Ori_training[Ori_index_2]) + ' ***** ' + 'Ph = ' + str(Phase_index)]
                
                #     visualize_layer_indices = [2, 5, 7, 9, 12]
                    
                #     for layer in model.features:
                #         if isinstance(layer, torch.nn.MaxPool2d):
                #             layer.return_indices = True
                    
                #     x_img_greybackground = []
                #     x_img = [x_sample[0], x_sample[1]]
                    
                #     for layer_max_count in visualize_layer_indices:
                #         x_tensor_greybackground.squeeze_(0)
                        
                #         raw_feature_maps, deconv_layers_list, unpool_layers_list = forward_img(gpu, model, x_tensor_greybackground, layer_max_count)
                #         x_img_greybackground_temp = backward_feature_maps(raw_feature_maps, deconv_layers_list, unpool_layers_list)
                        
                #         for i in range(0, x_sample.size(0)):
                #             print("layer...%s" % layer_max_count)
                            
                #             x_img_greybackground.append(x_img_greybackground_temp)
                                                  
                #             raw_feature_maps, deconv_layers_list, unpool_layers_list = forward_img(gpu, model, x_sample[i], layer_max_count)
                #             x_img.append(backward_feature_maps(raw_feature_maps, deconv_layers_list, unpool_layers_list))
                            
                #     x_img_difference_SF[i, :] = x_img_difference_SF[i, :] + visualize(x_img_greybackground, x_img, y_title)
                    
                #     # Get two sample tensors of training/validation images with the same phase and orientation but different spatial frequency to visualize and quantify the convolutional networks
                #     if group_training == 'group2' or group_training == 'group4':      
                #         SF_index = np.array(range(0, len(SF_training)))
                #         random.shuffle(SF_index)
                #         SF_index_1 = SF_index[0]
                #         SF_index_2 = SF_index[1]
                #         Ori_index = random.randrange(len(Ori_training))
                #         Phase_index = random.randrange(180)
                        
                #         indices = torch.tensor(np.array([z_val_training[SF_index_1, Ori_index, Phase_index], z_val_training[SF_index_2, Ori_index, Phase_index]]), dtype = torch.long)
                #         x_sample = torch.index_select(x_tensor_training, 0, indices)
                #         y_title = ['SF = ' + str(SF_training[SF_index_1]) + ' ***** ' + 'Ori = ' + str(Ori_training[Ori_index]) + ' ***** ' + 'Ph = ' + str(Phase_index),
                #                   'SF = ' + str(SF_training[SF_index_2]) + ' ***** ' + 'Ori = ' + str(Ori_training[Ori_index]) + ' ***** ' + 'Ph = ' + str(Phase_index)]
                    
                #         visualize_layer_indices = [2, 5, 7, 9, 12]
                        
                #         for layer in model.features:
                #             if isinstance(layer, torch.nn.MaxPool2d):
                #                 layer.return_indices = True
                        
                #         x_img_greybackground = []
                #         x_img = [x_sample[0], x_sample[1]]
                        
                #         for layer_max_count in visualize_layer_indices:
                #             x_tensor_greybackground.squeeze_(0)
                            
                #             raw_feature_maps, deconv_layers_list, unpool_layers_list = forward_img(gpu, model, x_tensor_greybackground, layer_max_count)
                #             x_img_greybackground_temp = backward_feature_maps(raw_feature_maps, deconv_layers_list, unpool_layers_list)
                            
                #             for i in range(0, x_sample.size(0)):
                #                 print("layer...%s" % layer_max_count)
                                
                #                 x_img_greybackground.append(x_img_greybackground_temp)
                                                      
                #                 raw_feature_maps, deconv_layers_list, unpool_layers_list = forward_img(gpu, model, x_sample[i], layer_max_count)
                #                 x_img.append(backward_feature_maps(raw_feature_maps, deconv_layers_list, unpool_layers_list))
                                
                #         x_img_difference_Ori[i, :] = x_img_difference_Ori[i, :] + visualize(x_img_greybackground, x_img, y_title)
                
                # plt.figure()
                # plt.title("Euclidean Distance of Mapped Pixles in Consecutive Layers (Constant SF)")
                # plt.xlabel("Convolutional Layer Number")
                # plt.ylabel("Euclidean Distance")
                # plt.errorbar(range(1, 6), np.mean(x_img_difference_SF, axis = 0), yerr = np.std(x_img_difference_SF, axis = 0))
                # plt.xticks(np.arange(1, 6, 1.0))
                # plt.show()
                
                # if group_training == 'group2' or group_training == 'group4':
                #     plt.figure()
                #     plt.title("Euclidean Distance of Mapped Pixles in Consecutive Layers (Constant Ori)")
                #     plt.xlabel("Convolutional Layer Number")
                #     plt.ylabel("Euclidean Distance")
                #     plt.errorbar(range(1, 6), np.mean(x_img_difference_Ori, axis = 0), yerr = np.std(x_img_difference_Ori, axis = 0))
                #     plt.xticks(np.arange(1, 6, 1.0))
                #     plt.show()
    
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

def forward_img(gpu, model, x, layer_max_count):
    deconv_layers_list = []
    unpool_layers_list = []

    layer_count = 0
    x.unsqueeze_(0)
    x = x.cuda(gpu)

    for layer in model.features:
        if isinstance(layer, torch.nn.Conv2d):
            B, C, H, W = x.shape
            x = layer(x)
            deconv_layer = nn.ConvTranspose2d(layer.out_channels, C, layer.kernel_size, layer.stride, layer.padding)
            deconv_layer.weight = layer.weight
            deconv_layers_list.append(deconv_layer.cuda(gpu))

        if isinstance(layer, torch.nn.ReLU):
            x = layer(x)
            deconv_layers_list.append(layer.cuda(gpu))

        if isinstance(layer, torch.nn.MaxPool2d):
            x, index = layer(x)
            unpool_layers_list.append(index.cuda(gpu))
            unpool_layer = torch.nn.MaxUnpool2d(kernel_size = layer.kernel_size, stride = layer.stride, padding = layer.padding)
            deconv_layers_list.append(unpool_layer.cuda(gpu))

        layer_count += 1
        if layer_count == layer_max_count:
            break

    return x, deconv_layers_list, unpool_layers_list

def backward_feature_maps(y, deconv_layers_list, unpool_layers_list):
    for layer in reversed(deconv_layers_list):
        if isinstance(layer, nn.MaxUnpool2d):
            y = layer(y, unpool_layers_list.pop())
        else:
            y = layer(y)

    return y.squeeze_(0)

def visualize(x_img_greybackground, x_img, title):
    for i in range(0, len(x_img_greybackground)):
        x_img_greybackground[i] = x_img_greybackground[i].cpu()

    for i in range(0, len(x_img)):
        x_img[i] = x_img[i].cpu()       
    
    # x_img_input = make_grid(x_img[0:len(title)], nrow = len(title))
    # x_img_input = x_img_input.detach().numpy().transpose((1, 2, 0)) 
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # x_img_input = (std * x_img_input + mean) / 255.0
    # x_img_input = np.clip(x_img_input, 0, 1)
    # plt.figure()
    # plt.imshow(x_img_input)
    # plt.title(title)
    
    x_img_layers = []
    for i in range(len(title), len(x_img)):
        x_img_layers.append((x_img[i] - x_img[i].min()) * 255 / (x_img[i].max() - x_img[i].min()))
        
    # x_img_1 = make_grid(x_img_layers, nrow = len(title))
    # x_img_1 = x_img_1.detach().numpy().transpose((1, 2, 0)).astype('uint8')
    # plt.figure()
    # plt.imshow(x_img_1)
    # plt.title(title)
    
    # x_img_layers_background = []
    # for i in range(len(title), len(x_img)):
    #     x_img_temp = x_img[i] - x_img_greybackground[i - len(title)]
    #     x_img_layers_background.append((x_img_temp - x_img_temp.min()) * 255 / (x_img_temp.max() - x_img_temp.min()))
    
    # x_img_2 = make_grid(x_img_layers_background, nrow = len(title))
    # x_img_2 = x_img_2.detach().numpy().transpose((1, 2, 0)).astype('uint8')
    # plt.figure()
    # plt.imshow(x_img_2)
    # plt.title(title)
   
    x_img_difference = []
    counter = -1
    x_img_difference = np.zeros(5, dtype = np.float32)
    
    for i in range(len(title), len(x_img), 2):
        counter = counter + 1
        # x_img_difference[counter] = (x_img_layers[i - len(title) + 1] - x_img_layers[i - len(title)]).pow(2).sum().pow(0.5).detach().numpy()
        x_img_difference[counter] = (x_img[i + 1] - x_img[i]).pow(2).sum().pow(0.5).detach().numpy()
    
    # plt.figure()
    # plt.title("Euclidean Distance of Mapped Pixles in Consecutive Layers")
    # plt.xlabel("Convolutional Layer Number")
    # plt.ylabel("Euclidean Distance")
    # plt.plot(range(1, 6), x_img_difference)
    # plt.xticks(np.arange(1, 6, 1.0))
    # plt.show()
        
    return x_img_difference
    
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