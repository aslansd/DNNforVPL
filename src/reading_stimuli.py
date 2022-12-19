"""
Created by Aslan Satary Dizaji (a.satarydizaji@eni-g.de)
"""

import gc
import glob
import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms

from PIL import Image

def reading_stimuli(file_names, file_name_paths, orientation, spatial_frequency):
       
    # Define the main variables
    x_val = np.zeros((len(spatial_frequency) * len(orientation) * 180, 224, 224, 3), dtype = np.float32)
    y_val = np.zeros((len(spatial_frequency) * len(orientation) * 180, 1), dtype = np.int64)
    z_val = np.zeros((len(spatial_frequency), len(orientation), 180), dtype = np.int64)
    
    x_tensor = []
    y_tensor = []
    
    counter = -1
    
    for i in range(len(file_names)):                 
        
        # Construct the main descriptive variables
        name_digits = file_names[i].split('_')
        
        flag_image_name = False
        
        for j in range(len(spatial_frequency)):
            for k in range(len(orientation)):
                SFplusOri = str(spatial_frequency[j]) + str(orientation[k])
                
                if (SFplusOri) in name_digits[0]:
                    Phase = int(name_digits[0].replace(SFplusOri,''))
                    
                    if Phase % 2 == 1:
                        counter = counter + 1
                        flag_image_name = True
                        
                        if k <= int(len(orientation) / 2 - 1):
                            y_val[counter] = 0
                        else:
                            y_val[counter] = 1
                            
                        z_val[j][k][((Phase + 1) // 2) - 1] = counter
        
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
            x_val[counter, :, :, :] = img[:, :, :]
            
            # Convert image to tensor and normalize
            x_temp = torch.from_numpy(np.transpose(x_val[counter, :, :, :], (2, 0, 1)))
            normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            x_tensor.append(normalize(x_temp))
            
            # Convert target to tensor
            y_tensor.append(torch.from_numpy(y_val[counter]))
        
    x_tensor = torch.stack(x_tensor)
    y_tensor = torch.stack(y_tensor)
    
    return x_val, y_val, z_val, x_tensor, y_tensor