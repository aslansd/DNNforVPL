"""
Created on Thu Feb  4 12:30:02 2021

@author: Aslan
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import scipy.io

import torch
import torchvision.transforms as transforms

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from intrinsic_dimension import estimate
from intrinsic_dimension_GD import estimate_GD
from intrinsic_dimension_LFCIE import center_and_normalize, FCI, fit_FCI

### Initializing the main variables

all_x_sample_ID = np.zeros((2, 4, 10), dtype = np.float32)
all_x_sample_mu_stats = np.zeros((2, 3, 10), dtype = np.float32)
all_x_sample_two_NN_stats = np.zeros((2, 2, 10), dtype = np.float32)

x_sample_artiphysiology_index = []
min_distance_ratio = []

x_sample_artiphysiology_index_temp = []

for i in range(0, 1000):
    x_sample_artiphysiology_index_temp.append([random.randrange(1), random.randrange(20), random.randrange(180)])
    
x_sample_artiphysiology_index.append(x_sample_artiphysiology_index_temp)

for i in range(1, 10):
    x_sample_artiphysiology_index_temp = []
    
    for j in range(0, 20):
        for k in range(0, 180 // i):
            x_sample_artiphysiology_index_temp.append([0, j, k * i])
            
    x_sample_artiphysiology_index.append(x_sample_artiphysiology_index_temp)
        
parent_folder = 'all_stimuli_ID_various_sampling'
os.mkdir(parent_folder)
        
group_counter = -1

for group_training in ['group1', 'group3']:
    group_counter = group_counter + 1
            
    print('Group:   ', group_training)
    
    os.mkdir(parent_folder + '/' + group_training)
    
    # The structure of image names in different groups
    if group_training == 'group1':
        group_transfer = 'group1'
        SF_transfer = [96]
        Ori_transfer = [23325, 23350, 23375, 23400, 23425, 23450, 23475, 23500, 23525, 23550,
                        23650, 23675, 23700, 23725, 23750, 23775, 23800, 23825, 23850, 23875]
        
    elif group_training == 'group3':
        group_transfer = 'group3'
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
    
    # Select GPU
    global device
    gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use GPU: {} for training".format(gpu))
    
    min_distance_ratio_group = []
    
    for i in range(10):       
        print('Sampling_' + str(i))
        
        saving_folder = parent_folder + '/' + group_training + '/Sampling_' + str(i)
        os.mkdir(saving_folder)
                
        all_x_sample = np.zeros((len(x_sample_artiphysiology_index[i]), 3, 224, 224), dtype = np.float32)
                
        for j in range(len(x_sample_artiphysiology_index[i])):           
            index = torch.tensor(z_val_transfer[x_sample_artiphysiology_index[i][j][0], x_sample_artiphysiology_index[i][j][1], x_sample_artiphysiology_index[i][j][2]], dtype = torch.long)
            x_sample = torch.index_select(x_tensor_transfer, 0, index)
            x_sample = x_sample.cuda(gpu)
                        
            all_x_sample[j, :] = x_sample.detach().cpu().clone().numpy()
                                                                       
        # Calculating the intrinsic dimension of stimuli with four different approaches
        linear_dimensionality_PCA = PCA(n_components = 10).fit(all_x_sample.reshape(len(x_sample_artiphysiology_index[i]), -1))
        all_x_sample_ID[group_counter, 0, i] = np.argmax(linear_dimensionality_PCA.explained_variance_ratio_ < 0.05)
        
        pairwise_distance_matrix = squareform(pdist(all_x_sample.reshape(len(x_sample_artiphysiology_index[i]), -1), 'euclidean'))
        all_x_sample_ID[group_counter, 1, i], all_x_sample_two_NN_stats[group_counter, 0, i], all_x_sample_two_NN_stats[group_counter, 1, i] = estimate(pairwise_distance_matrix, fraction = 1.0)[2:]
        
        min_distance_ratio_temp = np.zeros((len(x_sample_artiphysiology_index[i])), dtype = np.float32)
        
        counter_1 = 0
        counter_2 = 0
        counter_3 = 0
               
        for j in range(len(x_sample_artiphysiology_index[i])):            
            sorted_index = np.argsort(pairwise_distance_matrix[j])
            min_distance_ratio_temp[j] = pairwise_distance_matrix[j][sorted_index[2]] / pairwise_distance_matrix[j][sorted_index[1]]
            
            if x_sample_artiphysiology_index[i][sorted_index[2]][0] == x_sample_artiphysiology_index[i][sorted_index[1]][0]:
                counter_1 = counter_1 + 1
            elif x_sample_artiphysiology_index[i][sorted_index[2]][1] == x_sample_artiphysiology_index[i][sorted_index[1]][1]:
                counter_2 = counter_2 + 1
            else:
                counter_3 = counter_3 + 1
                
        counter_sum = counter_1 + counter_2 + counter_3
        
        all_x_sample_mu_stats[group_counter, :, i] = [counter_1 / counter_sum, counter_2 / counter_sum, counter_3 / counter_sum] 
        
        min_distance_ratio_group.append(min_distance_ratio_temp)

        all_x_sample_ID[group_counter, 2, i] = estimate_GD(all_x_sample.reshape(len(x_sample_artiphysiology_index[i]), -1), n_neighbors = 100)[0]

        all_x_sample_ID[group_counter, 3, i] = fit_FCI(FCI(center_and_normalize(all_x_sample.reshape(len(x_sample_artiphysiology_index[i]), -1))))[0]

    min_distance_ratio.append(min_distance_ratio_group)
    
scipy.io.savemat(parent_folder + '/all_x_sample_ID.mat', mdict = {'all_x_sample_ID': all_x_sample_ID})
scipy.io.savemat(parent_folder + '/all_x_sample_mu_stats.mat', mdict = {'all_x_sample_mu_stats': all_x_sample_mu_stats})
scipy.io.savemat(parent_folder + '/all_x_sample_two_NN_stats.mat', mdict = {'all_x_sample_two_NN_stats': all_x_sample_two_NN_stats})
scipy.io.savemat(parent_folder + '/min_distance_ratio.mat', mdict = {'min_distance_ratio': min_distance_ratio})

# all_x_sample_ID = scipy.io.loadmat(parent_folder + '/all_x_sample_ID.mat')['all_x_sample_ID']
# all_x_sample_mu_stats = scipy.io.loadmat(parent_folder + '/all_x_sample_mu_stats.mat')['all_x_sample_mu_stats']
# all_x_sample_two_NN_stats = scipy.io.loadmat(parent_folder + '/all_x_sample_two_NN_stats.mat')['all_x_sample_two_NN_stats']
# min_distance_ratio = scipy.io.loadmat(parent_folder + '/min_distance_ratio.mat')['min_distance_ratio']

# Intrinsic dimensions of stimuli across groups and various samplings

fig, ax = plt.subplots(figsize = (8, 8))
fig.suptitle('Intrinsic Dimensions of Stimuli Using Various Samplings', fontsize = 20)

ax.plot(range(0, 10), all_x_sample_ID[0, 0, :], "r-", label = "Group 1: PCA")
ax.plot(range(0, 10), all_x_sample_ID[0, 1, :], "r--", label = "Group 1: 2-NN")
ax.plot(range(0, 10), all_x_sample_ID[0, 2, :], "r+", label = "Group 1: GD")
ax.plot(range(0, 10), all_x_sample_ID[0, 3, :], "rx", label = "Group 1: LFCIE")

ax.plot(range(0, 10), all_x_sample_ID[1, 0, :], "b-", label = "Group 3: PCA")
ax.plot(range(0, 10), all_x_sample_ID[1, 1, :], "b--", label = "Group 3: 2-NN")
ax.plot(range(0, 10), all_x_sample_ID[1, 2, :], "b+", label = "Group 3: GD")
ax.plot(range(0, 10), all_x_sample_ID[1, 3, :], "bx", label = "Group 3: LFCIE")

for j, (x, y) in enumerate(zip(range(0, 10), all_x_sample_ID[0, 1, :])):
    label = '$R^2$ = {:.4e}'.format(all_x_sample_two_NN_stats[0, 0, j]) + ' & p-value = {:.4e}'.format(all_x_sample_two_NN_stats[0, 1, j])
    ax.annotate(label, (x, y), textcoords = "offset points", xytext = (0, np.power(-1, j) * 20), ha = 'center', color = 'g')

ax.legend(loc = 'upper right', fontsize = 'medium')

ax.set_ylabel('Intrinsic Dimension')
ax.set_ylim((0, 10))
ax.set_xticks(range(0, 10))
ax.set_xticklabels(['Random', '1 Incr.', '2 Incr.', '3 Incr.', '4 Incr.', '5 Incr.', '6 Incr.', '7 Incr.', '8 Incr.', '9 Incr.'])
        
fig.savefig(parent_folder + '/Intrinsic Dimensions of Stimuli Using Various Samplings.png')

# Statistics of minimum distance ratios using various samplings
        
fig, ax = plt.subplots(figsize = (8, 8))
fig.suptitle('Normalized Count of Minimum Distance Ratio Based on the Difference of the Distance Pair Using Various Samplings', fontsize = 20)

ax.plot(range(0, 10), all_x_sample_mu_stats[0, 0, :], "r-", label = "Group 1 & 2-NN Method: Only Different Phase")
ax.plot(range(0, 10), all_x_sample_mu_stats[0, 1, :], "g-", label = "Group 1 & 2-NN Method: Only Different Orientation")
ax.plot(range(0, 10), all_x_sample_mu_stats[0, 2, :], "b-", label = "Group 1 & 2-NN Method: Different Phase & Orientation")

ax.plot(range(0, 10), all_x_sample_mu_stats[1, 0, :], "r--", label = "Group 3 & 2-NN Method: Only Different Phase")
ax.plot(range(0, 10), all_x_sample_mu_stats[1, 1, :], "g--", label = "Group 3 & 2-NN Method: Only Different Orientation")
ax.plot(range(0, 10), all_x_sample_mu_stats[1, 2, :], "b--", label = "Group 3 & 2-NN Method: Different Phase & Orientation")

ax.legend(loc = 'upper right', fontsize = 'medium')

ax.set_ylabel('Normalized Count')
ax.set_ylim((0, 1.1))
ax.set_xticks(range(0, 10))
ax.set_xticklabels(['Random', '1 Incr.', '2 Incr.', '3 Incr.', '4 Incr.', '5 Incr.', '6 Incr.', '7 Incr.', '8 Incr.', '9 Incr.'])
        
fig.savefig(parent_folder + '/Normalized Count of Minimum Distance Ratio Based on the Difference of the Distance Pair Using Various Samplings.png')

# Histograms of minimum distance ratios using various samplings
        
fig, axs = plt.subplots(2, 5, figsize = (2 * 8, 5 * 6))
fig.suptitle('Histograms of Minimum Distance Ratios Using Various Samplings for Group 1', fontsize = 20)

num_bins = 50
counter = -1

for i in range(2):
    for j in range(5):
        counter = counter + 1
        ax = axs[i, j]
        
        # The histogram of data
        min_distance_ratio_not_nan = min_distance_ratio[0][counter][np.isfinite(min_distance_ratio[0][counter])]
        
        n, bins, patches = ax.hist(min_distance_ratio_not_nan, num_bins, density = True)
        
        ax.set_xlabel(r'$\mu$', fontsize = 6)
        ax.set_ylabel('Probability Density', fontsize = 6)
        
        # Add a normal 'best fit' curve
        mean, var = scipy.stats.norm.fit(min_distance_ratio_not_nan)
       
        # Add a pareto 'best fit' curve
        d = scipy.stats.pareto.fit(min_distance_ratio_not_nan)[0]
        y = scipy.stats.pareto.pdf(bins, d)
        ax.plot(bins, y, '--')
        
        ax.set_title('mean = ' + '{:.3f}'.format(mean) + ' & var = ' + '{:.3f}'.format(var) + ' & d = ' + '{:.3f}'.format(d), fontsize = 8)
            
fig.savefig(parent_folder + '/Histograms of Minimum Distance Ratios Using Various Samplings for Group 1.png')

fig, axs = plt.subplots(2, 5, figsize = (2 * 8, 5 * 6))
fig.suptitle('Histograms of Minimum Distance Ratios Using Various Samplings for Group 3', fontsize = 20)

num_bins = 50
counter = -1

for i in range(2):
    for j in range(5):
        counter = counter + 1
        ax = axs[i, j]
        
        # The histogram of data
        min_distance_ratio_not_nan = min_distance_ratio[1][counter][np.isfinite(min_distance_ratio[1][counter])]
        
        n, bins, patches = ax.hist(min_distance_ratio_not_nan, num_bins, density = True)
        
        ax.set_xlabel(r'$\mu$', fontsize = 6)
        ax.set_ylabel('Probability Density', fontsize = 6)
        
        # Add a normal 'best fit' curve
        mean, var = scipy.stats.norm.fit(min_distance_ratio_not_nan)
       
        # Add a pareto 'best fit' curve
        d = scipy.stats.pareto.fit(min_distance_ratio_not_nan)[0]
        y = scipy.stats.pareto.pdf(bins, d)
        ax.plot(bins, y, '--')
        
        ax.set_title('mean = ' + '{:.3f}'.format(mean) + ' & var = ' + '{:.3f}'.format(var) + ' & d = ' + '{:.3f}'.format(d), fontsize = 8)
            
fig.savefig(parent_folder + '/Histograms of Minimum Distance Ratios Using Various Samplings for Group 3.png')