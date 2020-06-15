"""
Created on Mon Mar  2 14:09:29 2020

@author: satarydizaji
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat, savemat

os.mkdir('P:/ENI Projects/Caspar/Python Programs/PyTorch/Tuning_Analysis_Results')
saving_folder_1 = 'P:/ENI Projects/Caspar/Python Programs/PyTorch/Tuning_Analysis_Results'

all_groups_central_unit_activity_mean_channels_SF_analysis_layer_1 = np.zeros((4, 3))
all_groups_central_unit_activity_mean_channels_SF_analysis_layer_2 = np.zeros((4, 3))
all_groups_central_unit_activity_mean_channels_SF_analysis_layer_3 = np.zeros((4, 3))
all_groups_central_unit_activity_mean_channels_SF_analysis_layer_4 = np.zeros((4, 3))
all_groups_central_unit_activity_mean_channels_SF_analysis_layer_5 = np.zeros((4, 3))

all_groups_central_unit_activity_std_channels_SF_analysis_layer_1 = np.zeros((4, 3))
all_groups_central_unit_activity_std_channels_SF_analysis_layer_2 = np.zeros((4, 3))
all_groups_central_unit_activity_std_channels_SF_analysis_layer_3 = np.zeros((4, 3))
all_groups_central_unit_activity_std_channels_SF_analysis_layer_4 = np.zeros((4, 3))
all_groups_central_unit_activity_std_channels_SF_analysis_layer_5 = np.zeros((4, 3))

all_groups_central_unit_activity_SF_ttest_layer_1 = np.zeros((4, 3, 100))
all_groups_central_unit_activity_SF_ttest_layer_2 = np.zeros((4, 3, 100))
all_groups_central_unit_activity_SF_ttest_layer_3 = np.zeros((4, 3, 100))
all_groups_central_unit_activity_SF_ttest_layer_4 = np.zeros((4, 3, 100))
all_groups_central_unit_activity_SF_ttest_layer_5 = np.zeros((4, 3, 100))

all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_1 = np.zeros((4, 4))
all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_2 = np.zeros((4, 4))
all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_3 = np.zeros((4, 4))
all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_4 = np.zeros((4, 4))
all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_5 = np.zeros((4, 4))

all_groups_central_unit_activity_std_channels_Ori_analysis_layer_1 = np.zeros((4, 4))
all_groups_central_unit_activity_std_channels_Ori_analysis_layer_2 = np.zeros((4, 4))
all_groups_central_unit_activity_std_channels_Ori_analysis_layer_3 = np.zeros((4, 4))
all_groups_central_unit_activity_std_channels_Ori_analysis_layer_4 = np.zeros((4, 4))
all_groups_central_unit_activity_std_channels_Ori_analysis_layer_5 = np.zeros((4, 4))

all_groups_central_unit_activity_Ori_ttest_layer_1 = np.zeros((4, 4, 100))
all_groups_central_unit_activity_Ori_ttest_layer_2 = np.zeros((4, 4, 100))
all_groups_central_unit_activity_Ori_ttest_layer_3 = np.zeros((4, 4, 100))
all_groups_central_unit_activity_Ori_ttest_layer_4 = np.zeros((4, 4, 100))
all_groups_central_unit_activity_Ori_ttest_layer_5 = np.zeros((4, 4, 100))

group_counter = -1

for group_training in ['group1', 'group2', 'group3', 'group4']:
    print(group_training)
    
    os.mkdir(saving_folder_1 + '/' + group_training)
    saving_folder_2 = saving_folder_1 + '/' + group_training
        
    if group_training == 'group1' or group_training == 'group2':
        SF_tuning = [33, 140, 310]
        Ori_tuning = [23475, 23550,
                      23650, 23725]
        
    elif group_training == 'group3' or group_training == 'group4':
        SF_tuning = [33, 140, 310]
        Ori_tuning = [23225, 23300,
                      23900, 23975]
    
    conv_layer_channel = [64, 192, 384, 256, 256]
    
    central_unit_activity_all_channels_analysis_layer_1 = np.zeros((100, conv_layer_channel[0], len(SF_tuning), len(Ori_tuning)))
    central_unit_activity_all_channels_analysis_layer_2 = np.zeros((100, conv_layer_channel[1], len(SF_tuning), len(Ori_tuning)))
    central_unit_activity_all_channels_analysis_layer_3 = np.zeros((100, conv_layer_channel[2], len(SF_tuning), len(Ori_tuning)))
    central_unit_activity_all_channels_analysis_layer_4 = np.zeros((100, conv_layer_channel[3], len(SF_tuning), len(Ori_tuning)))
    central_unit_activity_all_channels_analysis_layer_5 = np.zeros((100, conv_layer_channel[4], len(SF_tuning), len(Ori_tuning)))
    
    for num_simulation in range(0, 100):
        print(num_simulation)
          
        data_folder_before = 'P:/ENI Projects/Caspar/Python Programs/PyTorch/100 Simulations_Central Units of Channels per Layer/Simulation_' + str(num_simulation + 1) + '/' + group_training + '/before_training'
        data_folder_after = 'P:/ENI Projects/Caspar/Python Programs/PyTorch/100 Simulations_Central Units of Channels per Layer/Simulation_' + str(num_simulation + 1) + '/' + group_training + '/after_training'
        
        data_feature = loadmat(data_folder_before + '/feature_sample_artiphysiology.mat')
        data_feature_matrix = data_feature['feature_sample_artiphysiology']
        
        data_layer_before = loadmat(data_folder_before + '/all_mean_unit_activity_Conv2d_1.mat')
        data_layer_after = loadmat(data_folder_after + '/all_mean_unit_activity_Conv2d_1.mat')
        central_unit_activity_all_channels_layer_1 = (data_layer_after['all_mean_unit_activity_Conv2d_1'] - data_layer_before['all_mean_unit_activity_Conv2d_1']).T
        
        data_layer_before = loadmat(data_folder_before + '/all_mean_unit_activity_Conv2d_2.mat')
        data_layer_after = loadmat(data_folder_after + '/all_mean_unit_activity_Conv2d_2.mat')
        central_unit_activity_all_channels_layer_2 = (data_layer_after['all_mean_unit_activity_Conv2d_2'] - data_layer_before['all_mean_unit_activity_Conv2d_2']).T
        
        data_layer_before = loadmat(data_folder_before + '/all_mean_unit_activity_Conv2d_3.mat')
        data_layer_after = loadmat(data_folder_after + '/all_mean_unit_activity_Conv2d_3.mat')
        central_unit_activity_all_channels_layer_3 = (data_layer_after['all_mean_unit_activity_Conv2d_3'] - data_layer_before['all_mean_unit_activity_Conv2d_3']).T
        
        data_layer_before = loadmat(data_folder_before + '/all_mean_unit_activity_Conv2d_4.mat')
        data_layer_after = loadmat(data_folder_after + '/all_mean_unit_activity_Conv2d_4.mat')
        central_unit_activity_all_channels_layer_4 = (data_layer_after['all_mean_unit_activity_Conv2d_4'] - data_layer_before['all_mean_unit_activity_Conv2d_4']).T
        
        data_layer_before = loadmat(data_folder_before + '/all_mean_unit_activity_Conv2d_5.mat')
        data_layer_after = loadmat(data_folder_after + '/all_mean_unit_activity_Conv2d_5.mat')
        central_unit_activity_all_channels_layer_5 = (data_layer_after['all_mean_unit_activity_Conv2d_5'] - data_layer_before['all_mean_unit_activity_Conv2d_5']).T
               
        for i in range(0, len(SF_tuning)):
            for j in range(0, len(Ori_tuning)):
                indices = np.intersect1d(np.where(data_feature_matrix[:, 0] == SF_tuning[i]), np.where(data_feature_matrix[:, 1] == Ori_tuning[j]))
                
                for k in range(0, len(indices)):
                    central_unit_activity_all_channels_analysis_layer_1[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_1[num_simulation, :, i, j] + central_unit_activity_all_channels_layer_1[:, indices[k]]
                    central_unit_activity_all_channels_analysis_layer_2[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_2[num_simulation, :, i, j] + central_unit_activity_all_channels_layer_2[:, indices[k]]
                    central_unit_activity_all_channels_analysis_layer_3[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_3[num_simulation, :, i, j] + central_unit_activity_all_channels_layer_3[:, indices[k]]
                    central_unit_activity_all_channels_analysis_layer_4[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_4[num_simulation, :, i, j] + central_unit_activity_all_channels_layer_4[:, indices[k]]
                    central_unit_activity_all_channels_analysis_layer_5[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_5[num_simulation, :, i, j] + central_unit_activity_all_channels_layer_5[:, indices[k]]
 
                central_unit_activity_all_channels_analysis_layer_1[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_1[num_simulation, :, i, j] / len(indices)
                central_unit_activity_all_channels_analysis_layer_2[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_2[num_simulation, :, i, j] / len(indices)
                central_unit_activity_all_channels_analysis_layer_3[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_3[num_simulation, :, i, j] / len(indices)
                central_unit_activity_all_channels_analysis_layer_4[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_4[num_simulation, :, i, j] / len(indices)
                central_unit_activity_all_channels_analysis_layer_5[num_simulation, :, i, j] = central_unit_activity_all_channels_analysis_layer_5[num_simulation, :, i, j] / len(indices)          
    
    indices_1 = []
    for i in range(0, conv_layer_channel[0]):
        if np.all(central_unit_activity_all_channels_analysis_layer_1[:, i, :, :] == 0) == False:
            indices_1.append(i)
    central_unit_activity_all_channels_analysis_layer_1 = central_unit_activity_all_channels_analysis_layer_1[:, indices_1, :, :]
    
    indices_2 = []
    for i in range(0, conv_layer_channel[1]):
        if np.all(central_unit_activity_all_channels_analysis_layer_2[:, i, :, :] == 0) == False:
            indices_2.append(i)
    central_unit_activity_all_channels_analysis_layer_2 = central_unit_activity_all_channels_analysis_layer_2[:, indices_2, :, :]
    
    indices_3 = []
    for i in range(0, conv_layer_channel[2]):
        if np.all(central_unit_activity_all_channels_analysis_layer_3[:, i, :, :] == 0) == False:
            indices_3.append(i)
    central_unit_activity_all_channels_analysis_layer_3 = central_unit_activity_all_channels_analysis_layer_3[:, indices_3, :, :]
    
    indices_4 = []
    for i in range(0, conv_layer_channel[3]):
        if np.all(central_unit_activity_all_channels_analysis_layer_4[:, i, :, :] == 0) == False:
            indices_4.append(i)
    central_unit_activity_all_channels_analysis_layer_4 = central_unit_activity_all_channels_analysis_layer_4[:, indices_4, :, :]
    
    indices_5 = []
    for i in range(0, conv_layer_channel[4]):
        if np.all(central_unit_activity_all_channels_analysis_layer_5[:, i, :, :] == 0) == False:
            indices_5.append(i)
    central_unit_activity_all_channels_analysis_layer_5 = central_unit_activity_all_channels_analysis_layer_5[:, indices_5, :, :]
      
    savemat(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_1.mat', mdict={'central_unit_activity_all_channels_analysis_layer_1': central_unit_activity_all_channels_analysis_layer_1})
    savemat(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_2.mat', mdict={'central_unit_activity_all_channels_analysis_layer_2': central_unit_activity_all_channels_analysis_layer_2})
    savemat(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_3.mat', mdict={'central_unit_activity_all_channels_analysis_layer_3': central_unit_activity_all_channels_analysis_layer_3})
    savemat(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_4.mat', mdict={'central_unit_activity_all_channels_analysis_layer_4': central_unit_activity_all_channels_analysis_layer_4})
    savemat(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_5.mat', mdict={'central_unit_activity_all_channels_analysis_layer_5': central_unit_activity_all_channels_analysis_layer_5})
       
    #Convolutional Layer 1:    
    plt.figure()
    plt.title("Central Unit Activity of all Channels in Convolutional Layer 1 of Different Stimuli Conditions")
    plt.xlabel("Channel Number")
    plt.ylabel("Central Unit Activity")
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 0, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 0, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 0, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 0, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 0, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 0, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 0, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 0, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 1, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 1, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 1, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 1, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 1, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 1, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 1, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 1, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 2, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 2, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 2, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 2, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 2, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 2, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_1.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, 2, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, 2, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 3')
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_1.tif')
    
    #Convolutional Layer 2:
    plt.figure()
    plt.title("Central Unit Activity of all Channels in Convolutional Layer 2 of Different Stimuli Conditions")
    plt.xlabel("Channel Number")
    plt.ylabel("Central Unit Activity")
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 0, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 0, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 0, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 0, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 0, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 0, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 0, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 0, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 1, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 1, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 1, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 1, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 1, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 1, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 1, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 1, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 2, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 2, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 2, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 2, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 2, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 2, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_2.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, 2, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, 2, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 3')
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_2.tif')
    
    #Convolutional Layer 3:
    plt.figure()
    plt.title("Central Unit Activity of all Channels in Convolutional Layer 3 of Different Stimuli Conditions")
    plt.xlabel("Channel Number")
    plt.ylabel("Central Unit Activity")
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 0, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 0, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 0, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 0, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 0, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 0, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 0, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 0, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 1, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 1, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 1, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 1, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 1, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 1, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 1, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 1, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 2, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 2, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 2, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 2, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 2, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 2, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_3.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, 2, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, 2, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 3')
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_3.tif')
    
    #Convolutional Layer 4:
    plt.figure()
    plt.title("Central Unit Activity of all Channels in Convolutional Layer 4 of Different Stimuli Conditions")
    plt.xlabel("Channel Number")
    plt.ylabel("Central Unit Activity")
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 0, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 0, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 0, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 0, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 0, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 0, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 0, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 0, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 1, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 1, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 1, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 1, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 1, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 1, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 1, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 1, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 2, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 2, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 2, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 2, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 2, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 2, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_4.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, 2, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, 2, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 3')
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_4.tif')
    
    #Convolutional Layer 5:
    plt.figure()
    plt.title("Central Unit Activity of all Channels in Convolutional Layer 5 of Different Stimuli Conditions")
    plt.xlabel("Channel Number")
    plt.ylabel("Central Unit Activity")
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 0, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 0, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 0, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 0, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 0, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 0, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 0, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 0, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 0, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 1, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 1, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 1, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 1, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 1, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 1, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 1, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 1, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 1, Ori = 3')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 2, 0], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 2, 0], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 0')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 2, 1], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 2, 1], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 1')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 2, 2], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 2, 2], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 2')
    plt.errorbar(range(1, central_unit_activity_all_channels_analysis_layer_5.shape[1] + 1), np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, 2, 3], axis = 0), np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, 2, 3], axis = 0), color = np.random.rand(3), label = 'Condition: SF = 2, Ori = 3')
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(saving_folder_2 + '/central_unit_activity_all_channels_analysis_layer_5.tif')

    group_counter = group_counter + 1
    
    for i in range(0, len(SF_tuning)):
        all_groups_central_unit_activity_mean_channels_SF_analysis_layer_1[group_counter, i] = np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, i, :])
        all_groups_central_unit_activity_mean_channels_SF_analysis_layer_2[group_counter, i] = np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, i, :])
        all_groups_central_unit_activity_mean_channels_SF_analysis_layer_3[group_counter, i] = np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, i, :])
        all_groups_central_unit_activity_mean_channels_SF_analysis_layer_4[group_counter, i] = np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, i, :])
        all_groups_central_unit_activity_mean_channels_SF_analysis_layer_5[group_counter, i] = np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, i, :])
        
        all_groups_central_unit_activity_std_channels_SF_analysis_layer_1[group_counter, i] = np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, i, :])
        all_groups_central_unit_activity_std_channels_SF_analysis_layer_2[group_counter, i] = np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, i, :])
        all_groups_central_unit_activity_std_channels_SF_analysis_layer_3[group_counter, i] = np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, i, :])
        all_groups_central_unit_activity_std_channels_SF_analysis_layer_4[group_counter, i] = np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, i, :])
        all_groups_central_unit_activity_std_channels_SF_analysis_layer_5[group_counter, i] = np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, i, :])
        
        all_groups_central_unit_activity_SF_ttest_layer_1[group_counter, i, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, i, :], axis = 1), axis = 1)
        all_groups_central_unit_activity_SF_ttest_layer_2[group_counter, i, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, i, :], axis = 1), axis = 1)
        all_groups_central_unit_activity_SF_ttest_layer_3[group_counter, i, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, i, :], axis = 1), axis = 1)
        all_groups_central_unit_activity_SF_ttest_layer_4[group_counter, i, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, i, :], axis = 1), axis = 1)
        all_groups_central_unit_activity_SF_ttest_layer_5[group_counter, i, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, i, :], axis = 1), axis = 1)
        
    for j in range(0, len(Ori_tuning)):
        all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_1[group_counter, j] = np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, :, j])
        all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_2[group_counter, j] = np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, :, j])
        all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_3[group_counter, j] = np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, :, j])
        all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_4[group_counter, j] = np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, :, j])
        all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_5[group_counter, j] = np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, :, j])
        
        all_groups_central_unit_activity_std_channels_Ori_analysis_layer_1[group_counter, j] = np.std(central_unit_activity_all_channels_analysis_layer_1[:, :, :, j])
        all_groups_central_unit_activity_std_channels_Ori_analysis_layer_2[group_counter, j] = np.std(central_unit_activity_all_channels_analysis_layer_2[:, :, :, j])
        all_groups_central_unit_activity_std_channels_Ori_analysis_layer_3[group_counter, j] = np.std(central_unit_activity_all_channels_analysis_layer_3[:, :, :, j])
        all_groups_central_unit_activity_std_channels_Ori_analysis_layer_4[group_counter, j] = np.std(central_unit_activity_all_channels_analysis_layer_4[:, :, :, j])
        all_groups_central_unit_activity_std_channels_Ori_analysis_layer_5[group_counter, j] = np.std(central_unit_activity_all_channels_analysis_layer_5[:, :, :, j])
        
        all_groups_central_unit_activity_Ori_ttest_layer_1[group_counter, j, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_1[:, :, :, j], axis = 1), axis = 1)
        all_groups_central_unit_activity_Ori_ttest_layer_2[group_counter, j, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_2[:, :, :, j], axis = 1), axis = 1)
        all_groups_central_unit_activity_Ori_ttest_layer_3[group_counter, j, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_3[:, :, :, j], axis = 1), axis = 1)
        all_groups_central_unit_activity_Ori_ttest_layer_4[group_counter, j, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_4[:, :, :, j], axis = 1), axis = 1)
        all_groups_central_unit_activity_Ori_ttest_layer_5[group_counter, j, :] = np.mean(np.mean(central_unit_activity_all_channels_analysis_layer_5[:, :, :, j], axis = 1), axis = 1)

### Spatial Frequency

#Convolutional Layer 1:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 1 of Different Spatial Frequencies")
plt.xlabel("Spatial Frequency")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_1[0, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_1[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_1[1, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_1[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_1[2, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_1[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_1[3, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_1[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(SF_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_SF_analysis_layer_1.tif')

#Convolutional Layer 2:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 2 of Different Spatial Frequencies")
plt.xlabel("Spatial Frequency")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_2[0, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_2[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_2[1, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_2[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_2[2, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_2[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_2[3, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_2[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(SF_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_SF_analysis_layer_2.tif')

#Convolutional Layer 3:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 3 of Different Spatial Frequencies")
plt.xlabel("Spatial Frequency")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_3[0, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_3[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_3[1, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_3[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_3[2, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_3[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_3[3, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_3[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(SF_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_SF_analysis_layer_3.tif')

#Convolutional Layer 4:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 4 of Different Spatial Frequencies")
plt.xlabel("Spatial Frequency")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_4[0, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_4[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_4[1, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_4[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_4[2, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_4[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_4[3, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_4[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(SF_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_SF_analysis_layer_4.tif')

#Convolutional Layer 5:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 5 of Different Spatial Frequencies")
plt.xlabel("Spatial Frequency")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_5[0, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_5[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_5[1, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_5[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_5[2, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_5[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(SF_tuning) + 1), all_groups_central_unit_activity_mean_channels_SF_analysis_layer_5[3, :], all_groups_central_unit_activity_std_channels_SF_analysis_layer_5[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(SF_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_SF_analysis_layer_5.tif')

### Orientation

#Convolutional Layer 1:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 1 of Different Orientations")
plt.xlabel("Orientation")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_1[0, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_1[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_1[1, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_1[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_1[2, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_1[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_1[3, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_1[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(Ori_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_1.tif')

#Convolutional Layer 2:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 2 of Different Orientations")
plt.xlabel("Orientation")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_2[0, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_2[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_2[1, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_2[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_2[2, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_2[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_2[3, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_2[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(Ori_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_2.tif')

#Convolutional Layer 3:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 3 of Different Orientations")
plt.xlabel("Orientation")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_3[0, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_3[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_3[1, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_3[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_3[2, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_3[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_3[3, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_3[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(Ori_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_3.tif')

#Convolutional Layer 4:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 4 of Different Orientations")
plt.xlabel("Orientation")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_4[0, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_4[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_4[1, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_4[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_4[2, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_4[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_4[3, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_4[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(Ori_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_4.tif')

#Convolutional Layer 5:    
plt.figure()
plt.title("Average Central Unit Activity of Non-Zero Channels in Convolutional Layer 5 of Different Orientations")
plt.xlabel("Orientation")
plt.ylabel("Average Central Unit Activity")
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_5[0, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_5[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_5[1, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_5[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_5[2, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_5[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(1, len(Ori_tuning) + 1), all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_5[3, :], all_groups_central_unit_activity_std_channels_Ori_analysis_layer_5[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.xticks(np.arange(0, len(Ori_tuning) + 2, 1.0))
plt.show()
plt.savefig(saving_folder_1 + '/all_groups_central_unit_activity_mean_channels_Ori_analysis_layer_5.tif')

### Statistical Analysis of Spatial Frequency:

t_test_SF_12 = np.zeros((3, 2))
t_test_SF_34 = np.zeros((3, 2))

#Convolutional Layer 1:
[t_test_SF_12[0, 0], t_test_SF_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_1[0, 0, :], all_groups_central_unit_activity_SF_ttest_layer_1[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[1, 0], t_test_SF_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_1[0, 1, :], all_groups_central_unit_activity_SF_ttest_layer_1[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[2, 0], t_test_SF_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_1[0, 2, :], all_groups_central_unit_activity_SF_ttest_layer_1[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_SF_34[0, 0], t_test_SF_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_1[2, 0, :], all_groups_central_unit_activity_SF_ttest_layer_1[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[1, 0], t_test_SF_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_1[2, 1, :], all_groups_central_unit_activity_SF_ttest_layer_1[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[2, 0], t_test_SF_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_1[2, 2, :], all_groups_central_unit_activity_SF_ttest_layer_1[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 1: (SF) (group 1&2)')
print(t_test_SF_12[0, 0], ' , ', t_test_SF_12[0, 1])
print(t_test_SF_12[1, 0], ' , ', t_test_SF_12[1, 1])
print(t_test_SF_12[2, 0], ' , ', t_test_SF_12[2, 1])

print('Convolutional Layer 1: (SF) (group 3&4)')
print(t_test_SF_34[0, 0], ' , ', t_test_SF_34[0, 1])
print(t_test_SF_34[1, 0], ' , ', t_test_SF_34[1, 1])
print(t_test_SF_34[2, 0], ' , ', t_test_SF_34[2, 1])

#Convolutional Layer 2:
[t_test_SF_12[0, 0], t_test_SF_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_2[0, 0, :], all_groups_central_unit_activity_SF_ttest_layer_2[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[1, 0], t_test_SF_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_2[0, 1, :], all_groups_central_unit_activity_SF_ttest_layer_2[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[2, 0], t_test_SF_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_2[0, 2, :], all_groups_central_unit_activity_SF_ttest_layer_2[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_SF_34[0, 0], t_test_SF_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_2[2, 0, :], all_groups_central_unit_activity_SF_ttest_layer_2[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[1, 0], t_test_SF_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_2[2, 1, :], all_groups_central_unit_activity_SF_ttest_layer_2[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[2, 0], t_test_SF_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_2[2, 2, :], all_groups_central_unit_activity_SF_ttest_layer_2[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 2: (SF) (group 1&2)')
print(t_test_SF_12[0, 0], ' , ', t_test_SF_12[0, 1])
print(t_test_SF_12[1, 0], ' , ', t_test_SF_12[1, 1])
print(t_test_SF_12[2, 0], ' , ', t_test_SF_12[2, 1])

print('Convolutional Layer 2: (SF) (group 3&4)')
print(t_test_SF_34[0, 0], ' , ', t_test_SF_34[0, 1])
print(t_test_SF_34[1, 0], ' , ', t_test_SF_34[1, 1])
print(t_test_SF_34[2, 0], ' , ', t_test_SF_34[2, 1])

#Convolutional Layer 3:
[t_test_SF_12[0, 0], t_test_SF_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_3[0, 0, :], all_groups_central_unit_activity_SF_ttest_layer_3[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[1, 0], t_test_SF_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_3[0, 1, :], all_groups_central_unit_activity_SF_ttest_layer_3[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[2, 0], t_test_SF_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_3[0, 2, :], all_groups_central_unit_activity_SF_ttest_layer_3[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_SF_34[0, 0], t_test_SF_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_3[2, 0, :], all_groups_central_unit_activity_SF_ttest_layer_3[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[1, 0], t_test_SF_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_3[2, 1, :], all_groups_central_unit_activity_SF_ttest_layer_3[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[2, 0], t_test_SF_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_3[2, 2, :], all_groups_central_unit_activity_SF_ttest_layer_3[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 3: (SF) (group 1&2)')
print(t_test_SF_12[0, 0], ' , ', t_test_SF_12[0, 1])
print(t_test_SF_12[1, 0], ' , ', t_test_SF_12[1, 1])
print(t_test_SF_12[2, 0], ' , ', t_test_SF_12[2, 1])

print('Convolutional Layer 3: (SF) (group 3&4)')
print(t_test_SF_34[0, 0], ' , ', t_test_SF_34[0, 1])
print(t_test_SF_34[1, 0], ' , ', t_test_SF_34[1, 1])
print(t_test_SF_34[2, 0], ' , ', t_test_SF_34[2, 1])

#Convolutional Layer 4:
[t_test_SF_12[0, 0], t_test_SF_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_4[0, 0, :], all_groups_central_unit_activity_SF_ttest_layer_4[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[1, 0], t_test_SF_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_4[0, 1, :], all_groups_central_unit_activity_SF_ttest_layer_4[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[2, 0], t_test_SF_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_4[0, 2, :], all_groups_central_unit_activity_SF_ttest_layer_4[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_SF_34[0, 0], t_test_SF_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_4[2, 0, :], all_groups_central_unit_activity_SF_ttest_layer_4[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[1, 0], t_test_SF_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_4[2, 1, :], all_groups_central_unit_activity_SF_ttest_layer_4[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[2, 0], t_test_SF_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_4[2, 2, :], all_groups_central_unit_activity_SF_ttest_layer_4[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 4: (SF) (group 1&2)')
print(t_test_SF_12[0, 0], ' , ', t_test_SF_12[0, 1])
print(t_test_SF_12[1, 0], ' , ', t_test_SF_12[1, 1])
print(t_test_SF_12[2, 0], ' , ', t_test_SF_12[2, 1])

print('Convolutional Layer 4: (SF) (group 3&4)')
print(t_test_SF_34[0, 0], ' , ', t_test_SF_34[0, 1])
print(t_test_SF_34[1, 0], ' , ', t_test_SF_34[1, 1])
print(t_test_SF_34[2, 0], ' , ', t_test_SF_34[2, 1])

#Convolutional Layer 5:
[t_test_SF_12[0, 0], t_test_SF_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_5[0, 0, :], all_groups_central_unit_activity_SF_ttest_layer_5[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[1, 0], t_test_SF_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_5[0, 1, :], all_groups_central_unit_activity_SF_ttest_layer_5[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_12[2, 0], t_test_SF_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_5[0, 2, :], all_groups_central_unit_activity_SF_ttest_layer_5[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_SF_34[0, 0], t_test_SF_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_5[2, 0, :], all_groups_central_unit_activity_SF_ttest_layer_5[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[1, 0], t_test_SF_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_5[2, 1, :], all_groups_central_unit_activity_SF_ttest_layer_5[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_SF_34[2, 0], t_test_SF_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_SF_ttest_layer_5[2, 2, :], all_groups_central_unit_activity_SF_ttest_layer_5[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 5: (SF) (group 1&2)')
print(t_test_SF_12[0, 0], ' , ', t_test_SF_12[0, 1])
print(t_test_SF_12[1, 0], ' , ', t_test_SF_12[1, 1])
print(t_test_SF_12[2, 0], ' , ', t_test_SF_12[2, 1])

print('Convolutional Layer 5: (SF) (group 3&4)')
print(t_test_SF_34[0, 0], ' , ', t_test_SF_34[0, 1])
print(t_test_SF_34[1, 0], ' , ', t_test_SF_34[1, 1])
print(t_test_SF_34[2, 0], ' , ', t_test_SF_34[2, 1])

### Statistical Analysis of Orientation:

t_test_Ori_12 = np.zeros((4, 2))
t_test_Ori_34 = np.zeros((4, 2))

#Convolutional Layer 1:
[t_test_Ori_12[0, 0], t_test_Ori_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_1[0, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_1[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[1, 0], t_test_Ori_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_1[0, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_1[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[2, 0], t_test_Ori_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_1[0, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_1[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[3, 0], t_test_Ori_12[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_1[0, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_1[1, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_Ori_34[0, 0], t_test_Ori_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_1[2, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_1[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[1, 0], t_test_Ori_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_1[2, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_1[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[2, 0], t_test_Ori_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_1[2, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_1[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[3, 0], t_test_Ori_34[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_1[2, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_1[3, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 1: (Ori) (group 1&2)')
print(t_test_Ori_12[0, 0], ' , ', t_test_Ori_12[0, 1])
print(t_test_Ori_12[1, 0], ' , ', t_test_Ori_12[1, 1])
print(t_test_Ori_12[2, 0], ' , ', t_test_Ori_12[2, 1])
print(t_test_Ori_12[3, 0], ' , ', t_test_Ori_12[3, 1])

print('Convolutional Layer 1: (Ori) (group 3&4)')
print(t_test_Ori_34[0, 0], ' , ', t_test_Ori_34[0, 1])
print(t_test_Ori_34[1, 0], ' , ', t_test_Ori_34[1, 1])
print(t_test_Ori_34[2, 0], ' , ', t_test_Ori_34[2, 1])
print(t_test_Ori_34[3, 0], ' , ', t_test_Ori_34[3, 1])

#Convolutional Layer 2:
[t_test_Ori_12[0, 0], t_test_Ori_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_2[0, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_2[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[1, 0], t_test_Ori_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_2[0, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_2[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[2, 0], t_test_Ori_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_2[0, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_2[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[3, 0], t_test_Ori_12[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_2[0, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_2[1, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_Ori_34[0, 0], t_test_Ori_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_2[2, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_2[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[1, 0], t_test_Ori_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_2[2, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_2[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[2, 0], t_test_Ori_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_2[2, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_2[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[3, 0], t_test_Ori_34[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_2[2, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_2[3, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 2: (Ori) (group 1&2)')
print(t_test_Ori_12[0, 0], ' , ', t_test_Ori_12[0, 1])
print(t_test_Ori_12[1, 0], ' , ', t_test_Ori_12[1, 1])
print(t_test_Ori_12[2, 0], ' , ', t_test_Ori_12[2, 1])
print(t_test_Ori_12[3, 0], ' , ', t_test_Ori_12[3, 1])

print('Convolutional Layer 2: (Ori) (group 3&4)')
print(t_test_Ori_34[0, 0], ' , ', t_test_Ori_34[0, 1])
print(t_test_Ori_34[1, 0], ' , ', t_test_Ori_34[1, 1])
print(t_test_Ori_34[2, 0], ' , ', t_test_Ori_34[2, 1])
print(t_test_Ori_34[3, 0], ' , ', t_test_Ori_34[3, 1])

#Convolutional Layer 3:
[t_test_Ori_12[0, 0], t_test_Ori_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_3[0, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_3[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[1, 0], t_test_Ori_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_3[0, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_3[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[2, 0], t_test_Ori_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_3[0, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_3[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[3, 0], t_test_Ori_12[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_3[0, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_3[1, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_Ori_34[0, 0], t_test_Ori_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_3[2, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_3[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[1, 0], t_test_Ori_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_3[2, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_3[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[2, 0], t_test_Ori_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_3[2, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_3[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[3, 0], t_test_Ori_34[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_3[2, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_3[3, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 3: (Ori) (group 1&2)')
print(t_test_Ori_12[0, 0], ' , ', t_test_Ori_12[0, 1])
print(t_test_Ori_12[1, 0], ' , ', t_test_Ori_12[1, 1])
print(t_test_Ori_12[2, 0], ' , ', t_test_Ori_12[2, 1])
print(t_test_Ori_12[3, 0], ' , ', t_test_Ori_12[3, 1])

print('Convolutional Layer 3: (Ori) (group 3&4)')
print(t_test_Ori_34[0, 0], ' , ', t_test_Ori_34[0, 1])
print(t_test_Ori_34[1, 0], ' , ', t_test_Ori_34[1, 1])
print(t_test_Ori_34[2, 0], ' , ', t_test_Ori_34[2, 1])
print(t_test_Ori_34[3, 0], ' , ', t_test_Ori_34[3, 1])

#Convolutional Layer 4:
[t_test_Ori_12[0, 0], t_test_Ori_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_4[0, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_4[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[1, 0], t_test_Ori_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_4[0, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_4[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[2, 0], t_test_Ori_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_4[0, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_4[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[3, 0], t_test_Ori_12[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_4[0, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_4[1, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_Ori_34[0, 0], t_test_Ori_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_4[2, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_4[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[1, 0], t_test_Ori_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_4[2, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_4[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[2, 0], t_test_Ori_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_4[2, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_4[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[3, 0], t_test_Ori_34[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_4[2, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_4[3, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 4: (Ori) (group 1&2)')
print(t_test_Ori_12[0, 0], ' , ', t_test_Ori_12[0, 1])
print(t_test_Ori_12[1, 0], ' , ', t_test_Ori_12[1, 1])
print(t_test_Ori_12[2, 0], ' , ', t_test_Ori_12[2, 1])
print(t_test_Ori_12[3, 0], ' , ', t_test_Ori_12[3, 1])

print('Convolutional Layer 4: (Ori) (group 3&4)')
print(t_test_Ori_34[0, 0], ' , ', t_test_Ori_34[0, 1])
print(t_test_Ori_34[1, 0], ' , ', t_test_Ori_34[1, 1])
print(t_test_Ori_34[2, 0], ' , ', t_test_Ori_34[2, 1])
print(t_test_Ori_34[3, 0], ' , ', t_test_Ori_34[3, 1])

#Convolutional Layer 5:
[t_test_Ori_12[0, 0], t_test_Ori_12[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_5[0, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_5[1, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[1, 0], t_test_Ori_12[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_5[0, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_5[1, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[2, 0], t_test_Ori_12[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_5[0, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_5[1, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_12[3, 0], t_test_Ori_12[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_5[0, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_5[1, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

[t_test_Ori_34[0, 0], t_test_Ori_34[0, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_5[2, 0, :], all_groups_central_unit_activity_Ori_ttest_layer_5[3, 0, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[1, 0], t_test_Ori_34[1, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_5[2, 1, :], all_groups_central_unit_activity_Ori_ttest_layer_5[3, 1, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[2, 0], t_test_Ori_34[2, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_5[2, 2, :], all_groups_central_unit_activity_Ori_ttest_layer_5[3, 2, :], axis = None, equal_var = False, nan_policy = 'propagate')
[t_test_Ori_34[3, 0], t_test_Ori_34[3, 1]] = stats.ttest_ind(all_groups_central_unit_activity_Ori_ttest_layer_5[2, 3, :], all_groups_central_unit_activity_Ori_ttest_layer_5[3, 3, :], axis = None, equal_var = False, nan_policy = 'propagate')

print('Convolutional Layer 5: (Ori) (group 1&2)')
print(t_test_Ori_12[0, 0], ' , ', t_test_Ori_12[0, 1])
print(t_test_Ori_12[1, 0], ' , ', t_test_Ori_12[1, 1])
print(t_test_Ori_12[2, 0], ' , ', t_test_Ori_12[2, 1])
print(t_test_Ori_12[3, 0], ' , ', t_test_Ori_12[3, 1])

print('Convolutional Layer 5: (Ori) (group 3&4)')
print(t_test_Ori_34[0, 0], ' , ', t_test_Ori_34[0, 1])
print(t_test_Ori_34[1, 0], ' , ', t_test_Ori_34[1, 1])
print(t_test_Ori_34[2, 0], ' , ', t_test_Ori_34[2, 1])
print(t_test_Ori_34[3, 0], ' , ', t_test_Ori_34[3, 1])