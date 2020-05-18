# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:43:43 2020

@author: satarydizaji
"""

import numpy as np
import matplotlib.pyplot as plt

parent_folder = 'P:/ENI Projects/Caspar/Python Programs/PyTorch/New_Results/Simulation_1/'

all_transfer_results_mean = np.zeros((4, 6))
all_transfer_results_std = np.zeros((4, 6))

group_counter = -1

for group in ['group1', 'group2', 'group3', 'group4']:
    group_counter = group_counter + 1
    layer_counter = -1
    
    for layer in [None, 0, 3, 6, 8, 10]:
        layer_counter = layer_counter + 1
        
        file_name = parent_folder + group + '/after_training_' + str(layer) + '/Transfer_Accuracy.txt'
        transfer_results = np.loadtxt(file_name)

        all_transfer_results_mean[group_counter, layer_counter] = np.mean(transfer_results)
        all_transfer_results_std[group_counter, layer_counter] = np.std(transfer_results)

plt.figure()
plt.title("Average Transfer Results per Freezed Layer across Groups")
plt.xlabel("Freezed Layer")
plt.ylabel("Transfer Rate %")
plt.errorbar(range(0, 6), all_transfer_results_mean[0, :], all_transfer_results_std[0, :], color = 'b', label = 'Group 1')
plt.errorbar(range(0, 6), all_transfer_results_mean[1, :], all_transfer_results_std[1, :], color = 'g', label = 'Group 2')
plt.errorbar(range(0, 6), all_transfer_results_mean[2, :], all_transfer_results_std[2, :], color = 'r', label = 'Group 3')
plt.errorbar(range(0, 6), all_transfer_results_mean[3, :], all_transfer_results_std[3, :], color = 'c', label = 'Group 4')
plt.legend(loc="upper right")
plt.show()
plt.savefig(parent_folder + '/average_transfer_results_per_freezed_layer_across_groups.tif')