"""
Created on Mon Sep  7 17:13:34 2020

@author: satarydizaji
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from pwcca import compute_pwcca
# from cka import feature_space_linear_cka
    
def main():
    
    ### Initializing the main variables
    
    num_sample_artiphysiology = 1000    
    number_simulation = 3
    number_group = 4
    number_layer = 5
    number_layer_freeze = 6
    
    parent_folder = '3 Simulations_AlexNet_2Str_ArtPhy_FL_DR_LR_PWCCA_CKA'
    
    all_simulation_all_PWCCA = np.zeros((int(number_simulation * (number_simulation - 1) / 2), number_group, number_layer, number_layer_freeze), dtype = np.float32)
    # all_simulation_all_CKA = np.zeros((int(number_simulation * (number_simulation - 1) / 2), number_group, number_layer, number_layer_freeze), dtype = np.float32)
        
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
                    
                    # all_simulation_all_CKA[counter, group_counter, 0, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_1, all_unit_activity_layer_j_1, debiased = True)
                    # all_simulation_all_CKA[counter, group_counter, 1, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_2, all_unit_activity_layer_j_2, debiased = True)
                    # all_simulation_all_CKA[counter, group_counter, 2, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_3, all_unit_activity_layer_j_3, debiased = True)
                    # all_simulation_all_CKA[counter, group_counter, 3, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_4, all_unit_activity_layer_j_4, debiased = True)
                    # all_simulation_all_CKA[counter, group_counter, 4, layer_freeze_counter] = 1 - feature_space_linear_cka(all_unit_activity_layer_i_5, all_unit_activity_layer_j_5, debiased = True)
                    
    ### Saving the main variables
   
    scipy.io.savemat(parent_folder + '/all_simulation_all_PWCCA.mat', mdict = {'all_simulation_all_PWCCA': all_simulation_all_PWCCA})
    # scipy.io.savemat(parent_folder + '/all_simulation_all_CKA.mat', mdict = {'all_simulation_all_CKA': all_simulation_all_CKA})
    
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
        
        ax.plot(range(0, 5), all_simulation_all_PWCCA.mean(0)[0, :, i], "-b", label = "Group 1")
        ax.fill_between(range(0, 5), all_simulation_all_PWCCA.mean(0)[0, :, i] - all_simulation_all_PWCCA.std(0)[0, :, i] / number_simulation ** 0.5, all_simulation_all_PWCCA.mean(0)[0, :, i] + all_simulation_all_PWCCA.std(0)[0, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot(range(0, 5), all_simulation_all_PWCCA.mean(0)[1, :, i], "-g", label = "Group 2")
        ax.fill_between(range(0, 5), all_simulation_all_PWCCA.mean(0)[1, :, i] - all_simulation_all_PWCCA.std(0)[1, :, i] / number_simulation ** 0.5, all_simulation_all_PWCCA.mean(0)[1, :, i] + all_simulation_all_PWCCA.std(0)[1, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
        ax.plot(range(0, 5), all_simulation_all_PWCCA.mean(0)[2, :, i], "-r", label = "Group 3")
        ax.fill_between(range(0, 5), all_simulation_all_PWCCA.mean(0)[2, :, i] - all_simulation_all_PWCCA.std(0)[2, :, i] / number_simulation ** 0.5, all_simulation_all_PWCCA.mean(0)[2, :, i] + all_simulation_all_PWCCA.std(0)[2, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
        ax.plot(range(0, 5), all_simulation_all_PWCCA.mean(0)[3, :, i], "-c", label = "Group 4")
        ax.fill_between(range(0, 5), all_simulation_all_PWCCA.mean(0)[3, :, i] - all_simulation_all_PWCCA.std(0)[3, :, i] / number_simulation ** 0.5, all_simulation_all_PWCCA.mean(0)[3, :, i] + all_simulation_all_PWCCA.std(0)[3, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
        ax.legend(loc = 'upper left', fontsize = 'medium')
        ax.set_ylim((0, 0.005))
        ax.set_xticks(range(0, 5))
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
                
    fig.savefig(parent_folder + '/PWCCA Distance.tif')
       
    # ### CKA: Similarity of Neural Network Representations Revisited
    
    # fig, axs = plt.subplots(2, 3, figsize = (2 * 8, 3 * 6))
    # fig.suptitle('Centered Kernel Alignment', fontsize = 20)
    
    # for i in range(number_layer_freeze):
    #     if i <= 2:
    #         ax = axs[0, i]
    #     elif i > 2:
    #         ax = axs[1, i - 3]
        
    #     ax.set_title('Freezed Layer = ' + str(i), fontsize = 12)
    #     ax.set_ylabel('CKA distance')
        
    #     ax.plot(range(0, 5), all_simulation_all_CKA.mean(0)[0, :, i], "-b", label = "Group 1")
    #     ax.fill_between(range(0, 5), all_simulation_all_CKA.mean(0)[0, :, i] - all_simulation_all_CKA.std(0)[0, :, i] / number_simulation ** 0.5, all_simulation_all_CKA.mean(0)[0, :, i] + all_simulation_all_CKA.std(0)[0, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
    #     ax.plot(range(0, 5), all_simulation_all_CKA.mean(0)[1, :, i], "-g", label = "Group 2")
    #     ax.fill_between(range(0, 5), all_simulation_all_CKA.mean(0)[1, :, i] - all_simulation_all_CKA.std(0)[1, :, i] / number_simulation ** 0.5, all_simulation_all_CKA.mean(0)[1, :, i] + all_simulation_all_CKA.std(0)[1, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'g', facecolor = 'g')
        
    #     ax.plot(range(0, 5), all_simulation_all_CKA.mean(0)[2, :, i], "-r", label = "Group 3")
    #     ax.fill_between(range(0, 5), all_simulation_all_CKA.mean(0)[2, :, i] - all_simulation_all_CKA.std(0)[2, :, i] / number_simulation ** 0.5, all_simulation_all_CKA.mean(0)[2, :, i] + all_simulation_all_CKA.std(0)[2, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'r', facecolor = 'r')
        
    #     ax.plot(range(0, 5), all_simulation_all_CKA.mean(0)[3, :, i], "-c", label = "Group 4")
    #     ax.fill_between(range(0, 5), all_simulation_all_CKA.mean(0)[3, :, i] - all_simulation_all_CKA.std(0)[3, :, i] / number_simulation ** 0.5, all_simulation_all_CKA.mean(0)[3, :, i] + all_simulation_all_CKA.std(0)[3, :, i] / number_simulation ** 0.5, alpha = 0.5, edgecolor = 'c', facecolor = 'c')
                
    #     ax.legend(loc = 'upper left', fontsize = 'medium')
    #     ax.set_ylim((0, 0.005))
    #     ax.set_xticks(range(0, 5))
    #     ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
                
    # fig.savefig(parent_folder + '/CKA Distance.tif')
   
if __name__ == '__main__':
    main()