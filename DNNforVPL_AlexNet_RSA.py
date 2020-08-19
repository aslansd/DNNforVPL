"""
Created on Mon Aug 17 09:44:41 2020

@author: satarydizaji
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

parent_folder = 'W:/satarydizaji/ENI Projects/Caspar/Python Programs/PyTorch/46 Simulations_SF Transfer_Captum_AlexNet'

num_simulations = 46
num_groups = 4
num_layers = 5
num_stimuli = 36

def main():

    RSA_results = np.zeros((num_simulations, num_groups, num_layers, num_stimuli, num_stimuli))
    
    all_simulation_unit_activity_all_channels_layer_1 = np.zeros((num_simulations, num_groups, num_stimuli, 64 * 55 * 55), dtype = np.float32)
    all_simulation_unit_activity_all_channels_layer_2 = np.zeros((num_simulations, num_groups, num_stimuli, 192 * 27 * 27), dtype = np.float32)
    all_simulation_unit_activity_all_channels_layer_3 = np.zeros((num_simulations, num_groups, num_stimuli, 384 * 13 * 13), dtype = np.float32)
    all_simulation_unit_activity_all_channels_layer_4 = np.zeros((num_simulations, num_groups, num_stimuli, 256 * 13 * 13), dtype = np.float32)
    all_simulation_unit_activity_all_channels_layer_5 = np.zeros((num_simulations, num_groups, num_stimuli, 256 * 13 * 13), dtype = np.float32)
    
    for i in range(num_simulations):
        print(i)
        
        group_counter = -1
        
        for group_training in ['group1', 'group2', 'group3', 'group4']:
            print(group_training)
            
            group_counter = group_counter + 1
        
            if group_training == 'group1' or group_training == 'group2':
                SF_tuning = [33, 53, 140, 170, 340, 480]
                Ori_tuning = [23350, 23450, 23550,
                              23650, 23750, 23850]
                
            elif group_training == 'group3' or group_training == 'group4':
                SF_tuning = [33, 53, 140, 170, 340, 480]
                Ori_tuning = [23100, 23200, 23300,
                              23900, 24000, 24100]
            
            all_unit_activity_all_channels_analysis_layer_1 = np.zeros((num_stimuli, 64, 55, 55), dtype = np.float32)
            all_unit_activity_all_channels_analysis_layer_2 = np.zeros((num_stimuli, 192, 27, 27), dtype = np.float32)
            all_unit_activity_all_channels_analysis_layer_3 = np.zeros((num_stimuli, 384, 13, 13), dtype = np.float32)
            all_unit_activity_all_channels_analysis_layer_4 = np.zeros((num_stimuli, 256, 13, 13), dtype = np.float32)
            all_unit_activity_all_channels_analysis_layer_5 = np.zeros((num_stimuli, 256, 13, 13), dtype = np.float32)
              
            data_folder_after = parent_folder + '/Simulation_' + str(i + 1) + '/' + group_training + '/after_training_None'
            
            data_feature = loadmat(data_folder_after + '/feature_sample_artiphysiology.mat')
            data_feature_matrix = data_feature['feature_sample_artiphysiology']
            
            data_layer_after = loadmat(data_folder_after + '/all_unit_activity_Conv2d_1.mat')
            all_unit_activity_all_channels_layer_1 = data_layer_after['all_unit_activity_Conv2d_1']
                        
            data_layer_after = loadmat(data_folder_after + '/all_unit_activity_Conv2d_2.mat')
            all_unit_activity_all_channels_layer_2 = data_layer_after['all_unit_activity_Conv2d_2']
           
            data_layer_after = loadmat(data_folder_after + '/all_unit_activity_Conv2d_3.mat')
            all_unit_activity_all_channels_layer_3 = data_layer_after['all_unit_activity_Conv2d_3']
            
            data_layer_after = loadmat(data_folder_after + '/all_unit_activity_Conv2d_4.mat')
            all_unit_activity_all_channels_layer_4 = data_layer_after['all_unit_activity_Conv2d_4']
             
            data_layer_after = loadmat(data_folder_after + '/all_unit_activity_Conv2d_5.mat')
            all_unit_activity_all_channels_layer_5 = data_layer_after['all_unit_activity_Conv2d_5']
                   
            for j in range(0, len(SF_tuning)):
                for k in range(0, len(Ori_tuning)):
                    indices = np.intersect1d(np.where(data_feature_matrix[:, 0] == SF_tuning[j]), np.where(data_feature_matrix[:, 1] == Ori_tuning[k]))
                      
                    all_unit_activity_all_channels_analysis_layer_1[j * len(SF_tuning) + k, :] = np.mean(all_unit_activity_all_channels_layer_1[indices, :], axis = 0)
                    all_unit_activity_all_channels_analysis_layer_2[j * len(SF_tuning) + k, :] = np.mean(all_unit_activity_all_channels_layer_2[indices, :], axis = 0)
                    all_unit_activity_all_channels_analysis_layer_3[j * len(SF_tuning) + k, :] = np.mean(all_unit_activity_all_channels_layer_3[indices, :], axis = 0)
                    all_unit_activity_all_channels_analysis_layer_4[j * len(SF_tuning) + k, :] = np.mean(all_unit_activity_all_channels_layer_4[indices, :], axis = 0)
                    all_unit_activity_all_channels_analysis_layer_5[j * len(SF_tuning) + k, :] = np.mean(all_unit_activity_all_channels_layer_5[indices, :], axis = 0)
                
            RSA_results[i, group_counter, 0, :, :] = RDM(all_unit_activity_all_channels_analysis_layer_1.reshape(num_stimuli, -1))
            RSA_results[i, group_counter, 1, :, :] = RDM(all_unit_activity_all_channels_analysis_layer_2.reshape(num_stimuli, -1))
            RSA_results[i, group_counter, 2, :, :] = RDM(all_unit_activity_all_channels_analysis_layer_3.reshape(num_stimuli, -1))
            RSA_results[i, group_counter, 3, :, :] = RDM(all_unit_activity_all_channels_analysis_layer_4.reshape(num_stimuli, -1))
            RSA_results[i, group_counter, 4, :, :] = RDM(all_unit_activity_all_channels_analysis_layer_5.reshape(num_stimuli, -1))
            
            all_simulation_unit_activity_all_channels_layer_1[i, group_counter, :, :] = all_unit_activity_all_channels_analysis_layer_1.reshape(num_stimuli, -1)
            all_simulation_unit_activity_all_channels_layer_2[i, group_counter, :, :] = all_unit_activity_all_channels_analysis_layer_2.reshape(num_stimuli, -1)
            all_simulation_unit_activity_all_channels_layer_3[i, group_counter, :, :] = all_unit_activity_all_channels_analysis_layer_3.reshape(num_stimuli, -1)
            all_simulation_unit_activity_all_channels_layer_4[i, group_counter, :, :] = all_unit_activity_all_channels_analysis_layer_4.reshape(num_stimuli, -1)
            all_simulation_unit_activity_all_channels_layer_5[i, group_counter, :, :] = all_unit_activity_all_channels_analysis_layer_5.reshape(num_stimuli, -1)
            
    RSA_results_average = np.mean(RSA_results, axis = 0)
    
    # Aggregate all responses into one dict
    rdm_dict = {}
    for i in range(num_groups):
        for j in range(num_layers):
            label = 'group' + str(i + 1) + ' & ' + 'layer' + str(j + 1)
            rdm_dict[label] = RSA_results_average[i, j, :, :]
      
    # with plt.xkcd():
    plot_multiple_rdm(rdm_dict, num_groups, num_layers)
    
    # Correlate off-diagonal elements of representational dissimilarity matrices    
    # with plt.xkcd():
    plot_rdm_rdm_correlations(rdm_dict)    
    
    # Dimensionality reduction   
    resp_dict = {}
    for i in range(num_groups):
        label = 'group' + str(i + 1) + ' & ' + 'layer1'
        resp_dict[label] = all_simulation_unit_activity_all_channels_layer_1.mean(0)[i, :, :]
        
        label = 'group' + str(i + 1) + ' & ' + 'layer2'
        resp_dict[label] = all_simulation_unit_activity_all_channels_layer_2.mean(0)[i, :, :]
        
        label = 'group' + str(i + 1) + ' & ' + 'layer3'
        resp_dict[label] = all_simulation_unit_activity_all_channels_layer_3.mean(0)[i, :, :]
        
        label = 'group' + str(i + 1) + ' & ' + 'layer4'
        resp_dict[label] = all_simulation_unit_activity_all_channels_layer_4.mean(0)[i, :, :]
        
        label = 'group' + str(i + 1) + ' & ' + 'layer5'
        resp_dict[label] = all_simulation_unit_activity_all_channels_layer_5.mean(0)[i, :, :]
    
    # with plt.xkcd():
    plot_resp_lowd(resp_dict, num_groups, num_layers)
    
def RDM(resp):
    """Compute the representational dissimilarity matrix (RDM)
    Args:
      resp (ndarray): S x N matrix with population responses to each stimulus in each row
    Returns:
      ndarray: S x S representational dissimilarity matrix
    """
    
    # z-score responses to each stimulus
    zresp = zscore(resp, axis = 1)
    
    # Compute RDM
    RDM = 1 - (zresp @ zresp.T) / zresp.shape[1]
    
    return RDM

def plot_corr_matrix(rdm, ax, column):
    """Plot dissimilarity matrix
      
    Args:
      rdm (numpy array): n_stimuli x n_stimuli representational dissimilarity matrix
      ax (matplotlib axes): axes onto which to plot    
    Returns:
      nothing
    """
    
    if column == 0:
        vmin = 0
        vmax = 0.0008
    elif column == 1:
        vmin = 0
        vmax = 0.012
    elif column == 2:
        vmin = 0
        vmax = 0.015
    elif column == 3:
        vmin = 0
        vmax = 0.018
    elif column == 4:
        vmin = 0
        vmax = 0.015
                
    image = ax.imshow(rdm, vmin = vmin, vmax = vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax = ax, label = 'dissimilarity')
    
def plot_multiple_rdm(rdm_dict, num_groups, num_layers):
    """Draw multiple subplots for each RDM in rdm_dict."""
    
    fig, axs = plt.subplots(num_groups, num_layers, figsize = (num_groups * 4, num_layers * 3.5))
    fig.suptitle('Representational Dissimilarity Matrices')
      
    # Compute RDM's for each set of responses and plot
    for i, (label, rdm) in enumerate(rdm_dict.items()):
        row, column = np.unravel_index(i, (num_groups, num_layers), order = 'C')
        axs[row, column].set_title(label)
        plot_corr_matrix(rdm, axs[row, column], column)
      
def correlate_rdms(rdm1, rdm2):
    """Correlate off-diagonal elements of two RDM's
    Args:
      rdm1 (np.ndarray): S x S representational dissimilarity matrix
      rdm2 (np.ndarray): S x S representational dissimilarity matrix to correlate with rdm1
    Returns:
      float: correlation coefficient between the off-diagonal elements of rdm1 and rdm2
    """
    
    # Extract off-diagonal elements of each RDM
    ioffdiag = np.triu_indices(rdm1.shape[0], k = 1)  # indices of off-diagonal elements
    
    rdm1_offdiag = rdm1[ioffdiag]
    rdm2_offdiag = rdm2[ioffdiag]
    
    corr_coef = np.corrcoef(rdm1_offdiag, rdm2_offdiag)[0, 1]
    
    return corr_coef

def plot_rdm_rdm_correlations(rdm_dict):
    """Draw a bar plot showing between-RDM correlations."""
    
    corr_matrix_rdm_rdm = np.zeros((len(rdm_dict), len(rdm_dict)))
    label = []
    
    # Compute RDM's for each set of responses and plot
    for i, (label1, rdm1) in enumerate(rdm_dict.items()):
        label.append(label1)
        
        for j, (label2, rdm2) in enumerate(rdm_dict.items()):
            corr_matrix_rdm_rdm[i, j] = correlate_rdms(rdm1, rdm2)
          
    f, ax = plt.subplots()
    image = ax.imshow(corr_matrix_rdm_rdm)
    plt.colorbar(image, ax = ax, label = 'correlation')
    
    ax.set_xticks(np.arange(0, 20))
    ax.set_xticklabels(label, rotation = 'vertical')
    ax.set_yticks(np.arange(0, 20))
    ax.set_yticklabels(label, rotation = 'horizontal')
    
    ax.set_title('Correlation of off-diagonal elements of representational dissimilarity matrices')
  
def plot_resp_lowd(resp_dict, num_groups, num_layers):
    """Plot a low-dimensional representation of each dataset in resp_dict."""
    
    fig, axs = plt.subplots(num_groups, num_layers, figsize = (num_groups * 4, num_layers * 3.5))
    fig.suptitle('Dimensionality Reduction with PCA')
    
    for i, (label, resp) in enumerate(resp_dict.items()): 
        row, column = np.unravel_index(i, (num_groups, num_layers), order = 'C')
        ax = axs[row, column]
        ax.set_title('%s responses' % label)
    
        # Do PCA to reduce dimensionality to 2 dimensions
        resp_lowd = PCA(n_components = 2).fit_transform(resp)
    
        # Plot dimensionality-reduced population responses on 2D axes, with each point colored by stimulus orientation
        x, y = resp_lowd[:, 0], resp_lowd[:, 1]
        pts = ax.scatter(x, y, c = np.arange(0, num_stimuli), cmap = 'twilight', vmin = 0, vmax = num_stimuli)
        fig.colorbar(pts, ax = ax, ticks = np.linspace(0, num_stimuli, 5), label = 'Stimulus')
    
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig, axs = plt.subplots(num_groups, num_layers, figsize = (num_groups * 4, num_layers * 3.5))
    fig.suptitle('Dimensionality Reduction with tSNE')
    
    for i, (label, resp) in enumerate(resp_dict.items()): 
        row, column = np.unravel_index(i, (num_groups, num_layers), order = 'C')
        ax = axs[row, column]
        ax.set_title('%s responses' % label)
    
        # Do tSNE to reduce dimensionality to 2 dimensions
        resp_lowd = TSNE(n_components = 2).fit_transform(resp)
    
        # Plot dimensionality-reduced population responses on 2D axes, with each point colored by stimulus orientation
        x, y = resp_lowd[:, 0], resp_lowd[:, 1]
        pts = ax.scatter(x, y, c = np.arange(0, num_stimuli), cmap = 'twilight', vmin = 0, vmax = num_stimuli)
        fig.colorbar(pts, ax = ax, ticks = np.linspace(0, num_stimuli, 5), label = 'Stimulus')
    
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_xticks([])
        ax.set_yticks([])
      
if __name__ == '__main__':
    main()