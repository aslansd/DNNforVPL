"""
Created by Aslan Satary Dizaji (a.satarydizaji@eni-g.de)
"""

import torch
import torch.nn as nn

### A class for measuring the cosine distance between two tensors with equal sizes 

def layer_rotation(weight_t, weight_0):
    # Cosine distance definition
    CosSim = nn.CosineSimilarity(dim = 0, eps = 1e-10)
    
    return (1 - CosSim(torch.flatten(weight_t), torch.flatten(weight_0)).item())