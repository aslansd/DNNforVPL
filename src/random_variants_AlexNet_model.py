"""
Created by Aslan Satary Dizaji (a.satarydizaji@eni-g.de)
"""

import torch
import torch.nn as nn

# The smaller variants of AlexNet are built from the original variant of AlexNet

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