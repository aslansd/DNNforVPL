"""
Created by Aslan Satary Dizaji (a.satarydizaji@eni-g.de)
"""

import copy
import gc
import glob
import numpy as np
import os
import random
import scipy.io
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms

from PIL import Image
from scipy.spatial.distance import pdist, squareform
from torch.hub import load_state_dict_from_url

from intrinsic_dimension_2NN import estimate
from smaller_variants_AlexNet_model import DNNforVPL_1, DNNforVPL_2, DNNforVPL_3, DNNforVPL_4, DNNforVPL_5
from reading_stimuli import reading_stimuli

# The pretrained weights of AlexNet
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}
pretrained_dict = load_state_dict_from_url(model_urls['alexnet'])

### A class for formatting different metrics of accuracy during training and transfer

class AverageMeter(object):
    """Computes and stores the average and current values"""
    
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

### A class for showing a progress bar during training and transfer

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

### A function for computing accuracy during training and transfer

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
    
### A function for adjusting the learning rate during training

def adjust_learning_rate(optimizer, session, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 1 session"""
    
    lr = lr * (0.5 ** (session))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
### A function for saving the checkpoints during training

def save_checkpoint(state, is_best, group, filename):
    """ Saves the checkpoints during training """
    
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'DNNforVPL_best_' + group + '.pth.tar')
        
### A fucntion which performs different experiments with the smaller variants of AlexNet

def smaller_variants_alexnet(parent_folder = 'Smaller Variants of Alexnet_New Results', number_simulation = 5, num_sample_artiphysiology = 1000):
    
    ### Initializing the main variables
        
    x_sample_artiphysiology_index = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
    
    for i in range(0, num_sample_artiphysiology):
        x_sample_artiphysiology_index[i, 0] = random.randrange(1)
        x_sample_artiphysiology_index[i, 1] = random.randrange(20)
        x_sample_artiphysiology_index[i, 2] = random.randrange(180)
        
    number_model = 5
    number_group = 4
        
    all_simulation_training_accuracy = []
    all_simulation_transfer_accuracy = []
    all_simulation_all_ID = []
    all_x_sample_ID = []
    
    all_simulation_training_accuracy_permuted = []
    all_simulation_all_ID_permuted = []
                   
    for i in range(number_model):
        number_layer = number_model
            
        all_simulation_training_accuracy.append(np.zeros((number_simulation, number_group, 180), dtype = np.float32))
        all_simulation_transfer_accuracy.append(np.zeros((number_simulation, number_group, 10), dtype = np.float32))
        all_simulation_all_ID.append(np.zeros((number_simulation, number_group, number_layer, 19), dtype = np.float32))
        all_x_sample_ID.append(np.zeros((number_simulation, number_group), dtype = np.float32))
        
        all_simulation_training_accuracy_permuted.append(np.zeros((number_simulation, number_group, 180), dtype = np.float32))
        all_simulation_all_ID_permuted.append(np.zeros((number_simulation, number_group, number_layer, 19), dtype = np.float32))
        
    os.mkdir(parent_folder)
    
    for simulation_counter in range(number_simulation):
        print('Simulation:   ', simulation_counter + 1)
        
        os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1))
            
        group_counter = -1
        
        for group_training in ['group1', 'group2', 'group3', 'group4']:
            gc.collect()
            best_acc1 = 0
            group_counter = group_counter + 1
                     
            print('Group:   ', group_training)
            
            os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training)
            
            ### Training Stimuli
            
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
                # file_name_paths = glob.glob(os.path.dirname(os.path.abspath("./")) + '/data/stimuli/training_groups1&2/*.TIFF')
                file_name_paths = glob.glob('D:/ENI Projects/Aslan/Neuro-Inspired Vision/Githubs/DNNforVPL/src/data/stimuli/training_groups1&2/*.TIFF')
            elif group_training == 'group3' or group_training == 'group4':
                # file_name_paths = glob.glob(os.path.dirname(os.path.abspath("./")) + '/data/stimuli/training_groups3&4/*.TIFF')
                file_name_paths = glob.glob('D:/ENI Projects/Aslan/Neuro-Inspired Vision/Githubs/DNNforVPL/src/data/stimuli/training_groups3&4/*.TIFF')
            
            file_names = [os.path.basename(x) for x in file_name_paths]
            
            x_val_training, y_val_training, z_val_training, x_tensor_training, y_tensor_training = reading_stimuli(file_names = file_names, file_name_paths = file_name_paths, orientation = Ori_training, spatial_frequency = SF_training)

            print(x_tensor_training.shape, y_tensor_training.shape)
            
            ### SF Transfer Stimuli
            
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
                # file_name_paths = glob.glob(os.path.dirname(os.path.abspath("./")) + '/data/stimuli/transferSF_groups1&2/*.TIFF')
                file_name_paths = glob.glob('D:/ENI Projects/Aslan/Neuro-Inspired Vision/Githubs/DNNforVPL/src/data/stimuli/transferSF_groups1&2/*.TIFF')
            elif group_transfer == 'group3' or group_transfer == 'group4':
                # file_name_paths = glob.glob(os.path.dirname(os.path.abspath("./")) + '/data/stimuli/transferSF_groups3&4/*.TIFF')
                file_name_paths = glob.glob('D:/ENI Projects/Aslan/Neuro-Inspired Vision/Githubs/DNNforVPL/src/data/stimuli/transferSF_groups3&4/*.TIFF')
            
            file_names = [os.path.basename(x) for x in file_name_paths]
            
            x_val_transfer, y_val_transfer, z_val_transfer, x_tensor_transfer, y_tensor_transfer = reading_stimuli(file_names = file_names, file_name_paths = file_name_paths, orientation = Ori_transfer, spatial_frequency = SF_transfer)

            print(x_tensor_transfer.shape, y_tensor_transfer.shape)
            
            for model_counter in range(number_model):                               
                print('DNN Model:   ' + str(model_counter + 1))
                
                # Read the reference image
                # file_name_path_ref = glob.glob(os.path.dirname(os.path.abspath("./")) + '/data/stimuli/reference_stimulus.TIFF')
                file_name_path_ref = glob.glob('D:/ENI Projects/Aslan/Neuro-Inspired Vision/Githubs/DNNforVPL/src/data/stimuli/reference_stimulus.TIFF')
                
                # Define the main reference variable
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
                
                # Convert image to tensor, then normalize and copy it
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
                if model_counter == 0:
                    model = DNNforVPL_1()
                elif model_counter == 1:
                    model = DNNforVPL_2()
                elif model_counter == 2:
                    model = DNNforVPL_3()
                elif model_counter == 3:   
                    model = DNNforVPL_4()
                elif model_counter == 4:    
                    model = DNNforVPL_5()
                    
                model_dict = model.state_dict()
                
                # Filter out unnecessary keys
                pretrained_dict_model = {k : v for k, v in pretrained_dict.items() if k in model_dict}
                # Overwrite entries in the existing state dict
                model_dict.update(pretrained_dict_model)
                # Load the new state dict
                model.load_state_dict(model_dict)
                
                # Initialize by zero the weights of the fully-connected layer of the model
                nn.init.zeros_(model.classifier[0].weight)
                nn.init.zeros_(model.classifier[0].bias)
                
                # Set all the parameters of the model to be trainable
                for param in model.parameters():
                    param.requires_grad = True
                                    
                # Send the model to GPU/CPU
                model = model.to(device)
                
                # Model summary
                print(model)
                    
                cudnn.benchmark = True
                                
                ### Extracting the activations of convolutional layers of the network per transfer stimulus before training
                
                # The indices of consecutive convolutional layers: (0, 3, 6, 8, 10)
                # The sizes of consecutive convolutional layers: (55, 27, 13, 13, 13)
                # The positions of central units of consecutive convolutional layers: (27, 13, 6, 6, 6)
                # The number of channels of consecutive convolutional layers: (64, 192, 384, 256, 256)
                
                os.mkdir(parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/before_training_' + str(model_counter))
                saving_folder = parent_folder + '/Simulation_' + str(simulation_counter + 1) + '/' + group_training + '/before_training_' + str(model_counter)
                
                # The target stimuli
                feature_sample_artiphysiology = np.zeros((num_sample_artiphysiology, 3), dtype = np.int64)
                
                all_x_sample = np.zeros((num_sample_artiphysiology, 3, 224, 224), dtype = np.float32)
                
                if model_counter == 0:
                    all_unit_activity_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64, 55, 55), dtype = np.float32)
                elif model_counter == 1:
                    all_unit_activity_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64, 55, 55), dtype = np.float32)
                    all_unit_activity_Conv2d_2 = np.zeros((num_sample_artiphysiology, 192, 27, 27), dtype = np.float32)
                elif model_counter == 2:
                    all_unit_activity_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64, 55, 55), dtype = np.float32)
                    all_unit_activity_Conv2d_2 = np.zeros((num_sample_artiphysiology, 192, 27, 27), dtype = np.float32)
                    all_unit_activity_Conv2d_3 = np.zeros((num_sample_artiphysiology, 384, 13, 13), dtype = np.float32)
                elif model_counter == 3:   
                    all_unit_activity_Conv2d_1 = np.zeros((num_sample_artiphysiology, 64, 55, 55), dtype = np.float32)
                    all_unit_activity_Conv2d_2 = np.zeros((num_sample_artiphysiology, 192, 27, 27), dtype = np.float32)
                    all_unit_activity_Conv2d_3 = np.zeros((num_sample_artiphysiology, 384, 13, 13), dtype = np.float32)
                    all_unit_activity_Conv2d_4 = np.zeros((num_sample_artiphysiology, 256, 13, 13), dtype = np.float32)
                elif model_counter == 4:    
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
                    
                    all_x_sample[i, :] = x_sample.detach().cpu().clone().numpy()
                                     
                    if model_counter == 0:
                        unit_activity_layer_0 = model.features[0](x_sample)
                        
                        all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                                
                    elif model_counter == 1:
                        unit_activity_layer_0 = model.features[0](x_sample)
                        unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                        unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                        unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                        
                        all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                        
                    elif model_counter == 2:
                        unit_activity_layer_0 = model.features[0](x_sample)
                        unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                        unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                        unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                        unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                        unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                        unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                        
                        all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                                               
                    elif model_counter == 3:   
                        unit_activity_layer_0 = model.features[0](x_sample)
                        unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                        unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                        unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                        unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                        unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                        unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                        unit_activity_layer_7 = model.features[7](unit_activity_layer_6)
                        unit_activity_layer_8 = model.features[8](unit_activity_layer_7)
                        
                        all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                        all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                                                
                    elif model_counter == 4:    
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
                        
                # Saving the properties of sample stimuli used for calculating intrinsic dimension
                scipy.io.savemat(saving_folder + '/feature_sample_artiphysiology.mat', mdict = {'feature_sample_artiphysiology': feature_sample_artiphysiology})
                
                ### Calculating the intrinsic dimension of stimuli
                        
                all_x_sample_ID[model_counter][simulation_counter, group_counter] = estimate(squareform(pdist(all_x_sample.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                
                if model_counter == 0:
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                    
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0]
                                                
                elif model_counter == 1:
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                    
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0]
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, 0]               
                
                elif model_counter == 2:
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                           
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0]
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, 0]
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 2, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, 0]               
                
                elif model_counter == 3:
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                    all_simulation_all_ID[model_counter][simulation_counter, group_counter, 3, 0] = estimate(squareform(pdist(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                            
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, 0]
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, 0]
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 2, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, 0]
                    all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 3, 0] = all_simulation_all_ID[model_counter][simulation_counter, group_counter, 3, 0]
                
                elif model_counter == 4:                        
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
                
                # Define the main learning parameters
                lr = 0.00001
                momentum = 0.9
                weight_decay = 0.0001
                
                # Define the loss function (criterion) and optimizer
                criterion = nn.CrossEntropyLoss().cuda(gpu)
                optimizer = torch.optim.SGD(model.parameters(), lr, momentum = momentum, weight_decay = weight_decay)
                                   
                # Define the main training parameters
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
                    
                            # Compute gradient and perform SGD step
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
                        
                        if (epoch + 1) % 10 == 0:
                            ID_counter = ID_counter + 1
                            
                            for i in range(num_sample_artiphysiology):                    
                                feature_sample_artiphysiology[i, :] = [SF_transfer[x_sample_artiphysiology_index[i, 0]], Ori_transfer[x_sample_artiphysiology_index[i, 1]], x_sample_artiphysiology_index[i, 2]]
                                
                                index = torch.tensor(z_val_transfer[x_sample_artiphysiology_index[i, 0], x_sample_artiphysiology_index[i, 1], x_sample_artiphysiology_index[i, 2]], dtype = torch.long)
                                x_sample = torch.index_select(x_tensor_transfer, 0, index)
                                x_sample = x_sample.cuda(gpu)
                                
                                if model_counter == 0:
                                    unit_activity_layer_0 = model.features[0](x_sample)
                                    
                                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                    
                                elif model_counter == 1:
                                    unit_activity_layer_0 = model.features[0](x_sample)
                                    unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                                    unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                                    unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                                    
                                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                                    
                                elif model_counter == 2:
                                    unit_activity_layer_0 = model.features[0](x_sample)
                                    unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                                    unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                                    unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                                    unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                                    unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                                    unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                                    
                                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                                    
                                elif model_counter == 3:   
                                    unit_activity_layer_0 = model.features[0](x_sample)
                                    unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                                    unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                                    unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                                    unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                                    unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                                    unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                                    unit_activity_layer_7 = model.features[7](unit_activity_layer_6)
                                    unit_activity_layer_8 = model.features[8](unit_activity_layer_7)
                                    
                                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                                    
                                elif model_counter == 4:    
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
                                    
                            ### Calculating the intrinsic dimension
                                    
                            if model_counter == 0:
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                               
                            elif model_counter == 1:
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                
                            elif model_counter == 2:
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                
                            elif model_counter == 3:
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 2, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID[model_counter][simulation_counter, group_counter, 3, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                
                            elif model_counter == 4:                                    
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
                    'optimizer': optimizer.state_dict(),
                }, is_best, group_training, 'DNNforVPL_' + group_training + '.pth.tar')
                
                # Read the reference image
                # file_name_path_ref = glob.glob(os.path.dirname(os.path.abspath("./")) + '/data/stimuli/reference_stimulus.TIFF')
                file_name_path_ref = glob.glob('D:/ENI Projects/Aslan/Neuro-Inspired Vision/Githubs/DNNforVPL/src/data/stimuli/reference_stimulus.TIFF')
                        
                # Define the main reference variable
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
                
                # Convert image to tensor, then normalize and copy it
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
                
                # Set all the parameters of the model to be trainable
                for param in model.parameters():
                    param.requires_grad = False
                
                # Send the model to GPU/CPU
                model = model.to(device)
                
                # Model summary
                print(model)
                
                cudnn.benchmark = True
                
                # Define the main validation parameters
                start_session = 0
                sessions = 10
                    
                for session in range(start_session, sessions):                   
                    z_val_shuffle = copy.deepcopy(z_val_transfer)
                    
                    for j in range(len(SF_transfer)):
                        for k in range(len(Ori_transfer)):
                            random.shuffle(z_val_shuffle[j, k, :])
                
                    # Evaluate on the validation set
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
            
                ### Training with Permuted Labels
                
                print('Training with Permuted Labels')
                
                # Read the reference image
                # file_name_path_ref = glob.glob(os.path.dirname(os.path.abspath("./")) + '/data/stimuli/reference_stimulus.TIFF')
                file_name_path_ref = glob.glob('D:/ENI Projects/Aslan/Neuro-Inspired Vision/Githubs/DNNforVPL/src/data/stimuli/reference_stimulus.TIFF')
                
                # Define the main reference variable
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
                
                # Convert image to tensor, then normalize and copy it
                x_temp = torch.from_numpy(np.transpose(x_val_ref[:, :, :], (2, 0, 1)))
                normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                
                for i in range(len(SF_training) * len(Ori_training)):
                    x_tensor_ref.append(normalize(x_temp))
                    
                x_tensor_ref = torch.stack(x_tensor_ref)
                print(x_tensor_ref.shape)
                
                # Load the PyTorch model
                if model_counter == 0:
                    model = DNNforVPL_1()
                elif model_counter == 1:
                    model = DNNforVPL_2()
                elif model_counter == 2:
                    model = DNNforVPL_3()
                elif model_counter == 3:   
                    model = DNNforVPL_4()
                elif model_counter == 4:    
                    model = DNNforVPL_5()
                
                model_dict = model.state_dict()
                
                # Filter out unnecessary keys
                pretrained_dict_model = {k : v for k, v in pretrained_dict.items() if k in model_dict}
                # Overwrite entries in the existing state dict
                model_dict.update(pretrained_dict_model)
                # Load the new state dict
                model.load_state_dict(model_dict)
                
                # Initialize by zero the weights of the fully-connected layer of the model
                nn.init.zeros_(model.classifier[0].weight)
                nn.init.zeros_(model.classifier[0].bias)
                
                # Set all the parameters of the model to be trainable
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
                    
                # Define the main training parameters
                start_session = 0
                sessions = 1
                
                # Random permutation of labels
                y_tensor_training_permuted = copy.deepcopy(y_tensor_training)
                idx = torch.randperm(y_tensor_training_permuted.nelement())
                y_tensor_training_permuted = y_tensor_training_permuted.view(-1)[idx].view(y_tensor_training_permuted.size())
                    
                for session in range(start_session, sessions):                   
                    # Adjust the learning rate
                    adjust_learning_rate(optimizer, session, lr)
                    
                    # Train on the training set        
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
                    
                            # Compute gradient and perform SGD step
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
                                
                                if model_counter == 0:
                                    unit_activity_layer_0 = model.features[0](x_sample)
                                    
                                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                    
                                elif model_counter == 1:
                                    unit_activity_layer_0 = model.features[0](x_sample)
                                    unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                                    unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                                    unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                                    
                                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                                    
                                elif model_counter == 2:
                                    unit_activity_layer_0 = model.features[0](x_sample)
                                    unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                                    unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                                    unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                                    unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                                    unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                                    unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                                    
                                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                                    
                                elif model_counter == 3:   
                                    unit_activity_layer_0 = model.features[0](x_sample)
                                    unit_activity_layer_1 = model.features[1](unit_activity_layer_0)
                                    unit_activity_layer_2 = model.features[2](unit_activity_layer_1)
                                    unit_activity_layer_3 = model.features[3](unit_activity_layer_2)
                                    unit_activity_layer_4 = model.features[4](unit_activity_layer_3)
                                    unit_activity_layer_5 = model.features[5](unit_activity_layer_4)
                                    unit_activity_layer_6 = model.features[6](unit_activity_layer_5)
                                    unit_activity_layer_7 = model.features[7](unit_activity_layer_6)
                                    unit_activity_layer_8 = model.features[8](unit_activity_layer_7)
                                    
                                    all_unit_activity_Conv2d_1[i, :] = unit_activity_layer_0[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_2[i, :] = unit_activity_layer_3[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_3[i, :] = unit_activity_layer_6[0].detach().cpu().clone().numpy()
                                    all_unit_activity_Conv2d_4[i, :] = unit_activity_layer_8[0].detach().cpu().clone().numpy()
                                    
                                elif model_counter == 4:    
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
                                    
                            if model_counter == 0:
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                               
                            elif model_counter == 1:
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                
                            elif model_counter == 2:
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 2, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                
                            elif model_counter == 3:
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 2, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 3, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                
                            elif model_counter == 4:                                    
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 0, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_1.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 1, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_2.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 2, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_3.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 3, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_4.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]
                                all_simulation_all_ID_permuted[model_counter][simulation_counter, group_counter, 4, ID_counter] = estimate(squareform(pdist(all_unit_activity_Conv2d_5.reshape(num_sample_artiphysiology, -1)), 'euclidean'), fraction = 1.0)[2]                      
                        
    ### Saving the main variables
   
    scipy.io.savemat(parent_folder + '/all_simulation_training_accuracy.mat', mdict = {'all_simulation_training_accuracy': all_simulation_training_accuracy})
    scipy.io.savemat(parent_folder + '/all_simulation_transfer_accuracy.mat', mdict = {'all_simulation_transfer_accuracy': all_simulation_transfer_accuracy})
    scipy.io.savemat(parent_folder + '/all_simulation_all_ID.mat', mdict = {'all_simulation_all_ID': all_simulation_all_ID})
    scipy.io.savemat(parent_folder + '/all_x_sample_ID.mat', mdict = {'all_x_sample_ID': all_x_sample_ID})
    
    scipy.io.savemat(parent_folder + '/all_simulation_training_accuracy_permuted.mat', mdict = {'all_simulation_training_accuracy_permuted': all_simulation_training_accuracy_permuted})
    scipy.io.savemat(parent_folder + '/all_simulation_all_ID_permuted.mat', mdict = {'all_simulation_all_ID_permuted': all_simulation_all_ID_permuted})           