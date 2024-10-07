
#Load packages
from netCDF4 import Dataset
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import netCDF4 as nc4
from torch import nn
from torch import optim
import progressbar
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
from torch import optim
import progressbar
import torch
from torch.utils.data import DataLoader, Dataset
from variables import *


### LOAD ALL THE RAW IMAGES ###

def load_raw_images():

    print('Loading raw images ...')
    print(file_raw_images)
    nc = nc4.Dataset(file_raw_images,'r')
    temperature = np.asfortranarray(nc.variables['temperature'])
    vorticity = np.asfortranarray(nc.variables['vorticity'])
    u = np.asfortranarray(nc.variables['u'])
    v = np.asfortranarray(nc.variables['v'])
    ssh = np.asfortranarray(nc.variables['ssh'])
    nc.close()
    print('Raw images loaded')
    
    return(temperature,vorticity,u,v,ssh)

def load_pdf():
    
    print('Loading pdf ...')
    print(file_pdf)
    nc = nc4.Dataset(file_pdf,'r')
    pdf = np.asfortranarray(nc.variables['pdf'])
    nc.close()
    print('Pdf loaded')
    return(pdf)

def load_pdf_filter():
    
    print('Loading filtered pdf ...')
    print(file_pdf_filter)
    nc = nc4.Dataset(file_pdf_filter,'r')
    pdf_filter = np.asfortranarray(nc.variables['pdf_filter'])
    nc.close()
    print('Pdf filtered loaded')
    return(pdf_filter)

def load_images():
    print('Loading images data ...')
    print(file_input)
    nc = nc4.Dataset(file_input,'r')
    images = np.asfortranarray(nc.variables['images'])[:,:,:,:]
    nc.close()
    print('Done')
    return(images)


[ic,jc] = np.load('/home2/datahome/tpicard/Pyticles/Inputs/ic_jc.npy')
ic_all_list = -(ic - np.linspace(1521,1611,6).astype(int)) ### WARNING BIAS OF 1km ###
jc_all_list = -(jc - np.linspace(570,660,6).astype(int))

def image_selector(pos,var):
    pos_j = pos%6
    pos_i = pos//6
    d_ic = ic_all_list[pos_i]
    d_jc = jc_all_list[pos_j]
    var = var[:,60+d_ic:-60+d_ic,60+d_jc:-60+d_jc]
    return(var)

# APPLY A SPATIAL DOWNSCALLING TO RAW IMAGES 
import torch
from torch import nn

def spatial_downscalling(variable,coef):
    
    avgpooling = nn.AvgPool2d(coef)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variable = torch.tensor(variable)
    variable = avgpooling(variable) # 400x400 -> dx_pdf x dx_pdf
    variable = variable.cpu().detach().numpy()  
    
    return variable

def Normalization(variable):
    mean = [np.nanmean(variable)]*variable.shape[1]
    std = [np.nanstd(variable)]*variable.shape[1]
    # Normalization of varible
    Normalize = transforms.Normalize(mean, std)
    variable = Normalize(torch.tensor(variable))
    return variable.cpu().detach().numpy()

def nan_to_0(variable):
    return np.nan_to_num(variable)



def image_process(temperature,vorticity,u,v,ssh):
    
    # Normalization of images
    temperature_norm = Normalization(temperature)
    vorticity_norm = Normalization(vorticity)
    u_norm = Normalization(u)
    v_norm = Normalization(v)
    ssh_norm = Normalization(ssh)
    
    #Nan removed
    temperature_norm = nan_to_0(temperature_norm)
    vorticity_norm = nan_to_0(vorticity_norm)
    u_norm = nan_to_0(u_norm)
    v_norm = nan_to_0(v_norm)
    ssh_norm = nan_to_0(ssh_norm)
    
    # Compilation of all the variable into one vector
    image_norm = np.concatenate((temperature_norm,vorticity_norm,u_norm,v_norm,ssh_norm),axis=1)
    
    return image_norm

def load_image_processed(data_type):

    print('Loading all images ...')
    (temperature,vorticity,u,v,ssh) = load_raw_images()
    image_norm = image_process(temperature,vorticity,u,v,ssh)
    print('Images loaded')

    return image_norm

# LOSS FUNCTION AND MODEL : 
# Use torch tensor
# compute only for 1 vertical level  
def Bhatta_loss(Y_hat,Y):

    epsilone = 1e-30
    loss_prediction_test = 1 - torch.sum(torch.sqrt(torch.abs((torch.mul(Y_hat[:,:,:], Y[:,:,:])+epsilone))),axis=(1,2))

    return torch.mean(loss_prediction_test)

#PDF TRAIN

def load_pdf_train():
    
    print('Loading pdf for training')
    print(file_pdf)
    nc = nc4.Dataset(file_pdf,'r')
    pdf = np.asfortranarray(nc.variables['pdf'][index_training_start:index_training_end,:,:,:,:])
    nc.close()
    print('pdf for training loaded')
    return(pdf)

def load_pdf_filter_train():
    
    print('Loading pdf filter for training')
    print(file_pdf_filter)
    nc = nc4.Dataset(file_pdf_filter,'r')
    pdf_filter = np.asfortranarray(nc.variables['pdf_filter'][index_training_start:index_training_end,:,:,:,:])
    nc.close()
    print('pdf filter for training loaded')
    return(pdf_filter)

# PDF VALIDATION

def load_pdf_validation():
    
    print('Loading pdf for validation')
    print(file_pdf)
    nc = nc4.Dataset(file_pdf,'r')
    pdf = np.asfortranarray(nc.variables['pdf'][index_validation_start:index_validation_end,:,:,:,:])
    nc.close()
    print('pdf for validation loaded')    
    return(pdf)

def load_pdf_filter_validation():
    
    print('Loading pdf filter for validation')
    print(file_pdf_filter)
    nc = nc4.Dataset(file_pdf_filter,'r')
    pdf_filter = np.asfortranarray(nc.variables['pdf_filter'][index_validation_start:index_validation_end,:,:,:,:])
    nc.close()
    print('pdf filter for validation loaded')
    return(pdf_filter)


#PDF TEST

def load_pdf_test():
    
    print('Loading pdf for test')
    print(file_pdf_test)
    nc = nc4.Dataset(file_pdf_test,'r')
    pdf = np.asfortranarray(nc.variables['pdf'][:,:,:,:,:])
    nc.close()
    print('pdf for test loaded')    
    return(pdf)

def load_pdf_filter_test():
    
    print('Loading pdf filter for test')
    print(file_pdf_filter_test)
    nc = nc4.Dataset(file_pdf_filter_test,'r')
    pdf_filter = np.asfortranarray(nc.variables['pdf_filter'][:,:,:,:,:])
    nc.close()
    print('pdf filter for test loaded')
    return(pdf_filter)


# IMAGE VALIDATION

def load_images_validation():
    print('Loading validation data ...')
    index_start = index_validation_start*36
    index_end = index_validation_end*36
    nc = nc4.Dataset(file_input,'r')
    images = np.asfortranarray(nc.variables['images'])[index_start:index_end,:,:,:]
    nc.close()
    print('Done')
    return(images)

def load_images_validation_surface():
    print('Loading validation surface data ...')
    index_start = index_validation_start*36
    index_end = index_validation_end*36
    nc = nc4.Dataset(file_input,'r')
    images = np.asfortranarray(nc.variables['images'][index_start:index_end,index_surface,:,:])
    nc.close()
    print('Done')
    return(images)

# IMAGE TRAIN

def load_images_train():

    print('Loading training 4L data ...')
    index_start = index_training_start*36
    index_end = index_training_end*36
    nc = nc4.Dataset(file_input,'r')
    images = np.asfortranarray(nc.variables['images'][index_start:index_end,:,:,:])
    print('Done')
    return(images)

def load_images_train_surface():

    print('Loading training surface data ...')
    index_start = index_training_start*36
    index_end = index_training_end*36
    nc = nc4.Dataset(file_input,'r')
    images = np.asfortranarray(nc.variables['images'][index_start:index_end,index_surface,:,:])
    print('Done')
    return(images)

#IMAGE TEST

def load_images_test():

    print('Loading testing 4L data ...')
    nc = nc4.Dataset(file_input_test,'r')
    images = np.asfortranarray(nc.variables['images'][:,:,:,:])
    print('Done')
    return(images)

def load_images_test_surface():

    print('Loading testing surface data ...')
    nc = nc4.Dataset(file_input_test,'r')
    images = np.asfortranarray(nc.variables['images'][:,index_surface,:,:])
    print('Done')
    return(images)

def load_images_test_ssh_j20():

    print('Loading testing ssh data ...')
    nc = nc4.Dataset(file_input_test,'r')
    images = np.asfortranarray(nc.variables['images'][:,-10,:,:])
    print('Done')
    return(images)


### ADD LIST_MEAN_TRAIN AND STD ####

def load_data_train_eval():

    ### TRAIN DATA ###
    
    image_train = load_images_train()
    # Normalization of inputs
    list_mean_train = np.mean(image_train,axis=(0,2,3))
    list_std_train = np.std(image_train,axis=(0,2,3))
    
    Normalize = transforms.Normalize(list_mean_train, list_std_train)
    image_train = Normalize(torch.tensor(image_train))

    pdf_train = load_pdf_train()
    pdf_filter_train = load_pdf_filter_train()

    pdf_train = load_pdf_train()
    pdf_filter_train = load_pdf_filter_train()

    pdf_train = np.transpose(pdf_train,(0,2,1,3,4))
    pdf_train = pdf_train.reshape(pdf_train.shape[0]*pdf_train.shape[1],8,dx_pdf,dx_pdf)
    
    pdf_filter_train = np.transpose(pdf_filter_train,(0,2,1,3,4))
    pdf_filter_train = pdf_filter_train.reshape(pdf_filter_train.shape[0]*pdf_filter_train.shape[1],8,dx_pdf,dx_pdf)

    
    ### EVALUATION DATA ###
    image_eval = load_images_validation()
    # Normalization of inputs
    
    Normalize = transforms.Normalize(list_mean_train, list_std_train)
    image_eval = Normalize(torch.tensor(image_eval))

    pdf_eval = load_pdf_validation()
    pdf_filter_eval = load_pdf_filter_validation()


    pdf_eval = np.transpose(pdf_eval,(0,2,1,3,4))
    pdf_eval = pdf_eval.reshape(pdf_eval.shape[0]*pdf_eval.shape[1],8,dx_pdf,dx_pdf)

    pdf_filter_eval = np.transpose(pdf_filter_eval,(0,2,1,3,4))
    pdf_filter_eval = pdf_filter_eval.reshape(pdf_filter_eval.shape[0]*pdf_filter_eval.shape[1],8,dx_pdf,dx_pdf)
    
    return(image_train,pdf_train,pdf_filter_train,image_eval,pdf_eval,pdf_filter_eval)

def load_data_train_eval_surface():

    ### TRAIN DATA ###
    
    image_train = load_images_train_surface()
    # Normalization of inputs
    print('Normalization ...')
    
    list_mean_train = np.mean(image_train,axis=(0,2,3))
    list_std_train = np.std(image_train,axis=(0,2,3))
    
    Normalize = transforms.Normalize(list_mean_train, list_std_train)
    image_train = Normalize(torch.tensor(image_train))
    
    print('Normalization done')
    
    pdf_train = load_pdf_train()
    pdf_filter_train = load_pdf_filter_train()

    pdf_train = load_pdf_train()
    pdf_filter_train = load_pdf_filter_train()

    pdf_train = np.transpose(pdf_train,(0,2,1,3,4))
    pdf_train = pdf_train.reshape(pdf_train.shape[0]*pdf_train.shape[1],8,dx_pdf,dx_pdf)
    
    pdf_filter_train = np.transpose(pdf_filter_train,(0,2,1,3,4))
    pdf_filter_train = pdf_filter_train.reshape(pdf_filter_train.shape[0]*pdf_filter_train.shape[1],8,dx_pdf,dx_pdf)

    
    ### EVALUATION DATA ###
    image_eval = load_images_validation_surface()
    # Normalization of inputs
    
    Normalize = transforms.Normalize(list_mean_train, list_std_train)
    image_eval = Normalize(torch.tensor(image_eval))

    pdf_eval = load_pdf_validation()
    pdf_filter_eval = load_pdf_filter_validation()


    pdf_eval = np.transpose(pdf_eval,(0,2,1,3,4))
    pdf_eval = pdf_eval.reshape(pdf_eval.shape[0]*pdf_eval.shape[1],8,dx_pdf,dx_pdf)

    pdf_filter_eval = np.transpose(pdf_filter_eval,(0,2,1,3,4))
    pdf_filter_eval = pdf_filter_eval.reshape(pdf_filter_eval.shape[0]*pdf_filter_eval.shape[1],8,dx_pdf,dx_pdf)
    
    return(image_train,pdf_train,pdf_filter_train,image_eval,pdf_eval,pdf_filter_eval)


### TO CHANGE ###

def load_data_test():

    ### TRAIN DATA ###
    
    print('Import data and normalization ...')
    
    image_train = load_images_train()
    # Normalization of inputs
    
    list_mean_train = np.mean(image_train,axis=(0,2,3))
    list_std_train = np.std(image_train,axis=(0,2,3))
    
    Normalize = transforms.Normalize(list_mean_train, list_std_train)

    
    del image_train
    
    image_test = load_images_test()
    image_test = Normalize(torch.tensor(image_test))
    
    print('---- Done -----')
        
    pdf_test = load_pdf_test()
    pdf_filter_test = load_pdf_filter_test()

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,dx_pdf,dx_pdf)
    
    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,dx_pdf,dx_pdf)
    
    return(image_test,pdf_test,pdf_filter_test)

def load_data_test_surface():

    ### TRAIN DATA ###
    
    image_train = load_images_train_surface()
    
    print('Import data and normalization surface ...')
        
    # Normalization of inputs
    list_mean_train = np.mean(image_train,axis=(0,2,3))
    list_std_train = np.std(image_train,axis=(0,2,3))
    
    Normalize = transforms.Normalize(list_mean_train, list_std_train)

    del image_train
    
    print('---- Done -----')
        
    image_test = load_images_test_surface()
    image_test = Normalize(torch.tensor(image_test))
    
    pdf_test = load_pdf_test()
    pdf_filter_test = load_pdf_filter_test()

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,dx_pdf,dx_pdf)
    
    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,dx_pdf,dx_pdf)
    
    return(image_test,pdf_test,pdf_filter_test)


def load_data_test_no_norm():

    ### TRAIN DATA ###
        
    image_test = load_images_test()
    
    pdf_test = load_pdf_test()
    pdf_filter_test = load_pdf_filter_test()

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,dx_pdf,dx_pdf)
    
    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,dx_pdf,dx_pdf)
    
    print('---- Done -----')
    
    return(image_test,pdf_test,pdf_filter_test)

def load_data_test_no_norm_surface():

    ### TRAIN DATA ###
    
        
    image_test = load_images_test_surface()
    
    pdf_test = load_pdf_test
    pdf_filter_test = load_pdf_filter_test()

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,dx_pdf,dx_pdf)
    
    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,dx_pdf,dx_pdf)
    
    print('---- Done -----')
    
    return(image_test,pdf_test,pdf_filter_test)

def load_data_test_fast_load():

    ### TRAIN DATA ###
    
    print('Loading 100 testing data ...')
    nc = nc4.Dataset(file_input_test,'r')
    image_test = np.asfortranarray(nc.variables['images'][:36*3,:,:,:])
    print('Done')

    print('Loading 100 pdf test')
    nc = nc4.Dataset(file_pdf_test,'r')
    pdf_test = np.asfortranarray(nc.variables['pdf'][:3,:,:,:,:])
    nc.close()
    print('pdf for test loaded')    
    
    print('Loading 100 pdf filter for test')
    nc = nc4.Dataset(file_pdf_filter_test,'r')
    pdf_filter_test = np.asfortranarray(nc.variables['pdf_filter'][:3,:,:,:,:])
    nc.close()
    print('pdf filter for test loaded')

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,dx_pdf,dx_pdf)
    
    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,dx_pdf,dx_pdf)
    
    print('---- Done -----')
    
    return(image_test,pdf_test,pdf_filter_test)