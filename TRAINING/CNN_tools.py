
#Load packages
from netCDF4 import Dataset
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import netCDF4 as nc4
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
from torch import optim
import progressbar
import torch
from torch.utils.data import DataLoader, Dataset
from variables import *

# LOAD ALL THE RAW IMAGES 

def load_raw_images():

    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_date_000000_006480.nc','r')
    temperature = np.asfortranarray(nc.variables['temperature'])
    vorticity = np.asfortranarray(nc.variables['vorticity'])
    u = np.asfortranarray(nc.variables['u'])
    v = np.asfortranarray(nc.variables['v'])
    ssh = np.asfortranarray(nc.variables['ssh'])
    nc.close()

    temperature = np.transpose(temperature, (3, 2, 0,1))
    vorticity = np.transpose(vorticity, (3, 2, 0,1))
    u = np.transpose(u, (3, 2, 0,1))
    v = np.transpose(v, (3, 2, 0,1))
    ssh = np.transpose(ssh, (2, 0, 1))
    ssh = ssh[:, np.newaxis, :, :]

    return(temperature,vorticity,u,v,ssh)

def load_raw_images_simu2():

    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_date_001900_005800_simu2.nc','r')
    temperature = np.asfortranarray(nc.variables['temperature'])
    vorticity = np.asfortranarray(nc.variables['vorticity'])
    u = np.asfortranarray(nc.variables['u'])
    v = np.asfortranarray(nc.variables['v'])
    ssh = np.asfortranarray(nc.variables['ssh'])
    nc.close()

    temperature = np.transpose(temperature, (3, 2, 0,1))
    vorticity = np.transpose(vorticity, (3, 2, 0,1))
    u = np.transpose(u, (3, 2, 0,1))
    v = np.transpose(v, (3, 2, 0,1))
    ssh = np.transpose(ssh, (2, 0, 1))
    ssh = ssh[:, np.newaxis, :, :]

    return(temperature,vorticity,u,v,ssh)


def load_pdf():
    
    print('Loading pdf ...')
    ncfile = 'pdf_8_levels_dt_0000_0107_100dx.nc'
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/pdf/{0}'.format(ncfile),'r')
    pdf = np.asfortranarray(nc.variables['pdf'])
    nc.close()
    print('Pdf loaded')
    return(pdf)


def load_pdf_simu2():

    print('Loading pdf simu2 ...')
    ncfile = 'pdf_8_levels_dt_0000_0065_100dx_simu2.nc'
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/pdf/{0}'.format(ncfile),'r')
    pdf = np.asfortranarray(nc.variables['pdf'][:-4])
    nc.close()
    print('Pdf simu2 loaded')
    return(pdf)


def load_pdf_filter():
    
    print('Loading filtered pdf ...')
    ncfile = 'filter_pdf_8_levels_dt_0000_0107_100dx.nc'
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/pdf/{0}'.format(ncfile),'r')
    pdf_filter = np.asfortranarray(nc.variables['pdf_filter'])
    nc.close()
    print('Pdf filtered loaded')
    return(pdf_filter)

def load_pdf_filter_simu2():

    print('Loading filtered pdf simu2 ...')
    ncfile = 'filter_pdf_8_levels_dt_0000_0065_100dx_simu2.nc'
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/pdf/{0}'.format(ncfile),'r')
    pdf_filter = np.asfortranarray(nc.variables['pdf_filter'][:-4])
    nc.close()
    print('Pdf filtered simu2 loaded')
    return(pdf_filter)


# %%


# step_time de 10 jours
# 0 < step_time < 3??

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
    variable = avgpooling(variable) # 400x400 -> 100x100
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



# %%


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


# %%

def load_image_processed(data_type):
    
    if data_type =='training':
        print('Loading  training images ...')
        (temperature,vorticity,u,v,ssh) = load_raw_images_train()
    elif data_type =='validation':
        print('Loading validation images ...')
        (temperature,vorticity,u,v,ssh) = load_raw_images_validation()
    elif data_type =='all':
        print('Loading all images ...')
        (temperature,vorticity,u,v,ssh) = load_raw_images()
        

    image_norm = image_process(temperature,vorticity,u,v,ssh)
    print('Images loaded')

    return image_norm


# %%



# LOSS FUNCTION AND MODEL : 
# Use torch tensor
# compute only for 1 vertical level  
def Bhatta_loss(Y_hat,Y):

    epsilone = 1e-30
    loss_prediction_test = 1 - torch.sum(torch.sqrt(torch.abs((torch.mul(Y_hat[:,:,:], Y[:,:,:])+epsilone))),axis=(1,2))

    return torch.mean(loss_prediction_test)


def load_pdf_train():
    
    ncfile = 'pdf_8_levels_dt_0000_0107_100dx_dt_correction.nc'
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/pdf/{0}'.format(ncfile),'r')
    pdf = np.asfortranarray(nc.variables['pdf'][index_training_start:index_training_end,:,:,:,:])
    nc.close()
    
    return(pdf)

def load_pdf_filter_train():
    
    ncfile = 'filter_pdf_8_levels_dt_0000_0107_100dx_dt_correction.nc'
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/pdf/{0}'.format(ncfile),'r')
    pdf_filter = np.asfortranarray(nc.variables['pdf_filter'][index_training_start:index_training_end,:,:,:,:])
    nc.close()
    
    return(pdf_filter)


def load_pdf_validation():
    
    ncfile = 'pdf_8_levels_dt_0000_0107_100dx_dt_correction.nc'
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/pdf/{0}'.format(ncfile),'r')
    pdf = np.asfortranarray(nc.variables['pdf'][index_validation_start:index_validation_end,:,:,:,:])
    nc.close()
    
    return(pdf)

def load_pdf_filter_validation():
    
    ncfile = 'filter_pdf_8_levels_dt_0000_0107_100dx_dt_correction.nc'
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/pdf/{0}'.format(ncfile),'r')
    pdf_filter = np.asfortranarray(nc.variables['pdf_filter'][index_validation_start:index_validation_end,:,:,:,:])
    nc.close()
    
    return(pdf_filter)

def load_raw_images_train():

    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_date_000000_006480.nc','r')
    temperature = np.asfortranarray(nc.variables['temperature'][:,:,:,index_training_start:index_training_end+4])
    vorticity = np.asfortranarray(nc.variables['vorticity'][:,:,:,index_training_start:index_training_end+4])
    u = np.asfortranarray(nc.variables['u'][:,:,:,index_training_start:index_training_end+4])
    v = np.asfortranarray(nc.variables['v'][:,:,:,index_training_start:index_training_end+4])
    ssh = np.asfortranarray(nc.variables['ssh'][:,:,index_training_start:index_training_end+4])
    nc.close()

    temperature = np.transpose(temperature, (3, 2, 0,1))
    vorticity = np.transpose(vorticity, (3, 2, 0,1))
    u = np.transpose(u, (3, 2, 0,1))
    v = np.transpose(v, (3, 2, 0,1))
    ssh = np.transpose(ssh, (2, 0, 1))
    ssh = ssh[:, np.newaxis, :, :]

    return(temperature,vorticity,u,v,ssh)

def load_raw_images_validation():

    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_date_000000_006480.nc','r')
    temperature = np.asfortranarray(nc.variables['temperature'][:,:,:,index_validation_start:index_validation_end+4])
    vorticity = np.asfortranarray(nc.variables['vorticity'][:,:,:,index_validation_start:index_validation_end+4])
    u = np.asfortranarray(nc.variables['u'][:,:,:,index_validation_start:index_validation_end+4])
    v = np.asfortranarray(nc.variables['v'][:,:,:,index_validation_start:index_validation_end+4])
    ssh = np.asfortranarray(nc.variables['ssh'][:,:,index_validation_start:index_validation_end+4])
    nc.close()

    temperature = np.transpose(temperature, (3, 2, 0,1))
    vorticity = np.transpose(vorticity, (3, 2, 0,1))
    u = np.transpose(u, (3, 2, 0,1))
    v = np.transpose(v, (3, 2, 0,1))
    ssh = np.transpose(ssh, (2, 0, 1))
    ssh = ssh[:, np.newaxis, :, :]

    return(temperature,vorticity,u,v,ssh)

def load_images_validation():
    print('Loading validation data ...')
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_validation_data_no_normalize.nc','r')
    images = np.asfortranarray(nc.variables['images'])
    nc.close()
    print('Done')
    return(images)

def load_images_validation_surface():
    print('Loading validation surface data ...')
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_validation_data_no_normalize.nc','r')
    images = np.asfortranarray(nc.variables['images'][:,index_surface,:,:])
    nc.close()
    print('Done')
    return(images)

def load_images_train(nb):

    print('Loading training data ...')
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_train_data_no_normalize.nc','r')
    if nb =='all':
        images = np.asfortranarray(nc.variables['images'])
    elif nb=='half':
        images = np.asfortranarray(nc.variables['images'][::2])
    else :
        images = np.asfortranarray(nc.variables['images'][::nb])
    nc.close()
    print('Done')
    return(images)

def load_images_train_surface(nb):

    print('Loading training surface data ...')
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_train_data_no_normalize.nc','r')
    if nb =='all':
        images = np.asfortranarray(nc.variables['images'][:,index_surface,:,:])
    elif nb=='half':
        images = np.asfortranarray(nc.variables['images'][::2,index_surface,:,:])
    nc.close()
    print('Done')
    return(images)

def load_images_simu2(nb):

    print('Loading simu2 images ...')
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_simu2_data_no_normalize.nc','r')
    if nb =='all':
        images = np.asfortranarray(nc.variables['images'][:,:,:,:])
    elif nb=='half':
        images = np.asfortranarray(nc.variables['images'][::2,:,:,:])
    elif nb=='cinquieme':
        images = np.asfortranarray(nc.variables['images'][::5,:,:,:])
    elif nb=='2cases':
        images = np.asfortranarray(nc.variables['images'][:36*2,:,:,:])
        
    nc.close()
    print('Done')
    return(images)
 
def load_images_simu2_surface(nb):

    print('Loading simu2 surface data ...')
    nc = nc4.Dataset('/home2/datawork/tpicard/DATA_CNN/image_inputs/inputs_simu2_data_no_normalize.nc','r')
    if nb =='all':
        images = np.asfortranarray(nc.variables['images'][:,index_surface,:,:])
    elif nb=='half':
        images = np.asfortranarray(nc.variables['images'][::2,index_surface,:,:])
    elif nb=='cinquieme':
        images = np.asfortranarray(nc.variables['images'][::5,index_surface,:,:])
    elif nb=='2cases':
        images = np.asfortranarray(nc.variables['images'][:36*2,index_surface,:,:])
    nc.close()
    print('Done')
    return(images)

def load_data_train(nb):

    image_train = load_images_train(nb)
    # Normalization of inputs
    Normalize = transforms.Normalize(list_mean_train, list_std_train)
    image_train = Normalize(torch.tensor(image_train))

    pdf_train = load_pdf_train()
    pdf_filter_train = load_pdf_filter_train()

    pdf_train = load_pdf_train()
    pdf_filter_train = load_pdf_filter_train()

    pdf_train = np.transpose(pdf_train,(0,2,1,3,4))
    pdf_train = pdf_train.reshape(pdf_train.shape[0]*pdf_train.shape[1],8,100,100)

    pdf_filter_train = np.transpose(pdf_filter_train,(0,2,1,3,4))
    pdf_filter_train = pdf_filter_train.reshape(pdf_filter_train.shape[0]*pdf_filter_train.shape[1],8,100,100)
   
    if nb =='half':
        pdf_train = pdf_train[::2]
        pdf_filter_train = pdf_filter_train[::2]
    elif nb=='cinquieme':
        pdf_train = pdf_train[::5]
        pdf_filter_train = pdf_filter_train[::5]

    return(image_train,pdf_train,pdf_filter_train)

def load_data_eval():

    #image_norm_eval = load_image_processed('validation')
    #image_eval = load_images_validation()
    image_eval = load_images_validation()
    # Normalization of inputs
    Normalize = transforms.Normalize(list_mean_train, list_std_train)
    image_eval = Normalize(torch.tensor(image_eval))

    pdf_eval = load_pdf_validation()
    pdf_filter_eval = load_pdf_filter_validation()


    pdf_eval = np.transpose(pdf_eval,(0,2,1,3,4))
    pdf_eval = pdf_eval.reshape(pdf_eval.shape[0]*pdf_eval.shape[1],8,100,100)

    pdf_filter_eval = np.transpose(pdf_filter_eval,(0,2,1,3,4))
    pdf_filter_eval = pdf_filter_eval.reshape(pdf_filter_eval.shape[0]*pdf_filter_eval.shape[1],8,100,100)

    return(image_eval,pdf_eval,pdf_filter_eval)

def load_data_test():


    image_test = load_images_simu2('cinquieme')

    pdf_test = load_pdf_simu2()
    pdf_filter_test = load_pdf_filter_simu2()

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,100,100)
    pdf_test = pdf_test[::5]

    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,100,100)
    pdf_filter_test = pdf_filter_test[::5]

    # Normalization of inputs
    Normalize = transforms.Normalize(list_mean_train, list_std_train)
    image_test = Normalize(torch.tensor(image_test))

    return(image_test,pdf_test,pdf_filter_test)

def load_data_train_surface(nb):

    image_train = load_images_train_surface(nb)
    # Normalization of inputs
    Normalize = transforms.Normalize(list_mean_train_surface, list_std_train_surface)
    image_train = Normalize(torch.tensor(image_train))

    pdf_train = load_pdf_train()
    pdf_filter_train = load_pdf_filter_train()

    pdf_train = load_pdf_train()
    pdf_filter_train = load_pdf_filter_train()

    pdf_train = np.transpose(pdf_train,(0,2,1,3,4))
    pdf_train = pdf_train.reshape(pdf_train.shape[0]*pdf_train.shape[1],8,100,100)

    pdf_filter_train = np.transpose(pdf_filter_train,(0,2,1,3,4))
    pdf_filter_train = pdf_filter_train.reshape(pdf_filter_train.shape[0]*pdf_filter_train.shape[1],8,100,100)
   
    if nb =='half':
        pdf_train = pdf_train[::2]
        pdf_filter_train = pdf_filter_train[::2]
    elif nb=='cinquieme':
        pdf_train = pdf_train[::5]
        pdf_filter_train = pdf_filter_train[::5]

    return(image_train,pdf_train,pdf_filter_train)

def load_data_eval_surface():

    #image_norm_eval = load_image_processed('validation')
    #image_eval = load_images_validation()
    image_eval = load_images_validation_surface()
        
    Normalize = transforms.Normalize(list_mean_train_surface, list_std_train_surface)
    image_eval = Normalize(torch.tensor(image_eval))

    pdf_eval = load_pdf_validation()
    pdf_filter_eval = load_pdf_filter_validation()


    pdf_eval = np.transpose(pdf_eval,(0,2,1,3,4))
    pdf_eval = pdf_eval.reshape(pdf_eval.shape[0]*pdf_eval.shape[1],8,100,100)

    pdf_filter_eval = np.transpose(pdf_filter_eval,(0,2,1,3,4))
    pdf_filter_eval = pdf_filter_eval.reshape(pdf_filter_eval.shape[0]*pdf_filter_eval.shape[1],8,100,100)

    return(image_eval,pdf_eval,pdf_filter_eval)

def load_data_test_surface():


    image_test = load_images_simu2_surface('cinquieme')

    pdf_test = load_pdf_simu2()
    pdf_filter_test = load_pdf_filter_simu2()

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,100,100)
    pdf_test = pdf_test[::5]

    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,100,100)
    pdf_filter_test = pdf_filter_test[::5]

    # Normalization of inputs
    Normalize = transforms.Normalize(list_mean_train_surface, list_std_train_surface)
    image_test = Normalize(torch.tensor(image_test))

    return(image_test,pdf_test,pdf_filter_test)


def load_data_test_surface_all():


    image_test = load_images_simu2_surface('all')

    pdf_test = load_pdf_simu2()
    pdf_filter_test = load_pdf_filter_simu2()

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,100,100)

    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,100,100)

    # Normalization of inputs
    Normalize = transforms.Normalize(list_mean_train_surface, list_std_train_surface)
    image_test = Normalize(torch.tensor(image_test))

    return(image_test,pdf_test,pdf_filter_test)

def load_data_test_all():


    image_test = load_images_simu2('all')

    pdf_test = load_pdf_simu2()
    pdf_filter_test = load_pdf_filter_simu2()

    pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
    pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,100,100)

    pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
    pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,100,100)
    
    # Normalization of inputs
    Normalize = transforms.Normalize(list_mean_train, list_std_train)
    image_test = Normalize(torch.tensor(image_test))

    return(image_test,pdf_test,pdf_filter_test)
