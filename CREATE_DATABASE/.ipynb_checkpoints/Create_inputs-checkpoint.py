#!/usr/bin/env python
# coding: utf-8

# In[1]:

########## VARIABLES  #############
from variables_create_data import *

#Load packages
from torch import nn
import torch
from netCDF4 import Dataset
import numpy as np
import sys
import netCDF4 as nc4
sys.path.append("/home2/datahome/tpicard/PhD_MOMOPAR/TRAIN_AND_VALIDATION_CNN/")


# In[2]:

[ic,jc] = np.load('/home2/datahome/tpicard/Pyticles/Inputs/ic_jc.npy')
ic_all_list = -(ic - np.linspace(1521,1611,6).astype(int)) ### WARNING BIAS OF 1km ###
jc_all_list = -(jc - np.linspace(570,660,6).astype(int))

def spatial_downscalling(variable,coef):
    
    avgpooling = nn.AvgPool2d(coef)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variable = torch.tensor(variable)
    variable = avgpooling(variable) # 400x400 -> 100x100
    variable = variable.cpu().detach().numpy()  
    
    return variable

def image_selector(pos,var):
    pos_j = pos%6
    pos_i = pos//6
    d_ic = ic_all_list[pos_i]
    d_jc = jc_all_list[pos_j]
    var = var[:,60+d_ic:-60+d_ic,60+d_jc:-60+d_jc]

    return(var)

def get_images_exp(dt,pos,raw_images):

    print(raw_images.shape)
    images_exp = raw_images[dt:dt+nb_sample,:,:,:]
    images_exp = images_exp[:,:,:,:].reshape(zdim,half_reso*2,half_reso*2) #number_time_step*number_vertical_levels*number_variable3D + number_time_step
    images_exp = image_selector(pos,images_exp)
    images_exp = spatial_downscalling(images_exp,coef_pooling)
        
        
    return(images_exp)

def image_concat(temperature,vorticity,u,v,ssh):

    #Nan removed
    temperature_norm = np.nan_to_num(temperature)
    vorticity_norm = np.nan_to_num(vorticity)
    u_norm = np.nan_to_num(u)
    v_norm = np.nan_to_num(v)
    ssh_norm = np.nan_to_num(ssh)

    # Compilation of all the variable into one vector
    image_norm = np.concatenate((temperature_norm,vorticity_norm,u_norm,v_norm,ssh_norm),axis=1)

    return image_norm

# Train data 

nc = nc4.Dataset(file_raw_images,'r')
temperature = np.asfortranarray(nc.variables['temperature'])
vorticity = np.asfortranarray(nc.variables['vorticity'])
u = np.asfortranarray(nc.variables['u'])
v = np.asfortranarray(nc.variables['v'])
ssh = np.asfortranarray(nc.variables['ssh'])
nc.close()

ssh = ssh[:, np.newaxis, :, :]

images = image_concat(temperature,vorticity,u,v,ssh)
nb_dt = images.shape[0] - nb_sample + 1  # (120-60) = 60/20 = 3 + 1
nb_cases = nb_dt*36 

file = file_input

nc = nc4.Dataset(file,'w')
#Dimensions used
nc.createDimension('size', nb_dx)
nc.createDimension('zdim', zdim)
nc.createDimension('nb_case',nb_cases)
nc.createVariable('images', 'f4', ('nb_case','zdim','size', 'size'))
nc.close()

t=0
for i in range(nb_dt):
    for pos in range(0,36):
        images_i = get_images_exp(i,pos,images)
        nc = nc4.Dataset(file,'r+')
        nc.variables['images'][t,:,:,:] = images_i
        nc.close()
        t=t+1
    #print(str(i)+' Done')


print('Input X of the model ready created at:')
print(file)