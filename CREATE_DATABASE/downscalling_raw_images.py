#!/usr/bin/env python
# coding: utf-8

# In[51]:


########## VARIABLES  #############
from variables_create_data import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
#Load packages
from torch import nn
import torch
from netCDF4 import Dataset
import numpy as np
import sys
import netCDF4 as nc4

sys.path.append("/home2/datahome/tpicard/PhD_MOMOPAR/TRAIN_AND_VALIDATION_CNN/")

coef_downscalling = 16 # 2 --> 24 km
#coef_downscalling = 20 # 2 --> 40 km
#coef_downscalling = 30 # 2 --> 60 km (13x13)
#coef_downscalling = 40 # 2 --> 100 km (4x4)

def spatial_downscalling(variable,coef):
    
    avgpooling = nn.AvgPool2d(coef)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variable = torch.tensor(variable)
    variable = avgpooling(variable) # 400x400 -> nb_dx * nb_dx
    variable = variable.cpu().detach().numpy()  
    
    return variable

#file_raw_images = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_100_stdepth_3000/raw_images_000710_006470_wsed100_stdepth3000_dx2km_training.nc'
# raw data 
nc = nc4.Dataset(file_raw_images,'r')
temperature = np.asfortranarray(nc.variables['temperature'])
vorticity = np.asfortranarray(nc.variables['vorticity'])
u = np.asfortranarray(nc.variables['u'])
v = np.asfortranarray(nc.variables['v'])
ssh = np.asfortranarray(nc.variables['ssh'])
nc.close()

[ic,jc] = np.load('/home2/datahome/tpicard/Pyticles/Inputs/ic_jc.npy')

sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
from Modules import *
from Modules_gula import *

# Take a time start and a center (ic,jc) that correspond to the PAP station
# Compute ssh, temperature, vorticity, u and v field 
# at the corresponding time 4 vertical layers
# All images are (520 x 520) spatial point resolution

str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-half_reso,jc+half_reso,ic-half_reso,ic+half_reso)
parameters = my_simul +str_para+ format(1900)
simul = load(simul = parameters, floattype=np.float64)

##############################################################
# Define horizontal coordinates (deg, km, or grid points)
########################################################

coord = 'deg'

if coord=='deg':
    #using lon,lat
    lon = simul.x; lat = simul.y
    xlabel = 'lon'; ylabel = 'lat'
elif coord=='km':
    # using km
    [lon,lat] = np.meshgrid(np.arange(simul.x.shape[0])+simul.coord[2],np.arange(simul.x.shape[1])+simul.coord[0])
    lon = (lon/np.mean(simul.pm)*1e-3).T
    lat = (lat/np.mean(simul.pn)*1e-3).T
    xlabel = 'km'; ylabel = 'km'
elif coord=='points':
    # using grid points
    [lon,lat] = np.meshgrid(np.arange(simul.x.shape[0])+simul.coord[2],np.arange(simul.x.shape[1])+simul.coord[0])
    lon,lat = lon.T,lat.T
    xlabel = 'grid pts'; ylabel = 'grid pts'
    
def downscalling_and_interpolation(var_down,var):
    
    var_interpol = np.zeros(var.shape)

    print("Computing interpolation ...")
    
    if len(var.shape)==4:
        
        for level in range(var.shape[1]):

            lat_down_nm = lat_down[:,:][~np.isnan(var_down[0,level,:,:])]
            lon_down_nm = lon_down[:,:][~np.isnan(var_down[0,level,:,:])]

            
            for i in range(var.shape[0]):
                var_down_nm = var_down[i,level,:][~np.isnan(var_down[i,level,:])]
                var_interpol_i = griddata((lon_down_nm,lat_down_nm), var_down_nm, (lon[:,:], lat[:,:]), method='cubic')
                var_interpol_i = np.where(np.isnan(var_interpol_i),var[i,level,:],var_interpol_i)
                var_interpol_i = np.where(np.isnan(var[i,level,:,:]),np.nan,var_interpol_i)
                var_interpol[i,level,:] = var_interpol_i
                
    else:
        lat_down_nm = lat_down[:,:][~np.isnan(var_down[0,:,:])]
        lon_down_nm = lon_down[:,:][~np.isnan(var_down[0,:,:])]
        
        for i in range(var.shape[0]):
            var_down_nm = var_down[i,:][~np.isnan(var_down[i,:])]
            var_interpol_i = griddata((lon_down_nm,lat_down_nm), var_down_nm, (lon[:,:], lat[:,:]), method='cubic')
            var_interpol_i = np.where(np.isnan(var_interpol_i),var[i,:],var_interpol_i)
            var_interpol_i = np.where(np.isnan(var[i,:,:]),np.nan,var_interpol_i)
            var_interpol[i,:] = var_interpol_i
            
    print("---- DONE ----")

    return(var_interpol)

# turn 0 (land) to nan data for interpolation
temperature_origine = temperature.copy()
temperature = np.where(temperature_origine==0,np.nan,temperature)
vorticity = np.where(temperature_origine==0,np.nan,vorticity)
u = np.where(temperature_origine==0,np.nan,u)
v = np.where(temperature_origine==0,np.nan,v)
ssh = np.where(ssh==0,np.nan,ssh)

#Downscalling raw data

temperature_down = spatial_downscalling(temperature,coef_downscalling)
vorticity_down = spatial_downscalling(vorticity,coef_downscalling)
u_down = spatial_downscalling(u,coef_downscalling)
v_down = spatial_downscalling(v,coef_downscalling)
ssh_down = spatial_downscalling(ssh,coef_downscalling)
lon_down = spatial_downscalling(lon[np.newaxis,:,:],coef_downscalling)[0,:]
lat_down = spatial_downscalling(lat[np.newaxis,:,:],coef_downscalling)[0,:]


# In[52]:


def downscalling_and_interpolation(var_down,var):
    
    var_interpol = np.zeros(var.shape)

    print("Computing interpolation ...")
    
    if len(var.shape)==4:
        
        for level in range(var.shape[1]):

            lat_down_nm = lat_down[:,:][~np.isnan(var_down[0,level,:,:])]
            lon_down_nm = lon_down[:,:][~np.isnan(var_down[0,level,:,:])]

            
            for i in range(var.shape[0]):
                var_down_nm = var_down[i,level,:][~np.isnan(var_down[i,level,:])]
                var_interpol_i = griddata((lon_down_nm,lat_down_nm), var_down_nm, (lon[:,:], lat[:,:]), method='cubic')
                #var_interpol_i = np.where(np.isnan(var_interpol_i),var[i,level,:],var_interpol_i)
                var_interpol_i = np.where(np.isnan(var[i,level,:,:]),np.nan,var_interpol_i)
                var_interpol[i,level,:] = var_interpol_i
                
    else:
        lat_down_nm = lat_down[:,:][~np.isnan(var_down[0,:,:])]
        lon_down_nm = lon_down[:,:][~np.isnan(var_down[0,:,:])]
        
        for i in range(var.shape[0]):
            var_down_nm = var_down[i,:][~np.isnan(var_down[i,:])]
            var_interpol_i = griddata((lon_down_nm,lat_down_nm), var_down_nm, (lon[:,:], lat[:,:]), method='cubic')
            #var_interpol_i = np.where(np.isnan(var_interpol_i),var[i,:],var_interpol_i)
            var_interpol_i = np.where(np.isnan(var[i,:,:]),np.nan,var_interpol_i)
            var_interpol[i,:] = var_interpol_i
            
    print("---- DONE ----")

    return(var_interpol)


# In[53]:


# On retranspose sur la grille d'origine
temperature_down = downscalling_and_interpolation(temperature_down,temperature)
vorticity_down = downscalling_and_interpolation(vorticity_down,temperature)
u_down = downscalling_and_interpolation(u_down,temperature)
v_down = downscalling_and_interpolation(v_down,temperature)
ssh_down = downscalling_and_interpolation(ssh_down,ssh)


# In[7]:


dx_reso = coef_downscalling*2 #downscalling to 24
nb_tpas = temperature_down.shape[0]

#test = False

if test == True:
    name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)
else:
    name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_training.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)

    file_raw_images = folder_images+name_raw_images # NAME FILE RAW DATA  

#creating the file
nc = nc4.Dataset(file_raw_images,'w')

#Dimensions used
nc.createDimension('xdim', half_reso*2)
nc.createDimension('ydim', half_reso*2)
nc.createDimension('depth', 4)
nc.createDimension('time', nb_tpas)

#Variables used
nc.createVariable('temperature', 'f4', ('time','depth','xdim', 'ydim'))
nc.createVariable('vorticity', 'f4', ('time','depth','xdim', 'ydim'))
nc.createVariable('u', 'f4', ('time','depth','xdim', 'ydim'))
nc.createVariable('v', 'f4', ('time','depth','xdim', 'ydim'))
nc.createVariable('ssh', 'f4', ('time','xdim', 'ydim'))

nc.variables['temperature'][:] = temperature_down
nc.variables['vorticity'][:] = vorticity_down
nc.variables['u'][:] = u_down
nc.variables['v'][:] = v_down
nc.variables['ssh'][:] = ssh_down

nc.close()


# In[ ]:




