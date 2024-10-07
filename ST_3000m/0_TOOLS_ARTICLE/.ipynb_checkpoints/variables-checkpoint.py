#Load packages

"""
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
from CNN_tools import *
from CNN_UNET import *
from pytorch_lightning.callbacks import ModelCheckpoint
from DATALOADER import Pdf_Image_DataSet
#from variables_training import *
#from variables import *
import importlib
#from Modules import *
#from Modules_gula import *
"""

import numpy as np
[ic,jc] = np.load('/home2/datahome/tpicard/Pyticles/Inputs/ic_jc.npy')
[ic_ori,jc_ori] = np.load('/home2/datahome/tpicard/Pyticles/Inputs/ic_jc.npy')



#### DO NOT EDIT YET ##### 
half_reso = 260 # = 520 point square centered at PAP = 1040 km square
coef_pooling = 4 # Resolution = 2km x coef pooling
nb_dx = 100 # Final resolution = ((half_reso*2)-120)/coef_pooling
dx_pdf = 100
dt_exp = 60 # 30 days time step (time that separate 2 experiences)
dt_image_sampling = 20 #images are sampled every 10 days
test = False # training data
dx_reso = 2 #Choose resolution of the inputs


# ############ TO EDIT ##########################################################

# ############ TRAINING PARAMETERS ##############################################

batch_size = 32
max_epochs = 50
num_workers=0
nb_gpus = 1
alpha1 = 0.8
alpha2 = 1 - alpha1

# dirSAVE = './saved_model/supermodel_wsed_{0}_sdepth{1}/'.format(wsed,depth_trap)
# name_model = 'CNN_UNET_k{0}_p{1}_b{2}_d{3}_nl{4}_ni_{5}_dx{6}km'.format(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs,dx_reso)


test = True # Test or training data ?
depth_cst = False # False = adapt 4 vertical levels depending on the trap depth
nb_sample = 5 # Number of images per experiment 
zdim = nb_sample*4*4+nb_sample #Number of images per experience (4levels)
full_time_exp = 120 # 50 days time step (time of 1 experience)
wsed = 100 # sinking speed of particles
depth_trap = 3000 #Depth of the sediment trap
list_level_cst=True # if true, compute pdf at : [900m,800m,...,200m], else compute pdf at list_level = np.linspace(200,depth_trap,9)[1:]

# folder_pdf = '/home2/scratch/tpicard/DATA_CNN/pdf/'
# folder_images = '/home2/scratch/tpicard/DATA_CNN/image_inputs/'

folder_pdf = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_{0}_stdepth_{1}/'.format(wsed,depth_trap)
folder_images = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_{0}_stdepth_{1}/'.format(wsed,depth_trap)

if test ==True:
    tpas_start = 0 # First experience ?
    tpas_end = 65 #65 Number of experiences
    date_start = 1900
    date_end = date_start + full_time_exp + 60*(tpas_end-1) # date end
    folder_in_pyticle = '/home2/scratch/tpicard/Pyticles/outputs_simu2/'
    my_simul = 'aperitif_simu2'
    name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)
    #name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_testing.nc'.format(date_start,date_end,wsed,depth_trap)
    name_pdf ='pdf_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_pdf)
    name_pdf_filter = 'filter_'+name_pdf
    name_input = 'inputs_{0:06}_{1:06}_wsed{2}_stdepth{3}_zdim{4}_dx{5}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,zdim,dx_reso)

else: 
    tpas_start = 0
    tpas_end = 95 #95
    date_start = 710
    date_end = date_start + full_time_exp + 60*(tpas_end-1) # date end
    folder_in_pyticle = '/home2/scratch/tpicard/Pyticles/outputs/'
    my_simul = 'apero'
    name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_training.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)
    #name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_training.nc'.format(date_start,date_end,wsed,depth_trap)
    name_pdf ='pdf_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}_training.nc'.format(date_start,date_end,wsed,depth_trap,dx_pdf)
    name_pdf_filter = 'filter_'+name_pdf
    name_input = 'inputs_{0:06}_{1:06}_wsed{2}_stdepth{3}_zdim{4}_dx{5}km_training.nc'.format(date_start,date_end,wsed,depth_trap,zdim,dx_reso)

file_raw_images = folder_images+name_raw_images # NAME FILE RAW DATA  
file_input = folder_images+name_input
file_pdf = folder_pdf + name_pdf
file_pdf_filter = folder_pdf + name_pdf_filter


def spatial_filter(px,py,pxcenter,pycenter,index_start):
    
    #On garde que le premier pas de temps
    px = px[0,0:10201]
    py = py[0,0:10201]
    
    a = np.where(abs(px - pxcenter) <= 3,1,0) # Filter on lon
    b = np.where(abs(py - pycenter) <= 3,1,0) # Filter on lat
    c = np.multiply(a,b) 
    index = np.argwhere(c == 1) # Combinason of both filter
    #print(index.shape)
    index_tot = []
    
    for i in range(index_start,index_start+20): # We consider all the particles released for a 10 days perdiod
        index_temp = index + 10201*(i)
        index_tot.append(index_temp)
        
    return np.array(index_tot).ravel()




def compute_variable(tstart,ic,jc):
    
    my_simul = 'apero'

    #parameters = my_simul + ' [1068,2068,117,1117,[1,100,1]] '+ format(date_plot_AC)
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-200,jc+200,ic-200,ic+200)
    parameters = my_simul +str_para+ format(tstart)
    simul = load(simul = parameters, floattype=np.float64)
    print(simul.date)

    depth = [0,-200,-500,-1000]
    temp = var('temp',simul,depths=depth).data
    ssh = var('zeta',simul).data
    u = var('u',simul,depths=depth).data
    v = var('v',simul,depths=depth).data


    
    vrt = np.zeros(temp.shape)
    for i in range(len(depth)):
        vrt[:,:,i] =  tools.psi2rho(tools.get_vrt(u[:,:,i],v[:,:,i],simul.pm,simul.pn) / tools.rho2psi(simul.f))
        #vrt[:,:,i] =  (tools.get_vrt(u[:,:,i],v[:,:,i],simul.pm,simul.pn) / (simul.f))
    
    u = tools.u2rho(u)
    v = tools.v2rho(v)
    
    topo = simul.topo
    ##############################################################
    # Define horizontal coordinates (deg, km, or grid points)
    ########################################################

    #coord = 'points'
    coord = 'km'

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
        
    lon = lon - lon[int(lon.shape[0]/2),0]
    lat = lat - lat[0,int(lat.shape[1]/2)]
    
    return (vrt,temp,ssh,u,v,lon,lat,topo)
