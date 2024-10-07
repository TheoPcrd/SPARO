#Load packages
from netCDF4 import Dataset
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import netCDF4 as nc
from torch import nn
from torch import optim
import progressbar
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
from torch import optim
import progressbar
#sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
sys.path.append("/home2/datahome/tpicard/PhD_MOMOPAR/CHAP2_APPLICATION_SAT/CNN_SSH_SST/")
sys.path.append("/home2/datahome/tpicard/PhD_MOMOPAR/CHAP2_APPLICATION_SAT/CNN_SSH_SST/TRAIN_AND_VAL_SSH_SST/")
sys.path.append("/home2/datahome/tpicard/PhD_MOMOPAR/CHAP2_APPLICATION_SAT/comparison_supermodel/Score_dx/test_data_sensibility/")

import torch
from torch.utils.data import DataLoader, Dataset
from CNN_tools import *
from CNN_UNET import *
from pytorch_lightning.callbacks import ModelCheckpoint
from DATALOADER import Pdf_Image_DataSet
from variables_training import *
import torch
from modules import image_concat,get_images_exp,downscalling_and_interpolation,lon,lat


# MASSE CENTER MOVEMENT
def masse_center_mov(pdf):
    nb_dx= 100
    lon = np.linspace(-400,400,nb_dx)
    lat = np.linspace(-400,400,nb_dx)
    mx = np.sum(np.multiply(lon,np.sum(pdf,axis=0)))
    my = np.sum(np.multiply(lat,np.sum(pdf,axis=1)))
    return np.sqrt(mx**2 + my**2)

def masse_center_mov_mxy(pdf):
    nb_dx= 100
    lon = np.linspace(-400,400,nb_dx)
    lat = np.linspace(-400,400,nb_dx)
    mx = np.sum(np.multiply(lon,np.sum(pdf,axis=0)))
    my = np.sum(np.multiply(lat,np.sum(pdf,axis=1)))
    return mx,my

# VARIANCE
def variance(pdf):
    nb_dx= 100
    lon = np.linspace(-400,400,nb_dx)
    lat = np.linspace(-400,400,nb_dx)
    D = 0
    mx,my = masse_center_mov_mxy(pdf)
    for i in range(100):
        for j in range(100):
            D = D + (500/499)*pdf[i,j]*((lon[i]-mx)**2+(lat[j]-my)**2)
    return D

# VARIANCE
def entropie(pdf):
    D = 0
    for i in range(100):
        for j in range(100):
            if pdf[i,j]==0:
                 continue
            else:
                D = D - pdf[i,j]*np.log(pdf[i,j])
    return D

def get_pdf(w,depth_trap):
    #w = 200
    if w == 80:
        full_time_exp=140
    elif w == 100:
        full_time_exp=120
    elif w == 150:
        full_time_exp=120
    elif w == 200:
        full_time_exp=100
    elif w == 300:
        full_time_exp=100
    else:
        full_time_exp=100
        
    #print(full_time_exp)
    
    folder_pdf = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_{0}_stdepth_{1}/'.format(w,depth_trap)
    tpas_start = 0 # First experience ?
    tpas_end = 65 #65 Number of experiences
    date_start = 1900
    date_end = date_start + full_time_exp + 60*(tpas_end-1) # date end
    name_pdf ='pdf_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx100_testing.nc'.format(date_start,date_end,w,depth_trap)
    name_pdf_filter = 'filter_'+name_pdf
    
    file = folder_pdf+name_pdf_filter
    #Load data
    nc_data = nc4.Dataset(file, 'r')
    pdf_filter = np.asfortranarray(nc_data.variables['pdf_filter'][:])
    nc_data.close()
    pdf_filter = np.transpose(pdf_filter,(0,2,1,3,4))
    pdf_filter = pdf_filter.reshape(pdf_filter.shape[0]*pdf_filter.shape[1],8,100,100)

    return pdf_filter[:,-1,:,:]


def get_supermodel_ssh_sst_sat(w):

    #dir_save_supermodel = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2004_2016/prediction/'
    dir_save_supermodel= '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/prediction/'
    name_nc = 'supermodel_ssh80_sst24_w{0}.nc'.format(w) #name_supermodel
    file = dir_save_supermodel+name_nc
    
    #LOAD super model
    nc = nc4.Dataset(file,'r')
    supermodel_sst_ssh = np.asfortranarray(nc.variables['supermodel'])
    nc.close()
    
    return (supermodel_sst_ssh)

def get_supermodel_ssh_sst_simu(w):

    dir_save_supermodel = '/home/datawork-lemar-apero/tpicard/DATA_CNN/supermodel/sst_ssh/wsed_sst24_ssh80/'
    name_nc = 'supermodel_wsed_{0}_nmode_10.nc'.format(w) #name_supermodel
    file = dir_save_supermodel+name_nc
    
    #LOAD super model
    nc = nc4.Dataset(file,'r')
    supermodel_sst_ssh = np.asfortranarray(nc.variables['supermodel_surface'])
    nc.close()
    
    return (supermodel_sst_ssh)

#Add sat stats
def get_mass_center(pred):

    list_mc = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        list_mc[i] = masse_center_mov(pred[i,:,:])
        
    return(list_mc)

#Add sat stats
def get_entropy(pred):

    list_entropy = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        list_entropy[i] = entropie(pred[i,:,:])
        
    return list_entropy


w_list = [80,100,150,200,300]

mc_base = []
mc_simu = []
mc_sat = []

for w in w_list:
    
    list_mass_center=[]
    pred_base=get_pdf(w,3000)
    mc_base.append(get_mass_center(pred_base))
    pred_sat = get_supermodel_ssh_sst_sat(w)
    pred_simu = get_supermodel_ssh_sst_simu(w)
    mc_simu.append(get_mass_center(pred_simu))
    mc_sat.append(get_mass_center(pred_sat))

mc_base = np.array(mc_base)
mc_simu = np.array(mc_simu)
mc_sat = np.array(mc_sat)

ent_base = []
ent_simu = []
ent_sat = []

for w in w_list:
    
    list_mass_center=[]
    pred_base=get_pdf(w,3000)
    ent_base.append(get_entropy(pred_base))
    pred_sat = get_supermodel_ssh_sst_sat(w)
    pred_simu = get_supermodel_ssh_sst_simu(w)
    ent_simu.append(get_entropy(pred_simu))
    ent_sat.append(get_entropy(pred_sat))


ent_base = np.array(ent_base)
ent_simu = np.array(ent_simu)
ent_sat = np.array(ent_sat)


nc_file = nc.Dataset("/home/datawork-lemar-apero/tpicard/STAT_PDF/mc_entropy_200m_2000_2020.nc",'w')

#Dimensions used
nc_file.createDimension('w', len(w_list))
nc_file.createDimension('tsimu', np.array(mc_simu).shape[1])
nc_file.createDimension('tsat', np.array(mc_sat).shape[1])
nc_file.createDimension('tbase', np.array(mc_base).shape[1])


nc_file.createVariable('mc_base', 'f4', ('w','tbase'))
nc_file.createVariable('mc_simu', 'f4', ('w','tsimu'))
nc_file.createVariable('mc_sat', 'f4', ('w','tsat'))

nc_file.createVariable('ent_base', 'f4', ('w','tbase'))
nc_file.createVariable('ent_simu', 'f4', ('w','tsimu'))
nc_file.createVariable('ent_sat', 'f4', ('w','tsat'))



nc_file.variables['mc_base'][:] = mc_base
nc_file.variables['mc_simu'][:] = mc_simu
nc_file.variables['mc_sat'][:] = mc_sat

nc_file.variables['ent_base'][:] = ent_base
nc_file.variables['ent_simu'][:] = ent_simu
nc_file.variables['ent_sat'][:] = ent_sat

nc_file.close()
