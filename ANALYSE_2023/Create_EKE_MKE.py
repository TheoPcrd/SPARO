# from Tool_EKE import *

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

sys.path.append("/home2/datahome/tpicard/PhD_MOMOPAR/TRAIN_AND_VALIDATION_CNN/OLD/")
import torch
sys.path.append("../")
sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
from Modules import *
from Modules_gula import *
[ic,jc] = np.load('/home2/datahome/tpicard/Pyticles/Inputs/ic_jc.npy')


dt_windows = 30 #30 = 15j avant et 15j après
dt_sample = 20 # 10 = 5j Tous les dt_sample/2 jours on calcul KE
t_start = 1900+dt_windows #
t_end = 5900-dt_windows #
half_box = 200 #*4 for km box
nb_sample = int((dt_windows*2)/dt_sample) +1

depth = list(range(0,-1050,-50))
folder = '/home/datawork-lemar-apero/tpicard/EKE/'
name_nc = 'MKE_EKE_box{0}km_windows{1}j_sample{2}.nc'.format(half_box*4,dt_windows,dt_sample)
file = folder+name_nc    

def comput_EKE_ini(t,dt_windows,dt_sample,half_box,nb_sample):
    
    #DOWSCALLING
    depth = list(range(0,-1050,-50))
    
    #dt = 60 #fenetre integration 
    my_simul = 'aperitif_simu2'
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-half_box,jc+half_box,ic-half_box,ic+half_box)
    parameters = my_simul +str_para+ format(t)
    simul = load(simul = parameters, floattype=np.float64, output=False)
    u_0 = tools.u2rho(var('u',simul,depths=depth).data)
    print(u_0.shape)
    print(nb_sample)
    u = np.zeros((nb_sample,u_0.shape[0],u_0.shape[1],u_0.shape[2]))
    v = np.zeros((nb_sample,u_0.shape[0],u_0.shape[1],u_0.shape[2]))
    


    #dt = 60 #fenetre integration 
    my_simul = 'aperitif_simu2'
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-half_box,jc+half_box,ic-half_box,ic+half_box)
    
    u_avg = np.zeros((half_box*2,half_box*2,len(depth)))
    v_avg = np.zeros((half_box*2,half_box*2,len(depth)))

    n = 0
    #with alive_bar(nb_dt) as bar:
    for i in range(t-dt_windows,t+dt_windows+dt_sample,dt_sample):

        #clear_output(wait=True)
        parameters = my_simul +str_para+ format(i)
        simul = load(simul = parameters, floattype=np.float64, output=False)
        u_n = var('u',simul,depths=depth).data
        v_n = var('v',simul,depths=depth).data
        u[n,:] = tools.u2rho(u_n)
        v[n,:] = tools.v2rho(v_n)
        n=n+1
    
    print(n)
    
    u_avg = np.nanmean(u,axis=0)
    v_avg = np.nanmean(v,axis=0)
    
    u_prim = u - u_avg
    v_prim = v - v_avg
    
    print(u_prim.shape)
    
    #EKE MEAN
    EKE = (1/2*(np.multiply(u_prim[0,:],u_prim[0,:])+np.multiply(v_prim[0,:],v_prim[0,:])))
    
    for i in range(1,nb_sample):
        EKE = EKE + (1/2*(np.multiply(u_prim[i,:],u_prim[i,:])+np.multiply(v_prim[i,:],v_prim[i,:])))
    
    EKE = EKE / nb_sample
    
    
    MKE = (1/2*(np.multiply(u_avg,u_avg)+np.multiply(v_avg,v_avg)))
    
    return MKE,EKE

def Create_nc_file_EKE():

    #creating the file
    nc = nc4.Dataset(file,'w')
    #Dimensions used
    nc.createDimension('xdim', half_box*2)
    nc.createDimension('ydim', half_box*2)
    nc.createDimension('depth', len(depth))
    nc.createDimension('time', 0)

    #Variables used
    nc.createVariable('EKE', 'f4', ('time','depth','xdim', 'ydim'))
    nc.createVariable('MKE', 'f4', ('time','depth','xdim', 'ydim'))
    nc.createVariable('time', 'f4', ('time'))
    
    print('EKE file create at '+file)
    nc.close()

def Update_nc_file_EKE(t,EKE,MKE):
    
    i = int((t-t_start)/dt_sample)  
    nc = nc4.Dataset(file,'r+')
    nc.variables['EKE'][i,:,:,:] = EKE
    nc.variables['MKE'][i,:,:,:] = MKE
    nc.variables['time'][i] = t
    nc.close()
    
    #print('EKE file updated for t= '+str(t))

# CREATION OF EKE

Create_nc_file_EKE()
(MKE,EKE) = comput_EKE_ini(t_start,dt_windows,dt_sample,half_box,nb_sample)
EKE = np.swapaxes(EKE, 0, 2)
EKE = np.swapaxes(EKE, 1, 2)
MKE = np.swapaxes(MKE, 0, 2)
MKE = np.swapaxes(MKE, 1, 2)
Update_nc_file_EKE(t_start,EKE,MKE)

dt_10j = 20
for t in range(t_start+dt_sample,t_end+dt_sample,dt_10j):
    (MKE,EKE) = comput_EKE_ini(t,dt_windows,dt_sample,half_box,nb_sample)
    EKE = np.swapaxes(EKE, 0, 2)
    EKE = np.swapaxes(EKE, 1, 2)
    MKE = np.swapaxes(MKE, 0, 2)
    MKE = np.swapaxes(MKE, 1, 2)
    Update_nc_file_EKE(t,EKE,MKE)

