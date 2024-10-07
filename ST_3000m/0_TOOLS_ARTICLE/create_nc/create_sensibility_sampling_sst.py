# # USE GPU

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
import torch
from torch.utils.data import DataLoader, Dataset
from CNN_tools import *
from CNN_UNET import *
from pytorch_lightning.callbacks import ModelCheckpoint
from DATALOADER import Pdf_Image_DataSet
from variables_training import *
import torch
from modules import image_concat,get_images_exp,downscalling_and_interpolation,lon,lat

nb_images = 200
#nb_images = 10
nb_dt = nb_images - nb_sample + 1  # (120-60) = 60/20 = 3 + 1

def create_data_test(sampler,type_transfo=None,):

    raw_images_simu = "/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_100_stdepth_3000/raw_images_001900_005860_wsed100_stdepth3000_large_domain_testing.nc"
    nc_file = nc.Dataset(raw_images_simu, 'r')
    
    ssh = nc_file.variables["ssh"][:nb_images,:,:]
    sst = nc_file.variables["temperature"][:nb_images,0,:,:]
    ssh = ssh[:, np.newaxis, :, :]
    sst = sst[:, np.newaxis, :, :]

    
    if type_transfo=="sampling":
        
        sst_down = sst[:,0,::sampler,::sampler]
        ssh_down = ssh[:,0,::sampler,::sampler]
        lon_down = lon[::sampler,::sampler]
        lat_down = lat[::sampler,::sampler]
        sst = downscalling_and_interpolation(sst_down,sst,lat_down,lon_down)
        ssh = downscalling_and_interpolation(ssh_down,ssh,lat_down,lon_down)
        sst = sst[:,:,40:-40,40:-40]
        ssh = ssh[:,:,40:-40,40:-40]

    
    elif type_transfo=="sampling_ssh40_sst":
        
        ssh_down = ssh[:,0,::40,::40]
        lon_down = lon[::40,::40]
        lat_down = lat[::40,::40]
        ssh = downscalling_and_interpolation(ssh_down,ssh,lat_down,lon_down)
        
        sst_down = sst[:,0,::sampler,::sampler]
        lon_down = lon[::sampler,::sampler]
        lat_down = lat[::sampler,::sampler]
        sst = downscalling_and_interpolation(sst_down,sst,lat_down,lon_down)
        sst = sst[:,:,40:-40,40:-40]
        ssh = ssh[:,:,40:-40,40:-40]
        
    elif type_transfo=="sampling_ssh_sst10":
        
        sst_down = sst[:,0,::10,::10]
        lon_down = lon[::10,::10]
        lat_down = lat[::10,::10]
        sst = downscalling_and_interpolation(sst_down,sst,lat_down,lon_down)
        
        ssh_down = ssh[:,0,::sampler,::sampler]
        lon_down = lon[::sampler,::sampler]
        lat_down = lat[::sampler,::sampler]
        ssh = downscalling_and_interpolation(ssh_down,ssh,lat_down,lon_down)
        sst = sst[:,:,40:-40,40:-40]
        ssh = ssh[:,:,40:-40,40:-40]
        
    elif type_transfo=="pooling_ssh_sst10":
        
        
        sst_down = spatial_downscalling(sst,10)[:,0,:,:]
        lon_down = spatial_downscalling(lon[np.newaxis,:,:],10)[0,:]
        lat_down = spatial_downscalling(lat[np.newaxis,:,:],10)[0,:]
        print(lat_down.shape)
        print(sst_down.shape)
        sst = downscalling_and_interpolation(sst_down,sst,lat_down,lon_down)
        
        ssh_down = spatial_downscalling(ssh,sampler)[:,0,:,:]
        lon_down = spatial_downscalling(lon[np.newaxis,:,:],sampler)[0,:]
        lat_down = spatial_downscalling(lat[np.newaxis,:,:],sampler)[0,:]
        ssh = downscalling_and_interpolation(ssh_down,ssh,lat_down,lon_down)
        sst = sst[:,:,40:-40,40:-40]
        ssh = ssh[:,:,40:-40,40:-40]
        
        
    images = image_concat(sst,ssh)
    nb_dt = images.shape[0] - nb_sample + 1
    nb_cases = nb_dt*36 
    inputs = []
        
    t=0
    for i in range(nb_dt):
        for pos in range(0,36):
            images_i = get_images_exp(i,pos,images)
            inputs.append(images_i)
            t=t+1
            
    inputs = np.array(inputs)
    
    return inputs

def score_and_supermodel(sampler,type_transfo,model=None):
    
    image_test = create_data_test(type_transfo=type_transfo,sampler=sampler)
    print(image_test.shape)

    print('Import data and normalization ...')
    #image_train = load_images_train()
    # Normalization of inputs
    image_test = torch.tensor(image_test,dtype=torch.float)

    image_test = Normalize(torch.tensor(image_test))

    reste = pdf_test.shape[0]%32
    if reste > 0:
        pdf_test_fc = pdf_test[:-reste]
        pdf_filter_test_fc = pdf_filter_test[:-reste]
        image_test = image_test[:-reste]
    else:
        pdf_test_fc = pdf_test[:]
        pdf_filter_test_fc = pdf_filter_test[:]
        image_test = image_test[:]
    #print(reste)

    test_set = Pdf_Image_DataSet(image_test,pdf_test_fc,pdf_filter_test_fc,transform= ToTensor())
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers = 0, shuffle = False, drop_last=False)

    # In[ ]: CREATION OF THE SUPERMODEL 

    device='cuda'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    super_model = np.zeros((pdf_test_fc.shape[0],nb_dx,nb_dx))
    supermodel_surface = np.zeros((pdf_test_fc.shape[0],nb_dx,nb_dx))
    prediction_10 = np.zeros((10,batch_size,nb_dx,nb_dx))
    pdf_10 = np.zeros((10,pdf_test_fc.shape[0],nb_dx,nb_dx))
    std_surface = np.zeros(pdf_test_fc.shape[0])

    i = 0

    kernel_size=5
    padding=2
    bias=False
    p_dropout=0
    nlayer0=64
    nb_inputs = image_test.shape[1] + 8 #76 ou 28

    if model=="sst2_ssh2":
        dirSAVE = '/home2/datahome/tpicard/PhD_MOMOPAR/CHAP2_APPLICATION_SAT/CNN_SSH_SST/TRAIN_AND_VAL_SSH_SST/saved_model/supermodel_wsed_{0}_sdepth{1}/'.format(wsed,depth_trap)
        name_model = 'CNN_SST_SSH_k{0}_p{1}_b{2}_d{3}_nl{4}_ni_{5}'.format(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs)
        
    elif model=="sst24_ssh80":
        dirSAVE = '/home2/datahome/tpicard/PhD_MOMOPAR/CHAP2_APPLICATION_SAT/CNN_SSH_SST/TRAIN_AND_VAL_SSH_SST/saved_model/ssh_80km_sst_28km_wsed/supermodel_wsed_{0}_sdepth{1}/'.format(wsed,depth_trap)
        name_model = 'CNN_SST_SSH_k{0}_p{1}_b{2}_d{3}_nl{4}_ni_{5}'.format(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs)


    with torch.no_grad():
        for batch, (X, y_filter, y) in enumerate(test_loader):

            X, y_filter, y = X.to(device), y_filter.to(device), y.to(device)

            for j in range(0,10):

                autoencoder = CNN_UNET_generic.load_from_checkpoint(dirSAVE+name_model+'_{0}.ckpt'.format(j),kernel_size=kernel_size,padding=padding,bias=bias,p_dropout=p_dropout,nlayer0=nlayer0,nb_inputs=nb_inputs).to(device)
                #filename_chkpt = 'CNN_UNET_k{0}_p{1}_b{2}_d{3}_nl{4}_ni_{5}_{6}'.format(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs,j)
                prediction_test = autoencoder(X,y_filter,y)[:,-1,:,:]
                prediction_10[j,:,:,:] = prediction_test.cpu().detach().numpy()
                pdf_10[:,i*32:i*32+32,:,:] = prediction_10

            prediction_10 = np.sort(prediction_10,axis=0)
            super_model[batch_size*i:batch_size*i+batch_size,:,:] = np.median(prediction_10,axis=0)
            std_surface[batch_size*i:batch_size*i+batch_size] = np.nanmean(np.nanstd(prediction_10,axis=0),axis=(1,2))
            i = i+1

    for i in range(pdf_test_fc.shape[0]):

        supermodel_surface[i,:,:] = super_model[i,:,:]/np.sum(super_model[i,:,:])

    bhatta_surface = np.zeros(supermodel_surface.shape[0])

    for i in range(supermodel_surface.shape[0]):
        bhatta_surface[i] = 1 - np.sum(np.sqrt(np.abs((np.multiply(supermodel_surface[i,:,:], pdf_filter_test_fc[i,-1,:,:])))))


    list_10_models_bhatta = np.zeros((pdf_filter_test_fc.shape[0],10))
    for i in range(pdf_filter_test_fc.shape[0]):
        for j in range(10):
            list_10_models_bhatta[i,j] = 1 - np.sum(np.sqrt(np.abs((np.multiply(pdf_10[j,i,:,:], pdf_filter_test_fc[i,-1,:,:])))))
            
    std = np.std(list_10_models_bhatta,axis=1)
    
    return(supermodel_surface,bhatta_surface,std)

#NORM
folder_images = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_100_stdepth_3000/ssh_80km_sst_28km/'.format(wsed,depth_trap)
name_input_train = 'inputs_000710_006470_wsed100_stdepth3000_zdim10_sst_ssh_training.nc'
file_input = folder_images+name_input_train
image_train = load_images_train()
list_mean_train = np.mean(image_train,axis=(0,2,3))
list_std_train = np.std(image_train,axis=(0,2,3))
Normalize = transforms.Normalize(list_mean_train, list_std_train)
del image_train

#PDF
pdf_test = load_pdf_test()
pdf_filter_test = load_pdf_filter_test()
pdf_test = pdf_test[:nb_dt,:,:,:]
pdf_filter_test = pdf_filter_test[:nb_dt,:,:,:]

pdf_test = np.transpose(pdf_test,(0,2,1,3,4))
pdf_test = pdf_test.reshape(pdf_test.shape[0]*pdf_test.shape[1],8,dx_pdf,dx_pdf)

pdf_filter_test = np.transpose(pdf_filter_test,(0,2,1,3,4))
pdf_filter_test = pdf_filter_test.reshape(pdf_filter_test.shape[0]*pdf_filter_test.shape[1],8,dx_pdf,dx_pdf)

#list_sampler = [5]
list_sampler = [5,10,20,30,40,50,60,70,80,90,100]
#list_sampler = [50,100]

bl_list = []
std_list = []

bl_list_80 = []
std_list_80 = []

for sampler in list_sampler:
    (supermodel_surface,bhatta_surface,std) = score_and_supermodel(sampler,type_transfo='sampling_ssh40_sst',model="sst2_ssh2")
    bl_list.append(bhatta_surface)
    std_list.append(std)
    
    (supermodel_surface,bhatta_surface,std) = score_and_supermodel(sampler,type_transfo='sampling_ssh40_sst',model="sst24_ssh80")
    bl_list_80.append(bhatta_surface)
    std_list_80.append(std)


nc_file = nc.Dataset("score_sensibility_sst_res.nc",'w')

#Dimensions used
nc_file.createDimension('reso', len(list_sampler))
nc_file.createDimension('time', np.array(bl_list_80).shape[1])

nc_file.createVariable('bl_list', 'f4', ('reso','time'))
nc_file.createVariable('bl_list_80', 'f4', ('reso','time'))
nc_file.createVariable('list_sampler', 'f4', ('reso'))

nc_file.variables['bl_list'][:] = bl_list
nc_file.variables['bl_list_80'][:] = bl_list_80
nc_file.variables['list_sampler'][:] = list_sampler
nc_file.close()
