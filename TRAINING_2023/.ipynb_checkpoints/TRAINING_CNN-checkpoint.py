#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
import torch
from torch.utils.data import DataLoader, Dataset
from CNN_tools import *
from CNN_UNET import *
from pytorch_lightning.callbacks import ModelCheckpoint
from DATALOADER import Pdf_Image_DataSet, Pdf_Image_DataSet_image_process
from variables import *
import importlib


# #  CNN TYPE ?

# In[2]:


# Save direction and type of cnn
cnn_type = '4L'
if cnn_type == '4L':
    autoencoder = CNN_UNET_4L()
    filename_chkpt = 'CNN_UNET_4L_all_data'
elif cnn_type =='surf':
    
    autoencoder = CNN_UNET_SURF()
    filename_chkpt = 'CNN_UNET_SURF'
    
dirSAVE = './Saved_model'
checkpoint_callback = ModelCheckpoint(monitor='loss_filter_200m_validation',
                                              dirpath= dirSAVE,
                                              filename= filename_chkpt + '-{epoch:02d}-{loss_filter_200m_validation:.2f}',
                                              save_top_k=2,
                                              mode='min')
dirLOG = './logs/{0}/'.format(filename_chkpt)


# # IMPORT DATA

# In[3]:


### LOAD DATA ####

image_train = load_images_train('all')
#image_norm_train = load_images_train()
pdf_train = load_pdf_train()
pdf_filter_train = load_pdf_filter_train()

#image_norm_eval = load_image_processed('validation')
image_eval = load_images_validation()
pdf_eval = load_pdf_validation()
pdf_filter_eval = load_pdf_filter_validation()


# In[4]:


pdf_train = np.transpose(pdf_train,(0,2,1,3,4))
pdf_train = pdf_train.reshape(pdf_train.shape[0]*pdf_train.shape[1],8,100,100)

pdf_filter_train = np.transpose(pdf_filter_train,(0,2,1,3,4))
pdf_filter_train = pdf_filter_train.reshape(pdf_filter_train.shape[0]*pdf_filter_train.shape[1],8,100,100)

pdf_eval = np.transpose(pdf_eval,(0,2,1,3,4))
pdf_eval = pdf_eval.reshape(pdf_eval.shape[0]*pdf_eval.shape[1],8,100,100)

pdf_filter_eval = np.transpose(pdf_filter_eval,(0,2,1,3,4))
pdf_filter_eval = pdf_filter_eval.reshape(pdf_filter_eval.shape[0]*pdf_filter_eval.shape[1],8,100,100)



# MEAN and STD for each channel
list_mean_train = []
list_std_train = []


for i in range(image_train.shape[1]):
    list_mean_train.append(np.mean(image_train[:,i,:,:]))
    list_std_train.append(np.std(image_train[:,i,:,:]))


# Normalization of inputs
Normalize = transforms.Normalize(list_mean_train, list_std_train)
image_eval = Normalize(torch.tensor(image_eval))
image_train = Normalize(torch.tensor(image_train))


# # CREATE TRAINING AND VALIDATION DATA

## reduce size dataset
                       
train_set = Pdf_Image_DataSet(image_train,pdf_train,pdf_filter_train,transform= ToTensor())
eval_set = Pdf_Image_DataSet(image_eval,pdf_eval,pdf_filter_eval,transform= ToTensor())

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = num_workers, shuffle = True, drop_last=False)

eval_loader = DataLoader(eval_set, batch_size=batch_size, num_workers = num_workers, shuffle = True, drop_last=False)


# # TRAIN MODEL

# In[ ]:


# TRAINNING 
autoencoder = CNN_UNET_4L()
trainer = pl.Trainer(max_epochs=max_epochs,gpus=nb_gpus, default_root_dir=dirLOG, callbacks=[checkpoint_callback])
trainer.fit(model=autoencoder, train_dataloaders=train_loader,val_dataloaders=eval_loader)


