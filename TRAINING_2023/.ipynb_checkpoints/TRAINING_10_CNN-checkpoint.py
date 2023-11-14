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
dirSAVE = './Saved_model/supermodel/'

# In[3]:


### LOAD DATA ####

#4L DATA
(image_train,pdf_train,pdf_filter_train) = load_data_train('all')
(image_eval,pdf_eval,pdf_filter_eval) = load_data_eval()

#SURFACE DATA
#(image_train,pdf_train,pdf_filter_train) = load_data_train_surface('all')
#(image_eval,pdf_eval,pdf_filter_eval) = load_data_eval_surface()

train_set = Pdf_Image_DataSet(image_train,pdf_train,pdf_filter_train,transform= ToTensor())
eval_set = Pdf_Image_DataSet(image_eval,pdf_eval,pdf_filter_eval,transform= ToTensor())

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 0, shuffle = True, drop_last=False)
eval_loader = DataLoader(eval_set, batch_size=batch_size, num_workers = 0, shuffle = False, drop_last=False)

# Save direction and type of cnn

kernel_size=5
padding=2
bias=False
p_dropout=0.2
nlayer0=64
nb_inputs = 28 #76 ou 28

# # CREATE TRAINING AND VALIDATION DATA

## reduce size dataset
                       
train_set = Pdf_Image_DataSet(image_train,pdf_train,pdf_filter_train,transform= ToTensor())
eval_set = Pdf_Image_DataSet(image_eval,pdf_eval,pdf_filter_eval,transform= ToTensor())

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = num_workers, shuffle = True, drop_last=False)

eval_loader = DataLoader(eval_set, batch_size=batch_size, num_workers = num_workers, shuffle = False, drop_last=False)


# TRAINNING 
for i in range(10):

    autoencoder = CNN_UNET_generic(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs)
    filename_chkpt = 'CNN_UNET_k{0}_p{1}_b{2}_d{3}_nl{4}_ni_{5}_{6}'.format(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs,i)
      
    checkpoint_callback = ModelCheckpoint(monitor='loss_filter_200m_validation',
                                              dirpath= dirSAVE,
                                              filename= filename_chkpt,
                                              save_top_k=1,
                                              mode='min')

    dirLOG = './logs/supermodel/{0}'.format(filename_chkpt)


    # TRAINNING 
    trainer = pl.Trainer(max_epochs=max_epochs,gpus=nb_gpus, default_root_dir=dirLOG, callbacks=[checkpoint_callback])
    trainer.fit(model=autoencoder, train_dataloaders=train_loader,val_dataloaders=eval_loader)


