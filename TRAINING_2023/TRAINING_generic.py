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

### LOAD DATA ####

(image_train,pdf_train,pdf_filter_train) = load_data_train('all')
(image_eval,pdf_eval,pdf_filter_eval) = load_data_eval()

#(image_train,pdf_train,pdf_filter_train) = load_data_train_surface('all')
#(image_eval,pdf_eval,pdf_filter_eval) = load_data_eval_surface()

train_set = Pdf_Image_DataSet(image_train,pdf_train,pdf_filter_train,transform= ToTensor())
eval_set = Pdf_Image_DataSet(image_eval,pdf_eval,pdf_filter_eval,transform= ToTensor())

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 0, shuffle = True, drop_last=False)
eval_loader = DataLoader(eval_set, batch_size=batch_size, num_workers = 0, shuffle = False, drop_last=False)

# Save direction and type of cnn

kernel_size_list=[5]
padding_list=[2]
bias=False
p_dropout_list=[0]
nlayer0_list=[64]
nb_inputs = 76 #28

for i in range(len(kernel_size_list)):
    kernel_size=kernel_size_list[i]
    padding=padding_list[i]
    p_dropout=p_dropout_list[i]
    nlayer0=nlayer0_list[i]
    
    autoencoder = CNN_UNET_generic(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs)
    filename_chkpt = 'CNN_UNET_k{0}_p{1}_b{2}_d{3}_nl{4}_ni_{5}'.format(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs)

    dirSAVE = './Saved_model'
    checkpoint_callback = ModelCheckpoint(monitor='loss_filter_200m_validation',
                                                  dirpath= dirSAVE,
                                                  filename= filename_chkpt + '-{epoch:02d}-{loss_filter_200m_validation:.2f}',
                                                  save_top_k=2,
                                                  mode='min')
    dirLOG = './logs/{0}/'.format(filename_chkpt)



    # TRAINNING 

    trainer = pl.Trainer(max_epochs=max_epochs,gpus=nb_gpus, default_root_dir=dirLOG, callbacks=[checkpoint_callback])
    trainer.fit(model=autoencoder, train_dataloaders=train_loader,val_dataloaders=eval_loader)
