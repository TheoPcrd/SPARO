#Load packages
from netCDF4 import Dataset
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import netCDF4 as nc4
sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
import torch
from torch.utils.data import DataLoader, Dataset

########## VARIABLES  #############
from variables_create_data import *
#FUNCTION FOR GAUSSIAN FILTER
def time_step_update(imag,j,k, dt = .1):
    lplcn = imag[j,k-1] + imag[j-1,k] + imag[j,k+1] + imag[j+1,k] - 4*imag[j,k]
    return imag[j,k] + dt*lplcn

#READ PDF

nc = nc4.Dataset(file_pdf,'r')
pdf = np.asfortranarray(nc.variables['pdf'])[:,:,:,:,:]
#pdf = np.asfortranarray(nc.variables['pdf'])
nc.close()

nb_time_step = pdf.shape[0]
#pdf_filtered = np.zeros(pdf.shape)

# SAVE PDF FILTERED IN NC FILE

#creating the file
nc = nc4.Dataset(file_pdf_filter,'w')

nc.createDimension('pdfsize', dx_pdf)
nc.createDimension('zdim', 8)
nc.createDimension('nb_time_step', pdf.shape[0])
nc.createDimension('position', 36)
nc.createVariable('pdf_filter', 'f4', ('nb_time_step','zdim','position','pdfsize', 'pdfsize'))
nc.close()

# LOOP FOR GAUSSIAN FILTRATION

for t in range(pdf.shape[0]):
    for z in range(pdf.shape[1]):
        for pos in range(pdf.shape[2]):
            imag = pdf[t,z,pos,:,:]
            cur_imag = imag.copy()
            for L in range(0,50):
                next_img = np.zeros(imag.shape)    
                for k in range(1, imag.shape[0]-1):
                    for j in range(1, imag.shape[1]-1):
                        next_img[j,k] = time_step_update(cur_imag,j,k)
                cur_imag = next_img
            #pdf_filtered[t,z,pos,:,:] = cur_imag
            nc = nc4.Dataset(file_pdf_filter,'r+')
            nc.variables['pdf_filter'][t,z,pos,:,:] = cur_imag
            nc.close()

    if t%10 ==0 : 
        print(t/nb_time_step * 100)

