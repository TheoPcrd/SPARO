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
sys.path.append("/home2/datahome/tpicard/PhD_MOMOPAR/TRAIN_AND_VALIDATION_CNN/OLD/")
import torch
from torch.utils.data import DataLoader, Dataset
from CNN_tools import *
from CNN_UNET import *
from pytorch_lightning.callbacks import ModelCheckpoint
from DATALOADER import Pdf_Image_DataSet
from variables import *
import importlib

def add_contour_per(H_sum_tot,proba_list):
    #proba_list = [99.99]
    n = 0

    level = np.zeros(len(proba_list))
    p = np.zeros(len(proba_list))

    for proba in proba_list:

        i = dichotomie(f,-6,1,0.01,proba,H_sum_tot)
        #print(i)

        H_tot = np.sum(H_sum_tot)

        H_filter = np.where(H_sum_tot > 10**i, H_sum_tot,0)
        p[n] = np.sum(H_filter)*100/H_tot

        level[n] = 10**i
        n=n+1

    fmt = {}
    for l, s in zip(level, p):
        #fmt[l] = str(round(s,1))+'%'
        fmt[l] = ''

    #CS = plt.contour(xedges,yedges,H_sum_tot,level,linewidths=2,alpha=1,colors=c,linestyles='-')
    #plt.clabel(CS,level, inline=1, fontsize=18,fmt = fmt)
    
    return level,fmt

def f(x,p,H_sum_tot):
    
    H_filter = np.where(H_sum_tot > 10**x, H_sum_tot,0)
    H_tot = np.sum(H_sum_tot)
    return(np.sum(H_filter)*100/H_tot-p) 
    
def dichotomie(f,a,b,e,p,H_sum_tot):
    delta = 1

    while delta > e:
        m = a + (b - a) / 2
        delta = abs(b - a)
        #print("{:15} , {:15} , {:15} , {:15} , {:15} , {:15} , {:15} ".format(a,b,m,f(a),f(b),f(m),delta) )
        if f(m,p,H_sum_tot) == 0:
            return m
        elif f(a,p,H_sum_tot) * f(m,p,H_sum_tot)  > 0:
            a = m
        else:
            b = m
    return m

# MAIN MODEL
SMOOTH = 1e-8
def iou_numpy(outputs, labels):
    #outputs = outputs.squeeze(1)

    outputs = np.where(outputs>0.0002,1,0)
    labels = np.where(labels>0.0002,1,0)
    intersection = (outputs & labels).sum((0, 1))
    union = (outputs | labels).sum((0, 1))
    
    #iou = 1 - (intersection + SMOOTH) / (union + SMOOTH)
    iou = 1 - ((intersection + SMOOTH) / (outputs.sum((0, 1)) + SMOOTH))

    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return iou  # Or thresholded.mean()

# MAIN MODEL
SMOOTH = 1e-8

def iou_particles(outputs, labels):
    #outputs = outputs.squeeze(1)

    outputs_bin = np.where(outputs>0.0002,1,0)
    labels = np.where(labels>0.0002,1,0)
    intersection = np.multiply(outputs_bin,labels)
    outputs_bin = np.where(intersection==1,outputs,0)
    iou = 1- np.sum(outputs_bin)

    return iou  # Or thresholded.mean()

def distribution_valid(title,list_metric,name_x):

    fig = plt.figure(figsize=(10,6))
    bin = np.linspace(0,1,21)
    hist,edge = np.histogram(list_metric, bins=bin)
    plt.hist(list_metric, bins=bin, density=False,histtype='stepfilled',alpha=0.8)
    plt.grid()
    plt.ylabel('Nb cases',size = 16)
    plt.xlabel(name_x,size = 16)
    coef = list_bhatta.shape[0]/100

    BC01 = np.sum(np.where(list_metric <0.2,1,0))/coef
    BC02 = np.sum(np.where(list_metric >0.3,1,0))/coef
    BC03 = (list_metric.shape[0] - BC01*coef - BC02*coef)/coef

    props = dict(boxstyle='round', facecolor='white', alpha=1,edgecolor='k')
    textstr1 = str(round(BC01))+'%'
    textstr2 = str(round(BC02))+'%'
    textstr3 = str(round(BC03))+'%'
    # place a text box in upper left in axes coords
    textstr = 'BC < 0.2 = '+str(round(BC01))+'%'+'\n'+'0.2 < BC < 0.3 = '+str(round(BC03))+'%'+'\n'+'BC > 0.3 = '+str(round((BC02)))+'%'
    # place a text box in upper left in axes coords
    plt.text(0.05, 400, textstr1, fontsize=14,
            verticalalignment='top', bbox=props,color ='green')
    plt.text(0.22, 400, textstr2, fontsize=14,
            verticalalignment='top', bbox=props,color='orange')
    plt.text(0.5, 400, textstr3, fontsize=14,
            verticalalignment='top', bbox=props,color='red')

    plt.vlines(0.2,0,2000,colors='k',linestyle='--')
    plt.vlines(0.3,0,2000,colors='k',linestyle='--')
    plt.ylim(0,np.max(hist)+20)
    plt.xlim(0,0.8)
    plt.title(title,size=20)

# MASSE CENTER MOVEMENT
def masse_center_mov(pdf):
    mx = np.sum(np.multiply(lon,np.sum(pdf,axis=0)))
    my = np.sum(np.multiply(lat,np.sum(pdf,axis=1)))
    return np.sqrt(mx**2 + my**2)

def masse_center_mov_mxy(pdf):
    mx = np.sum(np.multiply(lon,np.sum(pdf,axis=0)))
    my = np.sum(np.multiply(lat,np.sum(pdf,axis=1)))
    return mx,my

