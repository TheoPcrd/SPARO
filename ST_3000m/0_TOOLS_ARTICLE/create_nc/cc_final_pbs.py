#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:08:22 2023

@author: tpicard
"""

import pandas as pd
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import datetime
import netCDF4 as nc

import datetime, numpy as np
import calendar
from datetime import timedelta
#%%% Correlation fct
from sklearn.linear_model import LinearRegression
#from tools_analyse import add_contour_per
from datetime import datetime
from cycler import cycler
import matplotlib as mpl
import random

gap200 = 38
gap100 = 43

def toTimestamp(d):
  return calendar.timegm(d.timetuple())

plt.rcParams['lines.linewidth'] = 2
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.50
plt.rcParams['xtick.labelsize'] = 16  # Taille des ticks sur l'axe des x
plt.rcParams['ytick.labelsize'] = 16  # Taille des ticks sur l'axe des y
plt.rcParams['axes.titlesize'] = 16  # Taille du titre
plt.rcParams['axes.labelsize'] = 16  # Taille du label de l'axe des x
plt.rcParams['lines.markersize'] = 10
plt.rcParams['font.size'] = '20.0'
plt.rcParams['lines.markeredgewidth'] = 3


#################
### TO EDIT  ###
#################
year_start = 2009
year_end = 2020
filter_2011_2013 = False 
filter_Lampitt = False 


#file = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/cross_corr/cc_{0}_{1}_no_2011_2013_j2.nc'.format(year_start,year_end)
#file = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/cross_corr/cc_{0}_{1}_no_2011_2013_j2.nc'.format(year_start,year_end)
if filter_Lampitt:
    
    if filter_2011_2013:
        file = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/cross_corr/cc_{0}_{1}_no_2011_2013_j2_filter_lampitt_v2.nc'.format(year_start,year_end)

    else:    
        file = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/cross_corr/cc_{0}_{1}_j2_filter_lampitt_v2.nc'.format(year_start,year_end)

else : 
    
    if filter_2011_2013:

        file = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/cross_corr/cc_{0}_{1}_no_2011_2013_j2_v2.nc'.format(year_start,year_end)

    else:    
        file = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/cross_corr/cc_{0}_{1}_j2_v2.nc'.format(year_start,year_end)



dir_save_prediction = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/prediction/'
name_nc = 'supermodel_ssh80_sst24_w{0}.nc'.format(80)
#print(name_nc)
nc_file = nc.Dataset(dir_save_prediction+name_nc,'r')
prediction_pdf = nc_file.variables["supermodel"][:-2]
mid_date_prediction = nc_file.variables["mid_date"][:-2]
std = nc_file.variables["std"][:-2]
nc_file.close()

file_satellite_chloro = "/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/chl_sat_tint.nc"

nc_file = nc.Dataset(file_satellite_chloro,'r')
chloro_images = nc_file.variables["chl"][:]
date_sat = nc_file.variables["time"][:]
nc_file.close()


# ################
# ## PAP ST    ###
# ################

file_st_data = "/home/datawork-lemar-apero/tpicard/DATA_ST/PAP_data_3000m 1989_2019_theo.xlsx"
# Lire le fichier Excel
df = pd.read_excel(file_st_data)
mid_date = df['close date (dd/mm/yyyy)']
dry_weight = df['Dry Weight Flux mg/m2/day']

df_filtre = df.loc[df['close date (dd/mm/yyyy)'].dt.year <= year_end]
df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year >= year_start]

# year_to_remove=[2011,2013]

dj = -3

if filter_2011_2013 ==True :
    df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year != 2011]
    df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year != 2013]

if filter_Lampitt ==True :
    df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year != 2000]
    df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year != 2002]
    df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year != 2003]
    df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year != 2005]
    df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year != 2006]
    df_filtre = df_filtre.loc[df['close date (dd/mm/yyyy)'].dt.year != 2008]

filtre_cluster=True

filtre_month=False
if filtre_month ==True :
    df_filtre = df_filtre.loc[df_filtre['close date (dd/mm/yyyy)'].dt.month > 2]
    df_filtre = df_filtre.loc[df_filtre['close date (dd/mm/yyyy)'].dt.month < 10]

mid_date = df_filtre['mid date (dd/mm/yyyy)'][:-2]
close_date = df_filtre['close date (dd/mm/yyyy)'][:-2]
open_date = df_filtre['open date (dd/mm/yyyy)'][:-2]

dry_weight = df_filtre['Dry Weight Flux mg/m2/day'][:-2]
POC = df_filtre['Particulate Organic Carbon Flux mg/m2/day'][:-2]
TPN = df_filtre['Total Particulate Nitrogen Flux mg/m2/day'][:-2]
PIC = df_filtre['Particulate Inorganic Carbon Flux mg/m2/day'][:-2]
PIC = PIC.to_numpy()
PIC = np.array(PIC,dtype=float)
POC = POC.to_numpy()
POC = np.array(POC,dtype=float)
TPN = TPN.to_numpy()
TPN = np.array(TPN,dtype=float)
dry_weight = dry_weight.to_numpy()
dry_weight = np.array(dry_weight,dtype=float)
date_pap =  np.array([toTimestamp(d) for d in mid_date]) 

"""
# First 
lim_w300 = 10
lim_w200 = 16
lim_w150 = 24
lim_w100 = 32
#lim_w80 = 38 
"""

lim_w300 = 10 + 2
lim_w200 = 15 + 3
lim_w150 = 20 + 5
lim_w100 = 30 + 4
#lim_w80 = 38 

def prediction_w_i(jlag):

    if jlag > -lim_w300:
        w_i = 300 # !!
        std_lim = 1
    elif jlag <= -lim_w300 and jlag > -lim_w200:
        w_i = 200 # !! 
        std_lim = 1
    elif jlag <= -lim_w200 and jlag > -lim_w150:
        w_i = 150 # !!
        std_lim = 1
    elif jlag <= -lim_w150 and jlag > -lim_w100:
        w_i = 100 # !!    
        std_lim = 1
    elif jlag <= -lim_w100:
        w_i = 80
        std_lim = 1


    dir_save_prediction = '/home/datawork-lemar-apero/tpicard/DATA_SAT/2000_2023/prediction/'
    name_nc = 'supermodel_ssh80_sst24_w{0}.nc'.format(w_i)
    #print(name_nc)
    nc_file = nc.Dataset(dir_save_prediction+name_nc,'r')
    prediction_pdf = nc_file.variables["supermodel"][:-2]
    mid_date_prediction = nc_file.variables["mid_date"][:-2]
    std = nc_file.variables["std"][:-2]
    nc_file.close()

    ### Focus 
    focus = False
    if focus == True: 
        level_limit = 99
        for i in range(prediction_pdf.shape[0]):
            prediction_pdf_i = prediction_pdf[i,:,:]
            level,fmt = add_contour_per(prediction_pdf_i,[level_limit])
            prediction_pdf_i = np.where(prediction_pdf_i < level,0,prediction_pdf_i)
            prediction_pdf_i = prediction_pdf_i/np.nansum(prediction_pdf_i)
            prediction_pdf[i,:,:] = prediction_pdf_i


    nb_day_gap = 0 # Correction ?

    mid_date_prediction_datetime = [datetime.strptime(date, '%Y-%m-%d').date() for date in mid_date_prediction]
    date_pred =  np.array([toTimestamp(d)+3600*24*nb_day_gap for d in mid_date_prediction_datetime]) 

    
    mid_date_jlag = np.array([toTimestamp(d) for d in mid_date+timedelta(days=jlag)]) # Date source area 
    close_date_jlag = np.array([toTimestamp(d) for d in close_date+timedelta(days=jlag)]) # Date source area 
    open_date_jlag = np.array([toTimestamp(d) for d in open_date+timedelta(days=jlag)]) # Date source area 
    
    pred_chl = []
    pred_rand = []
    box200 = []
    box100 = []
    nan_coef_list = []

    for i in range(mid_date_jlag.shape[0]):

        #print(open_date_jlag[i])
        #print(date_pred.shape)

        # Mean pred
        index_pred = np.argwhere((date_pred >= open_date_jlag[i]+5*3600*24) & (date_pred <= close_date_jlag[i]-5*3600*24))[:,0]
        
        #print(index_pred)
        
        if index_pred.shape[0]==0:
            if np.argwhere((date_pred >= open_date_jlag[i]+0*3600*24) & (date_pred <= close_date_jlag[i]-0*3600*24)).shape[0]!=0:
                index_pred=np.array([np.argmin(np.abs(date_pred-mid_date_jlag[i]))])
                pred_i = prediction_pdf[index_pred,:]
            else:
                pred_i = np.nan
            
        else :
            pred_i = np.nansum(prediction_pdf[index_pred,:],axis=0)
            pred_i = pred_i/index_pred.shape[0] # nomalisation

        # Mean chl
        index_chl = np.argwhere((date_sat >= open_date_jlag[i]-5*3600*24) & (date_sat <= close_date_jlag[i]+5*3600*24))[:,0]
        if index_chl.shape[0]==0:
            index_chl=np.argwhere(date_sat==mid_date_jlag)
            #print('ok')
            
        
        random_index = random.randint(0,prediction_pdf.shape[0]-1)
        pred_rand_i = prediction_pdf[random_index,:]         

        chl_i = np.nanmean(chloro_images[index_chl,:],axis=0) # Try gaussian ponderation ? 
        nan_coef = np.sum(np.where(np.isnan(chl_i),pred_i,0))
        nan_coef_rand = np.sum(np.where(np.isnan(chl_i),pred_rand_i,0))

        pred_chl_i=np.nansum(np.multiply(chl_i,pred_i))
        pred_chl_rand_i=np.nansum(np.multiply(chl_i,pred_rand_i))

        box100_i=np.nanmean(chl_i[gap100:-gap100,gap100:-gap100])
        box200_i=np.nanmean(chl_i[gap200:-gap200,gap200:-gap200])
        
        
        if nan_coef>0.99:
            pred_chl_i=np.nan
            box100_i=np.nan
            box200_i=np.nan
            pred_chl_rand_i=np.nan
            
        elif nan_coef_rand>0.99:
            pred_chl_rand_i=np.nan
            
        elif np.nanmean(std[index_pred])>1:
            pred_chl_i=np.nan
            box100_i=np.nan
            box200_i=np.nan
            pred_chl_rand_i=np.nan

        elif np.isnan(box100_i):
            pred_chl_i=np.nan
            box200_i=np.nan
            pred_chl_rand_i=np.nan
            #box200_i=np.nan  
            
        else:
            pred_chl_i = pred_chl_i/(1-nan_coef)
            pred_chl_rand_i = pred_chl_rand_i/(1-nan_coef_rand)

        pred_rand.append(pred_chl_rand_i)
        pred_chl.append(pred_chl_i)
        nan_coef_list.append(nan_coef)
        box100.append(box100_i)
        box200.append(box200_i)


        
        #print(nan_coef)

    box100 = np.array(box100)
    box200 = np.array(box200)
    pred_chl = np.array(pred_chl)
    pred_rand=np.array(pred_rand)
    


    return mid_date_jlag,pred_chl,box200,box100,pred_rand


def correlation_j(st_data,sat_data):
    #coefficients = np.polyfit(FlC_list[index_pap][:,0], np.array(chl)[index_pap], deg=1)  # Régression linéaire d'ordre 1
    
    coefficients,residuals,rank,singular_values,rcond = np.polyfit(st_data, sat_data, 1, rcond=None, full=True, w=None, cov=False)
    regress_coef = 1 - residuals/(np.nansum((sat_data - np.nanmean(sat_data))**2))
    
    return regress_coef,coefficients

def plot_correlation_all(jlag_max,jlag_min,type_var):
    
    
    xlim=0.26
    
    if type_var=="dry weight":
        var = dry_weight
    elif type_var=="POC":
        var = POC
    elif type_var=="TPN":
        var = TPN
        #xlim=0.26
    elif type_var=="PIC":
        var = PIC


    
    alpha_vlines=0.4
    plt.vlines(-10,0,xlim,linestyle='--',color='k',alpha=alpha_vlines)
    plt.vlines(-16,0,xlim,linestyle='--',color='k',alpha=alpha_vlines)
    plt.vlines(-24,0,xlim,linestyle='--',color='k',alpha=alpha_vlines)
    plt.vlines(-32,0,xlim,linestyle='--',color='k',alpha=alpha_vlines)

    regress_coef_list =[]
    jlag_list = []
    #%%% jlag plot
    regress_coef_list_box = []
    regress_coef_list_box_100 = []
    regress_coef_list_pred = []
    #regress_coef_list_rand = []

    for jlag in range(jlag_min,jlag_max,dj):
        
        date_jlag,pred_chl,box200,box100,pred_rand = prediction_w_i(jlag)
        
        index_nan = np.isnan(box100)
        index_nan = np.where(np.isnan(pred_chl),True,index_nan)
        index_nan = np.where(np.isnan(var),True,index_nan)
        regress_coef,coefficients = correlation_j(var[~index_nan],box200[~index_nan])
        regress_coef_list_box.append(regress_coef)
        regress_coef,coefficients = correlation_j(var[~index_nan],pred_chl[~index_nan])
        regress_coef_list_pred.append(regress_coef)
        regress_coef,coefficients = correlation_j(var[~index_nan],box100[~index_nan])
        regress_coef_list_box_100.append(regress_coef)
        #regress_coef,coefficients = correlation_j(var[~index_nan],pred_rand[~index_nan])
        #regress_coef_list_rand.append(regress_coef)

        jlag_list.append(jlag)

    """    
    plt.plot(jlag_list,regress_coef_list_box)
    plt.plot(jlag_list,regress_coef_list_box_100)
    plt.plot(jlag_list,regress_coef_list_pred)
    #plt.plot(jlag_list,regress_coef_list_rand,alpha=0.5)
    plt.scatter(jlag_list,regress_coef_list_box,label="$box_{200km}$",marker='o')
    plt.scatter(jlag_list,regress_coef_list_box_100,label="$box_{100km}$",marker='o')
    plt.scatter(jlag_list,regress_coef_list_pred,label="$Unet_{sat}$",marker='o')
    #plt.scatter(jlag_list,regress_coef_list_rand,label="$rand(Unet_{sat})$",marker='o',alpha=0.5)
    
    plt.grid()
    plt.ylim(0,xlim)
    
    plt.xlim(jlag_max+1,jlag_min)
    
    """
    
    return(regress_coef_list_box_100,regress_coef_list_box,regress_coef_list_pred,jlag_list)

def plot_correlation_random(jlag_max,jlag_min,type_var):
    
    nb_experience = 100
    xlim=0.25
    
    if type_var=="dry weight":
        var = dry_weight
    elif type_var=="POC":
        var = POC
    elif type_var=="TPN":
        var = TPN
        #xlim=0.26
    elif type_var=="PIC":
        var = PIC


    
    alpha_vlines=0.4
    plt.vlines(-10,0,xlim,linestyle='--',color='k',alpha=alpha_vlines)
    plt.vlines(-16,0,xlim,linestyle='--',color='k',alpha=alpha_vlines)
    plt.vlines(-24,0,xlim,linestyle='--',color='k',alpha=alpha_vlines)
    plt.vlines(-32,0,xlim,linestyle='--',color='k',alpha=alpha_vlines)

    jlag_list = []
    #%%% jlag plot
    regress_coef_list_pred_10_per = []
    regress_coef_list_pred_90_per = []

    for jlag in range(jlag_min,jlag_max,dj):
        
        date_jlag,pred_chl,box200,box100,pred_rand = prediction_w_i(jlag)
        
        index_nan = np.isnan(box100)
        index_nan = np.where(np.isnan(pred_chl),True,index_nan)
        index_nan = np.where(np.isnan(var),True,index_nan)
        
        list_coef_i = []
        for j in range(nb_experience):
            date_jlag,pred_chl,box200,box100,pred_rand = prediction_w_i(jlag)
            regress_coef,coefficients = correlation_j(var[~index_nan],pred_rand[~index_nan])
            list_coef_i.append(regress_coef)
        #print(list_coef_i)
        
        list_coef_i=np.array(list_coef_i)
        
        regress_coef_list_pred_10_per.append(np.nanpercentile(list_coef_i,10))
        regress_coef_list_pred_90_per.append(np.nanpercentile(list_coef_i,90))
        
        jlag_list.append(jlag)

    return(regress_coef_list_pred_10_per,regress_coef_list_pred_90_per)


jlag_max = -120
jlag_min = 0

(regress_coef_list_box_100,regress_coef_list_box_200,regress_coef_list_pred,jlag_list) = plot_correlation_all(jlag_max,jlag_min,"dry weight")



#creating the file
nc_cc = nc.Dataset(file,'w')
#Dimensions used
nc_cc.createDimension('lags', len(jlag_list))
nc_cc.createDimension('z', 5)

nc_cc.createVariable('dry weight', 'f4', ('lags','z'))
nc_cc.createVariable('POC', 'f4', ('lags','z'))
nc_cc.createVariable('PIC', 'f4', ('lags','z'))
nc_cc.createVariable('TPN', 'f4', ('lags','z'))
nc_cc.createVariable('lag_list', 'f4', ('lags'))
nc_cc.createVariable('z_dim', str, ('z'))
#nc.createVariable('lon', 'f4', ('xdim','ydim'))


nc_cc.variables['z_dim'] = ["pred","box100","box200","pred_10_per","pred_90_per"]
nc_cc.variables['lag_list'][:] = jlag_list

# print('KE/vrt file create at '+file


list_var=["dry weight","POC","PIC","TPN"]
for var_i in list_var:
    
    print(var_i)
    (regress_coef_list_box_100,regress_coef_list_box_200,regress_coef_list_pred,jlag_list) = plot_correlation_all(jlag_max,jlag_min,var_i)
    (regress_coef_list_pred_10_per,regress_coef_list_pred_90_per) = plot_correlation_random(jlag_max,jlag_min,var_i)

    nc_cc = nc.Dataset(file,'r+')
    #Dimensions used
    #nc.createVariable('lon', 'f4', ('xdim','ydim'))
    nc_cc.variables[var_i][:,0] = np.array(regress_coef_list_pred)
    nc_cc.variables[var_i][:,1] = np.array(regress_coef_list_box_100)
    nc_cc.variables[var_i][:,2] = np.array(regress_coef_list_box_200)
    nc_cc.variables[var_i][:,3] = np.array(regress_coef_list_pred_10_per)
    nc_cc.variables[var_i][:,4] = np.array(regress_coef_list_pred_90_per)

    #print('KE/vrt file create at '+file)
    nc_cc.close()

    
