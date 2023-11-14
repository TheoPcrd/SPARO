#!/usr/bin/env python
# coding: utf-8

#From the pyticles outputs (64 nc files corresponding to 101 days of particles released)
#We create 64 nc files that containt all the pdf for corresponding period


########## VARIABLES  #############
from variables_create_data import *

#Load packages
from netCDF4 import Dataset
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import netCDF4 as nc4
sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
from Modules import *
from Modules_gula import *




def spatial_filter(px,py,pxcenter,pycenter,index_start):
    
# Function that take all the particules positions px and py 
# (pxcenter,pycenter) is the center of a sub-patch that containt 36 particles 
# index_start is generally 0, 20 or 40 and correspond to the number of time step after the starting period of simu
# Return a list of index that correspond to the particles located inside the sub-patch

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




def compute_outputs(time_step):
    
    # Take a date among the 64 period
    # Return all the 108 independant pdf during this period
    # 36 : number of sub-pathes 
    # 3 : number of time step considered in a 101 days period
    # nb_d x nb_dx: Spatial resolution of the pdf
    # 9 : number of vertical levels
    
    #time_step: From 0 to 108 
    
    jpdf_list = np.zeros((8,36,3,dx_pdf,dx_pdf))
    
    # List_level can depends on depth_trap :
    if list_level_cst == False:
        list_level = np.linspace(200,3000,9)[:-1]
        list_level= np.flip(list_level)
        list_level = list_level.tolist()
    else :
        # Cst list_level
        list_level = [900,800,700,600,500,400,300,200]
    
    x_disp_level = np.zeros((len(list_level),720))
    y_disp_level = np.zeros((len(list_level),720))
    
    #List of the 36 sub-patches centers 
    ic_all_list = np.linspace(1520.5,1610.5,6)
    jc_all_list = np.linspace(569.5,659.5,6)
    
    ########################
    # Choose the folder and step time
    ########################

    #dt = dt_exp # 30 days time step (time that separate 2 experiences)
    #dt_exp = full_time_exp # 50 days time step (time of 1 experience)
    
    date_start_i = date_start + (dt_exp*time_step) #
    date_end_i = date_start + (dt_exp*time_step) + full_time_exp
    
 
    if test==False:    #TRAINING
        file_in = folder_in_pyticle + 'apero_trap{0}m_wsed{1}_2000dx_100nx_{2:06}_{3:06}.nc'.format(depth_trap,wsed,date_start_i,date_end_i)
    else:        # TESTING
        file_in = folder_in_pyticle + 'aperitif_simu2_trap{0}m_wsed{1}_2000dx_100nx_{2:06}_{3:06}.nc'.format(depth_trap,wsed,date_start_i,date_end_i) 

    #Load data
    nc_data = Dataset(file_in, 'r')
    px = np.asfortranarray(nc_data.variables['px'])
    py = np.asfortranarray(nc_data.variables['py'])
    pdepth = np.asfortranarray(nc_data.variables['pdepth'])
    nc_data.close()
    dx, dy = 1.979, 1.979 #grid size

    #print('File for date start ={0} is building...'.format(time_end-101))
    
    # Create pdf
    l = 0 
    for level in list_level:
        x_disp = []
        y_disp = []
        for dt in range(0,3):
            i=0
            for ic in ic_all_list:
                for jc in jc_all_list:


                    #Load data for background
                    index_start = 20*dt #Choose 0, 20 or 40

                    x_disp = []
                    y_disp = []
                    #temp = []

                    index = spatial_filter(px,py,ic,jc,index_start) #Filtre spatial haut droite

                    npart_trap = index.size
                    count = 0

                    for ipart in index:
                        index_realtime = np.where( pdepth[:,ipart] != 0 )[0] # find the period when particles are released
                        #if ipart%100000 ==0:
                            #print('ipart = ' +str(ipart))

                        if index_realtime.size != 0:# check if particles are released
                            pdepth_tmp = pdepth[index_realtime,ipart]
                            index_200m = np.argmax(pdepth_tmp > -level)


                            if index_200m != 0: # particles have reached upper -200 m  
                                count = count+1
                                index_200m = np.round(index_realtime[index_200m])

                                x_disp.append( (px[index_200m, ipart]-px[index_realtime[0], ipart] )*dx ) #to have a centered scheme
                                y_disp.append( (py[index_200m, ipart]-py[index_realtime[0], ipart] )*dy )


                    xband, yband = np.linspace(-400,400,dx_pdf+1), np.linspace(-400,400,dx_pdf+1)
                    binbox = (xband[1]-xband[0])*(yband[1]-yband[0])
                    X, Y = x_disp, y_disp
                    H,xedges,yedges = np.histogram2d(X,Y,bins=[xband,yband])
                    #H=H.T
                    N = len(X)
                    jpdf = 1/(binbox*N)*H
                    jpdf_list[l,i,dt,:,:] = jpdf*((800/dx_pdf)**2)
                    i =i+1
        l=l+1
                
    return(jpdf_list)

#Put all the pdf for a 100 days period inside a nc_file
#zdim is created for covolutionnal operation

def Create_nc_file(dt_start,dt_end):
    
    # dt_start = 0 min
    # dt_end = 108 max
    
    delta_t = dt_end - dt_start
    nb_time_step = delta_t*3 # number of cases per time_step
    nb_dt = dt_exp 
    
    i = 0

    pdf_out = np.zeros((nb_time_step,8,36,dx_pdf,dx_pdf))

    for dt in range(dt_start,dt_end,1):
        
        date = dt*nb_dt
        
        print('dt = {0} start'.format(date))
        
        jpdf = compute_outputs(dt)
        
        for dtt in range(2,-1,-1):
            
            pdf_out[i,:,:,:,:] = jpdf[:,:,dtt,:,:]
            i = i+1
                
        print('dt = {0} done'.format(date))

    nc_file = file_pdf  
    nc = nc4.Dataset(nc_file,'w')
        
    #Dimensions used
    nc.createDimension('pdfsize', dx_pdf)
    nc.createDimension('zdim', 8)
    nc.createDimension('nb_time_step', nb_time_step)
    nc.createDimension('position', 36)
    nc.createVariable('pdf', 'f4', ('nb_time_step','zdim','position','pdfsize', 'pdfsize'))      
    nc.variables['pdf'][:] = pdf_out
    nc.close()


# Create the nc file
Create_nc_file(tpas_start,tpas_end)
print('nc file for PDF created at ')
print(folder_pdf)
