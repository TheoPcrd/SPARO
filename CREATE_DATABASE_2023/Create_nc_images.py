#!/usr/bin/env python
# coding: utf-8

# We create 64 nc files that containt all the images neeeded for corresponding period


########## VARIABLES  #############
from variables_create_data import *

#Load packages
from netCDF4 import Dataset
import numpy as np
import sys
import netCDF4 as nc4
#sys.path.append("/home2/datahome/tpicard/Pyticles/Modules/")
sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
from Modules import *
from Modules_gula import *

[ic,jc] = np.load('/home2/datahome/tpicard/Pyticles/Inputs/ic_jc.npy')

# Take a time start and a center (ic,jc) that correspond to the PAP station
# Compute ssh, temperature, vorticity, u and v field 
# at the corresponding time 4 vertical layers
# All images are (520 x 520) spatial point resolution

def compute_variable(tstart,ic,jc):
    
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-half_reso,jc+half_reso,ic-half_reso,ic+half_reso)
    parameters = my_simul +str_para+ format(tstart)
    simul = load(simul = parameters, floattype=np.float64)

    if depth_cst==False:
        depth = -np.linspace(0,depth_trap,5)[:-1]
        depth = depth.tolist()
    else :
        depth = [0,-200,-500,-1000]
    
    temp = var('temp',simul,depths=depth).data
    ssh = var('zeta',simul).data
    u = var('u',simul,depths=depth).data
    v = var('v',simul,depths=depth).data
    vrt = np.zeros(temp.shape)
    for i in range(len(depth)):
        vrt[:,:,i] =  tools.psi2rho(tools.get_vrt(u[:,:,i],v[:,:,i],simul.pm,simul.pn) / tools.rho2psi(simul.f))
    u = tools.u2rho(u)
    v = tools.v2rho(v)
    
    ##############################################################
    # Define horizontal coordinates (deg, km, or grid points)
    ########################################################

    coord = 'km'

    if coord=='deg':
        #using lon,lat
        lon = simul.x; lat = simul.y
        xlabel = 'lon'; ylabel = 'lat'
    elif coord=='km':
        # using km
        [lon,lat] = np.meshgrid(np.arange(simul.x.shape[0])+simul.coord[2],np.arange(simul.x.shape[1])+simul.coord[0])
        lon = (lon/np.mean(simul.pm)*1e-3).T
        lat = (lat/np.mean(simul.pn)*1e-3).T
        xlabel = 'km'; ylabel = 'km'
    elif coord=='points':
        # using grid points
        [lon,lat] = np.meshgrid(np.arange(simul.x.shape[0])+simul.coord[2],np.arange(simul.x.shape[1])+simul.coord[0])
        lon,lat = lon.T,lat.T
        xlabel = 'grid pts'; ylabel = 'grid pts'
        
    lon = lon - lon[int(lon.shape[0]/2),0]
    lat = lat - lat[0,int(lat.shape[1]/2)]
    
    vrt = np.swapaxes(vrt, 0, 2)
    vrt = np.swapaxes(vrt, 1, 2)
    temp = np.swapaxes(temp, 0, 2)
    temp = np.swapaxes(temp, 1, 2)
    u = np.swapaxes(u, 0, 2)
    u = np.swapaxes(u, 1, 2)
    v = np.swapaxes(v, 0, 2)
    v = np.swapaxes(v, 1, 2)
    
    return (vrt,temp,ssh,u,v)



# # Create the nc file
# For one period, create a nc file containning all the images needed

def Create_nc_file(date_start,date_end):

    nb_tpas = int((date_end-date_start)/20)
    
    vrt,temp,ssh,u,v = compute_variable(date_start,ic,jc)
    
    #creating the file
    nc = nc4.Dataset(file_raw_images,'w')

    #Dimensions used
    nc.createDimension('xdim', half_reso*2)
    nc.createDimension('ydim', half_reso*2)
    nc.createDimension('depth', 4)
    nc.createDimension('time', nb_tpas+1)

    #Variables used
    nc.createVariable('temperature', 'f4', ('time','depth','xdim', 'ydim'))
    nc.createVariable('vorticity', 'f4', ('time','depth','xdim', 'ydim'))
    nc.createVariable('u', 'f4', ('time','depth','xdim', 'ydim'))
    nc.createVariable('v', 'f4', ('time','depth','xdim', 'ydim'))
    nc.createVariable('ssh', 'f4', ('time','xdim', 'ydim'))
    nc.close()

    for i in range(0,nb_tpas+1):
        
        nc = nc4.Dataset(file_raw_images,'r+')
        
        time = date_start + i*20
        vrt,temp,ssh,u,v = compute_variable(time,ic,jc)
        nc.variables['temperature'][i,:,:,:] = temp
        nc.variables['vorticity'][i,:,:,:] = vrt
        nc.variables['u'][i,:,:,:] = u
        nc.variables['v'][i,:,:,:] = v
        nc.variables['ssh'][i,:,:] = ssh
        
        nc.close()

# Create allnc files
Create_nc_file(date_start,date_end)

print('Images created at ')
print(file_raw_images)



