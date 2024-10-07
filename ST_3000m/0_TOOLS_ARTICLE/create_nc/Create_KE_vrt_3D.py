
#Load packages
from netCDF4 import Dataset
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import netCDF4 as nc4
sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
sys.path.append("../")
from variables import *
from tools_analyse import *

#Module CROCO
from Modules import *
from Modules_gula import *


dt_sample = 20 # 10 = 5j Tous les dt_sample/2 jours on calcul KE
t_start = 1900 #
t_end = 5900 #
#t_end = 1940 #
half_box = 260 #*4 for km box

#depth = list(range(-200,-3000,-100))
folder = '/home/datawork-lemar-apero/tpicard/KE/POLGYR2/'
name_nc = 'KE_vrt_3D.nc'
file = folder+name_nc 


str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-half_box,jc+half_box,ic-half_box,ic+half_box)
parameters = my_simul +str_para+ format(1900)
simul = load(simul = parameters, floattype=np.float64,output=False)

depth=list(range(0,-3050,-50))

u = var('u',simul,depths=depth).data
v = var('v',simul,depths=depth).data
#vrt =  tools.psi2rho(tools.get_vrt(u[:,:],v[:,:],simul.pm,simul.pn) / tools.rho2psi(simul.f))

u = tools.u2rho(u)
v = tools.v2rho(v)

# using km
[lon,lat] = np.meshgrid(np.arange(simul.x.shape[0])+simul.coord[2],np.arange(simul.x.shape[1])+simul.coord[0])
lon = (lon/np.mean(simul.pm)*1e-3).T
lat = (lat/np.mean(simul.pn)*1e-3).T



def comput_KE_vrt(t,dt_sample,half_box,nb_sample):
    
    vrt=[]
    
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-half_box,jc+half_box,ic-half_box,ic+half_box)
    parameters = my_simul +str_para+ format(t)
    simul = load(simul = parameters, floattype=np.float64,output=False)

    #depth=[-3000]
    
    u = var('u',simul,depths=depth).data
    v = var('v',simul,depths=depth).data
    for i in range(len(depth)):
        vrt.append(tools.psi2rho(tools.get_vrt(u[:,:,i],v[:,:,i],simul.pm,simul.pn) / tools.rho2psi(simul.f)))
    
    vrt = np.array(vrt)
    u = tools.u2rho(u)
    v = tools.v2rho(v)
    
    
    KE = (u**2+v**2)*1/2
    KE = np.swapaxes(KE, 0, 2)
    KE = np.swapaxes(KE, 1, 2)
    
    date_str = [str(simul.year)+'-'+str(simul.month)+'-'+str(simul.day)]
    
    return(KE,vrt,date_str[0])


def Create_nc_file():

    #creating the file
    nc = nc4.Dataset(file,'w')
    #Dimensions used
    nc.createDimension('xdim', half_box*2)
    nc.createDimension('ydim', half_box*2)
    nc.createDimension('time', 0)
    nc.createDimension('z', len(depth))


    nc.createVariable('KE', 'f4', ('time','z','xdim', 'ydim'))
    nc.createVariable('vrt', 'f4', ('time','z','xdim', 'ydim'))
    nc.createVariable('time', str, ('time'))
    nc.createVariable('depth', 'f4', ('z'))
    #nc.createVariable('lon', 'f4', ('xdim','ydim'))
    nc.variables['depth'][:] = depth
    
    print('KE/vrt file create at '+file)
    nc.close()

def Update_nc_file(i,date_str,KE,vrt):
    

    nc = nc4.Dataset(file,'r+')
    nc.variables['KE'][i,:,:,:] = KE
    nc.variables['vrt'][i,:,:,:] = vrt
    nc.variables['time'][i] = date_str
    nc.close()


# CREATION OF EKE

i=0

Create_nc_file()
#(KE,vrt,date_str) = comput_KE_vrt(t_strart,dt_sample,half_box,nb_sample)
#Update_nc_file_EKE(i,date_str,KE,vrt)

dt_10j = 20

for t in range(t_start,t_end+dt_10j,dt_10j):
    (KE,vrt,date_str) = comput_KE_vrt(t,dt_sample,half_box,nb_sample)
    Update_nc_file(i,date_str,KE,vrt)
    i=i+1

