from netCDF4 import Dataset
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys

sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
#sys.path.append('/home2/datahome/lwang/Pyticles/Modules/')

from Modules import *
from Modules_gula import *

def plot_background(ic,jc,t):

    # parameters = my_simul + [0,nx,0,ny,[1,nz,1]] ; nx, ny, nz Roms domain's shape 
    dx, dy = 1.979, 1.979 #grid size
    my_simul = 'apero'

    #parameters = my_simul + ' [1068,2068,117,1117,[1,100,1]] '+ format(date_plot_AC)
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-125,jc+125,ic-125,ic+125)
    parameters = my_simul +str_para+ format(t)
    simul = load(simul = parameters, floattype=np.float64)

    depth = -200
    
    temp = var('temp',simul,depths=[depth]).data
    ssh = var('zeta',simul).data
    u = var('u',simul,depths=[depth]).data
    v = var('v',simul,depths=[depth]).data
    vrt =  tools.psi2rho(tools.get_vrt(u,v,simul.pm,simul.pn) / tools.rho2psi(simul.f))
    
           
    ##############################################################
    # Define horizontal coordinates (deg, km, or grid points)
    ########################################################

    coord = 'points'

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
        
    lon = (lon - ic)*dx
    lat = (lat - jc)*dy
    
    return (lon,lat,vrt)

def plot_background_8l(ic,jc,t):

    # parameters = my_simul + [0,nx,0,ny,[1,nz,1]] ; nx, ny, nz Roms domain's shape 
    dx, dy = 1.979, 1.979 #grid size
    my_simul = 'apero'

    #parameters = my_simul + ' [1068,2068,117,1117,[1,100,1]] '+ format(date_plot_AC)
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-125,jc+125,ic-125,ic+125)
    parameters = my_simul +str_para+ format(t)
    simul = load(simul = parameters, floattype=np.float64)

    depths = [-200,-300,-400,-500,-600,-700,-800,-900]
    
    temp = var('temp',simul,depths=depths).data
    ssh = var('zeta',simul).data
    u = var('u',simul,depths=depths).data
    v = var('v',simul,depths=depths).data
            
    vrt=np.zeros((250,250,len(depths)))
    for i in range(len(depths)): 
        vrt[:,:,i] =  tools.psi2rho(tools.get_vrt(u[:,:,i],v[:,:,i],simul.pm,simul.pn) / tools.rho2psi(simul.f))
    
    
           
    ##############################################################
    # Define horizontal coordinates (deg, km, or grid points)
    ########################################################

    coord = 'points'

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
        
    lon = (lon - ic)*dx
    lat = (lat - jc)*dy
    
    return (lon,lat,vrt)

def plot_background_level(ic,jc,t,list_level):

    # parameters = my_simul + [0,nx,0,ny,[1,nz,1]] ; nx, ny, nz Roms domain's shape 
    dx, dy = 1.979, 1.979 #grid size
    my_simul = 'apero'

    #parameters = my_simul + ' [1068,2068,117,1117,[1,100,1]] '+ format(date_plot_AC)
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-125,jc+125,ic-125,ic+125)
    parameters = my_simul +str_para+ format(t)
    simul = load(simul = parameters, floattype=np.float64)

    depths = -np.array(list_level)
    
    temp = var('temp',simul,depths=depths).data
    ssh = var('zeta',simul).data
    u = var('u',simul,depths=depths).data
    v = var('v',simul,depths=depths).data
            
    vrt=np.zeros((250,250,len(depths)))
    for i in range(len(depths)): 
        vrt[:,:,i] =  tools.psi2rho(tools.get_vrt(u[:,:,i],v[:,:,i],simul.pm,simul.pn) / tools.rho2psi(simul.f))
    
    
           
    ##############################################################
    # Define horizontal coordinates (deg, km, or grid points)
    ########################################################

    coord = 'points'

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
        
    lon = (lon - ic)*dx
    lat = (lat - jc)*dy
    
    return (lon,lat,vrt)


def plot_background_level(ic,jc,t,list_level,topo=None):

    # parameters = my_simul + [0,nx,0,ny,[1,nz,1]] ; nx, ny, nz Roms domain's shape 
    dx, dy = 1.979, 1.979 #grid size
    my_simul = 'apero'

    #parameters = my_simul + ' [1068,2068,117,1117,[1,100,1]] '+ format(date_plot_AC)
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-125,jc+125,ic-125,ic+125)
    parameters = my_simul +str_para+ format(t)
    simul = load(simul = parameters, floattype=np.float64)

    depths = -np.array(list_level)
    
    temp = var('temp',simul,depths=depths).data
    ssh = var('zeta',simul).data
    u = var('u',simul,depths=depths).data
    v = var('v',simul,depths=depths).data
    
    vrt=np.zeros((250,250,len(depths)))
    for i in range(len(depths)): 
        vrt[:,:,i] =  tools.psi2rho(tools.get_vrt(u[:,:,i],v[:,:,i],simul.pm,simul.pn) / tools.rho2psi(simul.f))
    
    if topo:
        topo_map =  simul.topo
        
           
    ##############################################################
    # Define horizontal coordinates (deg, km, or grid points)
    ########################################################

    coord = 'points'

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
        
    lon = (lon - ic)*dx
    lat = (lat - jc)*dy
    
    if topo:
        return(lon,lat,vrt,topo_map)
    else:
        return (lon,lat,vrt)

def plot_background_level_time(ic,jc,time,level):

    # parameters = my_simul + [0,nx,0,ny,[1,nz,1]] ; nx, ny, nz Roms domain's shape 
    dx, dy = 1.979, 1.979 #grid size
    my_simul = 'apero'

    #parameters = my_simul + ' [1068,2068,117,1117,[1,100,1]] '+ format(date_plot_AC)
    str_para = ' [{0},{1},{2},{3},[1,100,1]] '.format(jc-125,jc+125,ic-125,ic+125)
    parameters = my_simul +str_para+ format(time)
    simul = load(simul = parameters, floattype=np.float64)

    depths = -np.array(level)
    
    #temp = var('temp',simul,depths=depths).data
    #ssh = var('zeta',simul).data
    u = var('u',simul,depths=[depths]).data
    v = var('v',simul,depths=[depths]).data
            
    vrt=np.zeros((250,250))
    vrt=tools.psi2rho(tools.get_vrt(u[:,:],v[:,:],simul.pm,simul.pn) / tools.rho2psi(simul.f))
    
    
    ##############################################################
    # Define horizontal coordinates (deg, km, or grid points)
    ########################################################

    coord = 'points'

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
        
    lon = (lon - ic)*dx
    lat = (lat - jc)*dy
    print(vrt.shape)
    return (lon,lat,vrt)



def plot_pdf_200m(time,ic,jc,x_disp,y_disp,npart_trap):
    
    

    from matplotlib import rc, rcParams
    rc('axes', linewidth=2)
    
    (lon,lat,vrt) = plot_background(ic,jc,time)
    
    nb_dx = 100
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    cmap = 'viridis_r'
    xband, yband = np.linspace(-250,250,101), np.linspace(-250,250,101)
    binbox = (xband[1]-xband[0])*(yband[1]-yband[0])
    X, Y = x_disp, y_disp
    #H,xedges,yedges = np.histogram2d(X,Y,bins=[xband,yband])
    H,xedges,yedges = np.histogram2d(X,Y,bins=[xband,yband])
    N = len(X)
    jpdf_nomask = 1/(binbox*N)*H*((500/nb_dx)**2)
    jpdf =  np.ma.masked_where(jpdf_nomask < 0.000001,jpdf_nomask)

    plt.pcolormesh(lon,lat,vrt,alpha=1,cmap=plt.cm.RdBu_r,vmin=-1,vmax=1)
    #plt.pcolormesh(lon,lat,list_vrt[:,:],alpha=1,cmap=plt.cm.RdBu_r)
    cb = plt.colorbar()
    cb.set_label(r'$\zeta$ / f', fontsize=18, rotation=270,labelpad=20)
    cb.ax.tick_params(labelsize=16)
    #ctf = ax.contourf(0.5*(xband[:-1]+xband[1:]), 0.5*(yband[:-1]+yband[1:]), jpdf.T,  levels, norm=norm, cmap=cmap, extend='both')
    pmesh = ax.pcolormesh(0.5*(xband[:-1]+xband[1:]), 0.5*(yband[:-1]+yband[1:]), jpdf.T, cmap=cmap,vmin=0,vmax=np.max(jpdf))
    cb = fig.colorbar(pmesh, ax=ax)
    cb.formatter.set_powerlimits((0, 0))
    #cb.set_ticks([5.9e-3, 3e-3, 0])
    cb.set_label('Probability density of particles ', fontsize=18, rotation=270,labelpad=20)
    cb.ax.tick_params(labelsize=16)
    #plt.ticklabel_format(style="sci", scilimits=(0,0))
    cb.ax.tick_params(labelsize='large')
    ax.set_xlim(-250,250)
    ax.set_ylim(-250,250)
    #ax.set_xlim(-400,400)
    #ax.set_ylim(-400,400)

    plt.axvline(x=0,color='black',linestyle='--',linewidth=0.5)
    plt.axhline(y=0,color='black',linestyle='--',linewidth=0.5)

    #plt.xticks([-250,0,250])
    #plt.yticks([-250,0,250])

    plt.xticks([-200,-100,0,100,200])
    plt.yticks([-200,-100,0,100,200])

    #plt.text(600,600,'[km' + r'$^{-2}$' + ']', fontsize=16)
    plt.xlabel('[km]', fontsize=20)
    plt.ylabel('[km]', fontsize=20)
    #plt.text(-650,600,'[km]', fontsize=20)
    #plt.text(600,-600,'[km]', fontsize=20)
    #plt.text(-600, 650, '(' + label[i] + ')', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid(b=True, which='major', color='gray', linestyle='--',alpha=0.1)
    plt.grid(b=True, which='minor', color='gray', linestyle='--',alpha=0.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.title('{0} particles'.format(npart_trap), fontsize=16)

    xline = np.linspace(-250, 250, 250)
    yline = np.linspace(-250, 250, 250)

    levels = [0.000001]

    plt.contour(0.5*(xband[:-1]+xband[1:]), 0.5*(yband[:-1]+yband[1:]),jpdf_nomask.T,levels=levels,colors='k')

# Function that take all the particules positions px and py 
# (pxcenter,pycenter) is the center of a sub-patch that containt 36 particles 
# index_start is generally 0, 20 or 40 and correspond to the number of time step after the starting period of simu
# Return a list of index that correspond to the particles located inside the sub-patch

def spatial_filter(px,py,pxcenter,pycenter,index_start):
    
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

def spatial_filter(px,py,pxcenter,pycenter,index_start):
    
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

[ic_ori,jc_ori] = np.load('/home2/datahome/tpicard/Pyticles/Inputs/ic_jc.npy')

from netCDF4 import Dataset
import netCDF4 as nc4

def jpdf_no_centering(date_start):


    ########################
    # Choose the folder and step time
    ########################
    nb_dx = 100

    folder = '/home2/datawork/tpicard/DATA_CNN/Pyticle_outputs/'

    
    date_end = date_start + 100

    #file = folder + 'apero_trap1000m_sed50_2000dx_100nx_{0:06}_{1:06}_56_{1:04}.nc'.format(date_start,date_end)
    file = folder + 'aperitif_simu2_trap1000m_sed50_2000dx_100nx_{0:06}_{1:06}_56_{1:04}.nc'.format(date_start,date_end)

    #file = folder + 'aperitif_simu2_trap1000m_sed50_2000dx_100nx_001900_002000_56_2000.nc'



    # 36 : number of sub-pathes 
    # 3 : number of time step considered in a 101 days period
    # 50x50 : Spatial resolution of the pdf
    # 9 : number of vertical levels

    #time_step: From 0 to 108 

    jpdf_list = np.zeros((9,36,3,nb_dx,nb_dx))

    list_level = [1000,900,800,700,600,500,400,300,200]
    x_disp_level = np.zeros((len(list_level),720))
    y_disp_level = np.zeros((len(list_level),720))

    #List of the 36 sub-patches centers 
    ic_all_list = np.linspace(1520.5,1610.5,6) #LONGITUDE
    jc_all_list = np.linspace(569.5,659.5,6) #LAT

    #Load data
    nc_data = nc4.Dataset(file, 'r')
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

                                x_disp.append( (px[index_200m, ipart]-ic_ori )*dx ) #to have a centered scheme
                                y_disp.append( (py[index_200m, ipart]-jc_ori )*dy )


                    xband, yband = np.linspace(-260,260,100+1), np.linspace(-260,260,100+1)
                    binbox = (xband[1]-xband[0])*(yband[1]-yband[0])
                    X, Y = x_disp, y_disp
                    H,xedges,yedges = np.histogram2d(X,Y,bins=[xband,yband])
                    N = len(X)
                    H = H.T 
                    jpdf = 1/(binbox*N)*H
                    jpdf_list[l,i,dt,:,:] = jpdf*((520/100)**2)
                    i =i+1
        l=l+1

    print(jpdf_list.shape)
    return(jpdf_list)

def plot_pdf_level(time,ic,jc,x_disp,y_disp,npart_trap,list_level,level,vmax):
    
    

    from matplotlib import rc, rcParams
    rc('axes', linewidth=2)
    
    (lon,lat,vrt) = plot_background_level(ic,jc,time,list_level)
    
    nb_dx = 100
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    cmap = 'viridis_r'
    xband, yband = np.linspace(-250,250,101), np.linspace(-250,250,101)
    binbox = (xband[1]-xband[0])*(yband[1]-yband[0])
    X, Y = x_disp, y_disp
    #H,xedges,yedges = np.histogram2d(X,Y,bins=[xband,yband])
    H,xedges,yedges = np.histogram2d(X,Y,bins=[xband,yband])
    N = len(X)
    jpdf_nomask = 1/(binbox*N)*H*((500/nb_dx)**2)
    jpdf =  np.ma.masked_where(jpdf_nomask < 0.000001,jpdf_nomask)

    plt.pcolormesh(lon,lat,vrt[:,:,level],alpha=1,cmap=plt.cm.RdBu_r,vmin=-vmax,vmax=vmax)
    #plt.pcolormesh(lon,lat,list_vrt[:,:],alpha=1,cmap=plt.cm.RdBu_r)
    cb = plt.colorbar()
    cb.set_label(r'$\zeta$ / f', fontsize=18, rotation=270,labelpad=20)
    cb.ax.tick_params(labelsize=16)
    #ctf = ax.contourf(0.5*(xband[:-1]+xband[1:]), 0.5*(yband[:-1]+yband[1:]), jpdf.T,  levels, norm=norm, cmap=cmap, extend='both')
    pmesh = ax.pcolormesh(0.5*(xband[:-1]+xband[1:]), 0.5*(yband[:-1]+yband[1:]), jpdf.T, cmap=cmap,vmin=0,vmax=np.max(jpdf))
    cb = fig.colorbar(pmesh, ax=ax)
    cb.formatter.set_powerlimits((0, 0))
    #cb.set_ticks([5.9e-3, 3e-3, 0])
    cb.set_label('Probability density of particles ', fontsize=18, rotation=270,labelpad=20)
    cb.ax.tick_params(labelsize=16)
    #plt.ticklabel_format(style="sci", scilimits=(0,0))
    cb.ax.tick_params(labelsize='large')
    ax.set_xlim(-250,250)
    ax.set_ylim(-250,250)
    #ax.set_xlim(-400,400)
    #ax.set_ylim(-400,400)

    plt.axvline(x=0,color='black',linestyle='--',linewidth=0.5)
    plt.axhline(y=0,color='black',linestyle='--',linewidth=0.5)

    #plt.xticks([-250,0,250])
    #plt.yticks([-250,0,250])

    plt.xticks([-200,-100,0,100,200])
    plt.yticks([-200,-100,0,100,200])

    #plt.text(600,600,'[km' + r'$^{-2}$' + ']', fontsize=16)
    plt.xlabel('[km]', fontsize=20)
    plt.ylabel('[km]', fontsize=20)
    #plt.text(-650,600,'[km]', fontsize=20)
    #plt.text(600,-600,'[km]', fontsize=20)
    #plt.text(-600, 650, '(' + label[i] + ')', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid(b=True, which='major', color='gray', linestyle='--',alpha=0.1)
    plt.grid(b=True, which='minor', color='gray', linestyle='--',alpha=0.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.title('{0} particles'.format(npart_trap), fontsize=16)

    xline = np.linspace(-250, 250, 250)
    yline = np.linspace(-250, 250, 250)

    levels = [0.000001]

    plt.contour(0.5*(xband[:-1]+xband[1:]), 0.5*(yband[:-1]+yband[1:]),jpdf_nomask.T,levels=levels,colors='k')

    
