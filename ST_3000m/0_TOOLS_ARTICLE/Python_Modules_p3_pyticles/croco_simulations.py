'''
CV 2020/09/14: class with functions associated to croco simulations 
''' 
import numpy as np
from netCDF4 import Dataset 
import calendar as cal
import time as time
import datetime as datetime
import sys
sys.path.append('/home2/datahome/cvic/Python_Modules_p3/')
sys.path.append('/home2/datahome/tpicard/python/Python_Modules_p3/')
import R_tools_fort as toolsF

path_data = '/home/datawork-lops-rrex/cvic/' 

class Croco(object): 
    def __init__(self,*args,**kwargs):
        print(' ------------ initiate a class with croco simulation ------------ ') 
        self.name_exp    = args[0] 
        self.output_freq = args[1] 
        self.date_ini    = args[2]
        self.date_end    = args[3]
        self.grid_nc     = args[4]
        self.path_data   = path_data+self.name_exp.upper()+'/HIS/'
        # --- time formatting ---
        if self.name_exp in ['rrextra','rrextrb','rrextrc','rrextrd','rrextre','RREXNUM100']:
            if   self.output_freq == '1h':
                self.nfpf = 24 # number of frames per file   
                self.fs   = 1  # file step in days                  
            elif self.output_freq == '1d':
                self.nfpf = 5  # number of frames per file   
                self.fs   = 5  # file step in days                  
        self.time_ini = cal.timegm(self.date_ini)
        self.time_end = cal.timegm(self.date_end)
        time_sec      = np.arange(self.time_ini,self.time_end+self.fs*86400,self.fs*86400) # time line in seconds  
        self.nfiles   = time_sec.shape[0]-1
        time_fmt      = [time.gmtime(time_sec[i]) for i in range(self.nfiles+1)] # formatting 
        self.nt       = self.nfiles*self.nfpf
        # - list with file name for each frame -  
        self.file_names  = [] 
        self.frame_index = []
        file_avg = self.path_data+self.name_exp.upper()+'_'+self.output_freq+'_avg_' 
        for f in range(self.nfiles):
            for t in range(self.nfpf):   
                suffix0   = '%.4i-%.2i-%.2i'%(time_fmt[f][0],time_fmt[f][1],time_fmt[f][2])
                suffix1   = '%.4i-%.2i-%.2i'%(time_fmt[f+1][0],time_fmt[f+1][1],time_fmt[f+1][2])
                self.file_names.append(file_avg+suffix0+'-'+suffix1+'.nc')
                self.frame_index.append(t)
                #print(self.file_names)
        return  

    def get_grid(self):
        print(' ... get grid variables ... ') 
        #nc = Dataset(self.path_data+self.name_exp+'_grd.nc','r')
        nc = Dataset(self.path_data+self.grid_nc,'r')
        self.lonr = np.asfortranarray(nc.variables['lon_rho'][:].T)          
        self.latr = np.asfortranarray(nc.variables['lat_rho'][:].T) 
        self.h    = np.asfortranarray(nc.variables['h'][:].T)
        self.f    = np.asfortranarray(nc.variables['f'][:].T)
        self.Cs_r = nc.variables['Cs_r'][:] 
        self.Cs_w = nc.variables['Cs_w'][:]
        self.hc   = nc.variables['hc'][:] 
        self.pm   = np.asfortranarray(nc.variables['pm'][:].T) 
        self.pn   = np.asfortranarray(nc.variables['pn'][:].T) 
        nc.close()  
        return  

                

    def get_outputs(self,*args,**kwargs): 
        t = args[0] 
        get_date = kwargs.get('get_date',False)
        print('===================================================') 
        print(' ... get output variables at time index %.4i ... '%t) 
        print('===================================================') 
        var_list = args[1]
        nvar = len(var_list) 
        self.var = {} 
        nc = Dataset(self.file_names[t],'r')
        for var_name in var_list:
            print('     --> ',var_name) 
            self.var[var_name] = np.asfortranarray(nc.variables[var_name][self.frame_index[t]].T) 
        self.rho0 = nc.rho0
        self.hc   = nc.hc
        
        if get_date:  
            time_c = nc.variables['time_centered'][self.frame_index[t]] 
            diff_days = (datetime.datetime(1979,1,1) - datetime.datetime(1970,1,1)).days
            time_c += diff_days*86400
            #print('time_c =')
            #print(time_c)
            time_gm = time.gmtime(int(time_c)) 
            self.ymdhms = [datetime.datetime(time_gm[0],time_gm[1],time_gm[2],
                                             time_gm[3],time_gm[4],time_gm[5])]
            self.time_gm = time_gm
        nc.close()     
        return  
        

    def get_zlevs(self):
        print('     --> get vertical levels ')  
        [self.z_r,self.z_w] = toolsF.zlevs(self.h,self.var['zeta'],self.hc,self.Cs_r,self.Cs_w)
        return 
