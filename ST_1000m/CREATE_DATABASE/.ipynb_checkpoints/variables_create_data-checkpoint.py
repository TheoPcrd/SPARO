############# TO EDIT #################

test = True # Test or training data ?
depth_cst = True # True : Depth = [0, 200, 500, 1000] # False = adapt 4 vertical levels depending on the trap depth
nb_sample = 4 # Number of dt per experiment
zdim = nb_sample*4*4+nb_sample # Total number of images per experience with 4 levels
full_time_exp = 120 # 60 days time step (time of 1 Lagrangian output)
wsed = 50 # sinking speed of particles
depth_trap = 1000 #Depth of the sediment trap
list_level_cst=True # if true, compute pdf at : [900m,800m,...,200m], else compute pdf at list_level = np.linspace(200,depth_trap,9)[1:]

#### DO NOT EDIT#####
half_reso = 260 # = 520 point square centered at PAP = 1040 km square
coef_pooling = 4 # Resolution = 2km x coef pooling
nb_dx = int(400/coef_pooling) # Final resolution = ((half_reso*2)-120)/coef_pooling
dt_exp = 60 # 30 days time step (time that separate 2 Lagrangian outputs)
dt_image_sampling = 20 #images are sampled every 10 days
dx_pdf = 100
dx_reso = 2 #24 Choose resolution of the inputs


############# FOLDERS AND FILES #################

folder_pdf = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_{0}_stdepth_{1}/'.format(wsed,depth_trap)
folder_images = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_{0}_stdepth_{1}/'.format(wsed,depth_trap)

if test ==True:
    tpas_start = 0 # First experience ?
    tpas_end = 65 #65 Number of experiences
    date_start = 1900
    date_end = date_start + full_time_exp + 60*(tpas_end-1) # date end
    folder_in_pyticle = '/home2/scratch/tpicard/Pyticles/outputs_simu2/' # Folder with lagrangian data
    my_simul = 'aperitif_simu2'
    name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)
    name_pdf ='pdf_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_pdf)
    name_pdf_filter = 'filter_'+name_pdf
    name_input = 'inputs_{0:06}_{1:06}_wsed{2}_stdepth{3}_zdim{4}_dx{5}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,zdim,dx_reso)
    
else: 
    tpas_start = 0
    tpas_end = 95 #95
    date_start = 710
    date_end = date_start + full_time_exp + 60*(tpas_end-1) # date end
    folder_in_pyticle = '/home2/scratch/tpicard/Pyticles/outputs/' # Folder with lagrangian data
    my_simul = 'apero'
    name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_training.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)
    name_pdf ='pdf_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}_training.nc'.format(date_start,date_end,wsed,depth_trap,dx_pdf)
    name_pdf_filter = 'filter_'+name_pdf
    name_input = 'inputs_{0:06}_{1:06}_wsed{2}_stdepth{3}_zdim{4}_dx{5}km_training.nc'.format(date_start,date_end,wsed,depth_trap,zdim,dx_reso)
    
file_raw_images = folder_images+name_raw_images # NAME FILE RAW DATA  
file_input = folder_images+name_input
file_pdf = folder_pdf + name_pdf
file_pdf_filter = folder_pdf + name_pdf_filter


