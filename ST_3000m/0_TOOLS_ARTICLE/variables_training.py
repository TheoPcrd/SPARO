#### DO NOT EDIT YET ##### 
half_reso = 260 # = 520 point square centered at PAP = 1040 km square
coef_pooling = 4 # Resolution = 2km x coef pooling
nb_dx = int(400/coef_pooling) # Final resolution = ((half_reso*2)-120)/coef_pooling
dx_pdf = 100
dt_exp = 60 # 30 days time step (time that separate 2 experiences)
dt_image_sampling = 20 #images are sampled every 10 days
test = False # training data
dx_reso = 2 #Choose resolution of the inputs
""

# ############ TO EDIT ##########################################################

# ############ TRAINING PARAMETERS ##############################################

batch_size = 32
max_epochs = 50
num_workers=0
nb_gpus = 1
alpha1 = 0.8
alpha2 = 1 - alpha1

# dirSAVE = './saved_model/supermodel_wsed_{0}_sdepth{1}/'.format(wsed,depth_trap)
# name_model = 'CNN_UNET_k{0}_p{1}_b{2}_d{3}_nl{4}_ni_{5}_dx{6}km'.format(kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs,dx_reso)


# nb_convTrans4 = 0

# ############ INPUTS DATA PARAMETERS ############################################

depth_cst = False # False = adapt 4 vertical levels depending on the trap depth
nb_sample = 5 # Number of images per experiment 
full_time_exp = 140 # 50 days time step (time of 1 experience)
wsed = 80 # sinking speed of particles
depth_trap = 3000 #Depth of the sediment trap
list_level_cst=True # if true, compute pdf at : [900m,800m,...,200m], else compute pdf at list_level = np.linspace(200,depth_trap,9)[1:]

folder_in_pyticle = '/home2/scratch/tpicard/Pyticles/outputs_simu2/'
#folder_pdf = '/home2/scratch/tpicard/DATA_CNN/pdf/'
#folder_images = '/home2/scratch/tpicard/DATA_CNN/image_inputs/'

folder_pdf = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_{0}_stdepth_{1}/'.format(wsed,depth_trap)
folder_images = '/home/datawork-lemar-apero/tpicard/DATA_CNN/wsed_{0}_stdepth_{1}/'.format(wsed,depth_trap)

# ### NO NEED TO EDIT ############################################################## 

zdim = nb_sample*4*4+nb_sample #Number of images per experience (4levels)

n=0
index_surface=[]
for i in range(zdim):
    if n/17==1 :
        index_surface.append(i)
        n=1
    elif n%4==0:
        index_surface.append(i)
        n=n+1
    else:
        n=n+1  

index_training_start = 0
index_training_end = 240
index_validation_start = 250
index_validation_end = 285

tpas_start = 0 # First experience ?
tpas_end = 65 #65 Number of experiences
date_start = 1900
date_end = date_start + full_time_exp + 60*(tpas_end-1) # date end
date_end_test = date_end 

my_simul = 'aperitif_simu2'
name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)
name_pdf ='pdf_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_pdf)
name_pdf_filter = 'filter_'+name_pdf
name_input = 'inputs_{0:06}_{1:06}_wsed{2}_stdepth{3}_zdim{4}_dx{5}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,zdim,dx_reso)


file_raw_images_test = folder_images+name_raw_images # NAME FILE RAW DATA  
file_input_test = folder_images+name_input
file_pdf_test = folder_pdf + name_pdf
file_pdf_filter_test = folder_pdf + name_pdf_filter

if test ==True:
    tpas_start = 0 # First experience ?
    tpas_end = 65 #65 Number of experiences
    date_start = 1900
    date_end = date_start + full_time_exp + 60*(tpas_end-1) # date end
    my_simul = 'aperitif_simu2'
    name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)
    name_pdf ='pdf_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}_testing.nc'.format(date_start,date_end,wsed,depth_trap,dx_pdf)
    name_pdf_filter = 'filter_'+name_pdf
    name_input = 'inputs_{0:06}_{1:06}_wsed{2}_stdepth{3}_zdim{4}_dx{5}km_testing.nc'.format(date_start,date_end,wsed,depth_trap,zdim,dx_reso)
    file_input_training = folder_images+name_input
else: 
    tpas_start = 0
    tpas_end = 95 # 95 Number of experiences
    date_start = 710
    date_end = date_start + full_time_exp + 60*(tpas_end-1) # date end
    my_simul = 'apero'
    name_raw_images = 'raw_images_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}km_training.nc'.format(date_start,date_end,wsed,depth_trap,dx_reso)
    name_pdf ='pdf_{0:06}_{1:06}_wsed{2}_stdepth{3}_dx{4}_training.nc'.format(date_start,date_end,wsed,depth_trap,dx_pdf)
    name_pdf_filter = 'filter_'+name_pdf
    name_input = 'inputs_{0:06}_{1:06}_wsed{2}_stdepth{3}_zdim{4}_dx{5}km_training.nc'.format(date_start,date_end,wsed,depth_trap,zdim,dx_reso)

file_raw_images = folder_images+name_raw_images # NAME FILE RAW DATA  
file_input = folder_images+name_input
file_pdf = folder_pdf + name_pdf
file_pdf_filter = folder_pdf + name_pdf_filter

""
