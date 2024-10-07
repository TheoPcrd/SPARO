
### DATA BASE CREATION FOR TRAINING AND TEST ###

************************************************************************
STEP 1 : Extract particle catchment areas PDF from Lagrangian experience 
************************************************************************

-->Create_nc_pdf_8levels_100x100.py

    - return 1 NetCDF with pdf = ('nb_time_step'= nb_time_step,'zdim' = 8,'position' = 36 ,'pdfsize'= 100 , 'pdfsize' = 100)
    - zdim is the 8 vertical layers where PDFs are computed
    - pdfsize is the horizontal resolution (100x100 points over a 800km x 800km --> 8km resolution)
    - nb_time_step : Time index of particle release 
    - position : Position of the sediment trap
    - All PDF a centred with the particles source locations
    
Positions corresponding (see paper) :

31--32---33---34---35
|   |     |    |    |
25--26-..
|   |
19--20-..
|   |
13--14-..
|   |
7---8--12..
|   |   |   |   |   |
0---1---2---3---4---5

  
[Dans INPUTS : Fichier input.generic à éditer
- Fichier submit_run_Pyticles_test.sh : éditer les variables (wsed, depth_trap, tstart, tend, temps_experience). Attention en backtracking les temps sont inversés]


************************************************************************
STEP 2 : PDF Gaussian filtering
************************************************************************

--> Filter_Heat_equation_auto.py
    
    Same structure:
    - pdf_filter = ('nb_time_step'= nb_time_step,'zdim' = 8,'position' = 36 ,'pdfsize'= 100 , 'pdfsize' = 100) 
    
************************************************************************
Step 3 : Associated dynamical variables
************************************************************************

--> Create_nc_images_surface.py

    - Create 1 file with dynamical variables associated with each Lagrangian experience
    - variables are 2km resolution and computed in a 1040 km box centred at PAP station
    - Temperature, Vorticity, U, V, and SSH 
    - 4 time steps every 10 days
    - 4 vertical levels : 0m, 200m, 500m, 1000m
    
    
************************************************************************
STEP 5 : Centering raw dynamical variables in each corresponding 
sediment trap position + downscalling 
************************************************************************

--> Create_inputs.py


************************************************************************
STEP 6 : Testing
************************************************************************

--> Test_data.ipynb : Check everything is all right