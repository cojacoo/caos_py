# CAOS EFU Model Parameter File
#timing and files

wd='.'                     #working_directory                       
precf='irr_testcase.dat'   #top_boundary_flux_precip                
etf='etp.dat'              #top_boundary_flux_etp                   
inimf='moist_testcase.dat' #initial_moisture                        
outdir='out'               #output_directory                        
t_end=864                  #end_time[s]                             
part_sizefac=10            #particle_sized_definition_factor        
grid_sizefac=0.005         #grid_size_definition_factor [m]            
subsfac=10                 #subsampling_rate[percent]
macscalefac=10             #scaling factor converting converting macropore space and particle size 
t_dini=12                  #initial_time_step[s]                    
t_dmx=12                   #maximal_time_step[s]                    
t_dmn=0.01                 #minimal_time_step[s]                    
refarea=1.                 #reference_area_of_obs[m2]               
soildepth=-1.2             #depth_of_soil_column[m]
smooth=(6,6)             #smoothing window for thS calculations as no. of cells

#macropore 
macbf='macbase.dat'        #macropore definition file

#bromid tracer data
tracerbf='tracer.dat'      #tracer base file
tracer_t_ex=8600.0         #time to excavation   
tracer_horgrid=0.1         #gridspacing horizontally   
tracer_vertgrid=0.1        #gridspacing vertically

tracer_site=97
tracer_CI=25.43
tracer_SD_CI=2.1
tracer_appl_Br=4.19595
tracer_SD_Br=0.3465
tracer_time=2.3
tracer_intensity=11.05652174
tracer_c_br=0.165

#soil matrix properties
matrixbf='matrix.dat'      #matrix base file
matrixdeffi='matrixdef.dat'  #matrix definition file
