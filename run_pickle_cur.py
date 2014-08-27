import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.constants as const
import mcini_XI as mc
import dataread as dr
import mcpickle as mcp

mcp.mcpick_out(mc,'mc_dump_test2.pickle')
[mc,particles,npart]=dr.particle_setup(mc)
precTS=pd.read_csv(mc.precf, sep=',',skiprows=3)
precTS
import infilt as cinf
import partdyn as pdyn

[thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
drained=pd.DataFrame(np.array([]))
t_end=1200.
output=60.
leftover=0
start_offset=390.
runname='CAOSpy_diff_uf2_'

def CAOSpy_run(tstart,tstop,mc,pdyn,particles,leftover,drained):
    timenow=tstart
    #loop through time
    while timenow < tstop:
        print 'time:',timenow
        [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
        #define dt as Curant criterion
        dt_D=(mc.mgrid.vertfac.values[0])**2 / (2*np.amax(mc.D[np.amax(thS),:]))
        dt_ku=-mc.mgrid.vertfac.values[0]/np.amax(mc.ku[np.amax(thS),:])
        dt=np.amin([dt_D,dt_ku])
        #INFILT
        p_inf=cinf.pmx_infilt(timenow,precTS,mc,dt,leftover)
        #print timenow
        #print p_inf
        particlesnow=pd.concat([particles,p_inf])
        #p_backup=particlesnow.copy()
        #DIFFUSION
        [particlesnow,thS,npart,phi_mx]=pdyn.part_diffusion_split(particlesnow,npart,thS,mc,dt,True,10)
        #ADVECTION
        particlesnow=pdyn.mac_advection(particlesnow,mc,thS,dt)
        #drained particles
        drained=drained.append(particlesnow[particlesnow.flag==len(mc.maccols)+1])
        particlesnow=particlesnow[particlesnow.flag!=len(mc.maccols)+1]
        #MX-MAC-INTERACTION
        pdyn.mx_mp_interact(particlesnow,npart,thS,mc,dt)
        pondparts=(particlesnow.z<0.)
        leftover=np.count_nonzero(-pondparts)
        particles=particlesnow[pondparts]
        timenow=timenow+dt

    return(particles,npart,thS,leftover,drained,timenow)

def plotparticles2(runname,t,ix,particles,npart,mc):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.gridspec as gridspec
    
    fig=plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,2], height_ratios=[1,5])
    ax1 = plt.subplot(gs[0])
    ax11 = ax1.twinx()
    advect_dummy=np.bincount(np.round(100.0*particles.loc[((particles.age>200.)),'lat'].values).astype(np.int))
    old_dummy=np.bincount(np.round(100.0*particles.loc[((particles.age<200.)),'lat'].values).astype(np.int))
    ax1.plot((np.arange(0,len(advect_dummy))/100.)[1:],advect_dummy[1:],'b-')
    ax11.plot((np.arange(0,len(old_dummy))/100.)[1:],old_dummy[1:],'g-')
    ax11.set_ylabel('Particle Count', color='g')
    ax11.set_xlim([0.,mc.mgrid.width.values])
    ax1.set_xlim([0.,mc.mgrid.width.values])
    ax1.set_ylabel('New Particle Count', color='b')
    ax1.set_xlabel('Lat [m]')
    ax1.set_title('Lateral Particles Concentration')
    
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    ax2.text(0.1, 0.8, 'Particles @ t='+str(np.round(t))+'s', fontsize=20)
    
    ax3 = plt.subplot(gs[2])
    plt.imshow(sp.ndimage.filters.median_filter(npart,size=mc.smooth),vmin=1, vmax=mc.part_sizefac, cmap='jet')
    #plt.imshow(npart)
    plt.colorbar()
    plt.xlabel(''.join(['Width [cells a ',str(np.round(1000*mc.mgrid.latfac.values[0],decimals=1)),' mm]']))
    plt.ylabel(''.join(['Depth [cells a ',str(np.round(1000*mc.mgrid.vertfac.values[0],decimals=1)),' mm]']))
    plt.title('Particle Density')
    plt.tight_layout()

    ax4 = plt.subplot(gs[3])
    #ax41 = ax4.twiny()
    z1=np.append(particles.loc[((particles.age>200.)),'z'].values,mc.onepartpercell[1][:mc.mgrid.vertgrid.values.astype(int)])
    advect_dummy=np.bincount(np.round(-100.0*z1).astype(np.int))-1
    old_dummy=np.bincount(np.round(-100.0*particles.loc[((particles.age<200.)),'z'].values).astype(np.int))
    ax4.plot(advect_dummy,(np.arange(0,len(advect_dummy))/-100.),'r-',label='new particles')
    ax4.plot(advect_dummy+old_dummy,(np.arange(0,len(old_dummy))/-100.),'b-',label='all particles')
    ax4.plot(old_dummy,(np.arange(0,len(old_dummy))/-100.),'g-',label='old particles')
    ax4.set_xlabel('Particle Count')
    #ax4.set_xlabel('New Particle Count', color='r')
    ax4.set_ylabel('Depth [m]')
    #ax4.set_title('Number of Particles')
    ax4.set_ylim([mc.mgrid.depth.values,0.])
    ax4.set_xlim([0.,np.max(old_dummy+advect_dummy)])
    #ax41.set_xlim([0.,np.max(old_dummy[1:])])
    #ax41.set_ylim([mc.mgrid.depth.values,0.])
    handles1, labels1 = ax4.get_legend_handles_labels() 
    #handles2, labels2 = ax41.get_legend_handles_labels() 
    ax4.legend(handles1, labels1, loc=4)
    #    ax41.legend(loc=4)
    plt.savefig(''.join([runname,str(ix).zfill(3),'.png']))
    #savefig('runname %(i)03d .png'.translate(None, ' '))
    plt.close(fig)

#loop through plot cycles
dummy=np.floor(t_end/output)
for i in np.arange(dummy.astype(int)):
    [particles,npart,thS,leftover,drained,t]=CAOSpy_run(i*output+start_offset,(i+1)*output+start_offset,mc,pdyn,particles,leftover,drained)
    plotparticles2(runname,t,i,particles,npart,mc)
