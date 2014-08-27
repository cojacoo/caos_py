def macredist(lat,mc,activem):
    import numpy as np
    '''Distribute infiltration to macropores according to macropore drainage area
       Input: lateral position array, mc, bool mask of active macropores, mc.mgrid
       Output: index vector which macropore was appointed
    '''
    m_dist=np.diff(np.append(mc.md_pos[activem],mc.mgrid.width+mc.md_pos[0]))/2
    rightbound=mc.md_pos[activem]+m_dist
    activemacs=np.where(activem==True)
    latid=np.zeros(lat.shape, dtype=int)
    for i in np.arange(len(rightbound)):
        if i==0:
            macid=np.where((lat<rightbound[0]) | (lat>=rightbound[-1]))
        else:
            macid=np.where((lat<rightbound[i]) & (lat>=rightbound[i-1]))
    
        latid[macid]=activemacs[0][i]
    
    return latid


def pmx_infilt(ti,precip,mc,dt,prec_leftover=0):
    from dataread import waterdensity
    from partdyn import assignadvect, cellgrid
    import numpy as np
    import scipy as sp
    import scipy.stats as sps
    import pandas as pd
    '''Infiltration handling:
       A) check rainfall input data
       B) calculate infiltration domain states
       C) redistribution to macropore/matrix interaction
    '''
    # actual temperature of incoming water
    # DEBUG: handle later w/ time series
    T=np.array(9)

    # get timestep in prec time series
    prec_id=np.where((precip.tstart<=ti) & (precip.tend>ti))[0]
    if np.size(prec_id)>0:
        #prec_avail=np.round(precip.intense.values[prec_id]*dt*mc.refarea*waterdensity(T,np.array(-9999))/mc.particlemass)
        prec_avail=np.round(precip.intense.values[prec_id]*dt*mc.mgrid.width.values/mc.particleA)
        #avail. water particles
        prec_c=precip.conc.values[prec_id]
    else:
        prec_avail=0
        prec_c=0.

    # reset particle definition in infilt container
    prec_potinf=prec_avail+prec_leftover
    
    if prec_potinf>0:
        particles_infilt=pd.DataFrame(np.zeros(prec_potinf*9).reshape(prec_potinf,9),columns=['lat', 'z', 'conc', 'temp', 'age', 'cell', 'flag', 'fastlane', 'advect'])
        # place particles at surface and redistribute later according to ponding
        particles_infilt.z=-0.00001
        particles_infilt.lat=np.random.rand(prec_potinf)*mc.mgrid.width.values
        particles_infilt.conc=prec_c[0]
        particles_infilt.temp=T
        particles_infilt.age=ti
        particles_infilt.cell=cellgrid(particles_infilt.lat,particles_infilt.z,mc).astype(int)

        if any(particles_infilt.cell<0):
            print 'cell error at infilt'

        particles_infilt.flag=0
        particles_infilt.fastlane=np.random.randint(len(mc.t_cdf_fast.T), size=prec_potinf)
        particles_infilt.advect=0.
        # first layer as contact layer to surface
        # take cell numbers (binned data) and allow all but one for free drain
        ucellfreq=sps.itemfreq(particles_infilt.cell.values)
        freeparts=np.sum(ucellfreq[ucellfreq[:,1]>0,1]-1)
        # select number of free particles from particles_infilt at random (without proper reference of their position as it was there at random anyways)
        idx_adv=np.random.randint(prec_potinf,size=freeparts)
        if type(mc.nomac)==float:
            idx_adv=prec_potinf #all particles selected
            particles_infilt.lat=mc.md_pos[0] #all into first cell
            particles_infilt.flag=1
        elif mc.nomac!=True:
            # assign to different macropores
            particles_infilt.advect=assignadvect(prec_potinf,mc,particles_infilt.fastlane.values,True)
            activem=np.array([True, True, True, True], dtype=bool) #all macropores are active :: DEBUG: > make dynamic
            idx_red=macredist(particles_infilt.lat.values[idx_adv],mc,activem)
            # assign incidences to particles
            particles_infilt.flag.iloc[idx_adv]=idx_red
            particles_infilt.lat.iloc[idx_adv]=mc.md_pos[idx_red-1]+(np.random.rand(len(idx_adv))-0.5)*mc.mgrid.vertfac.values  
    else:
        particles_infilt=pd.DataFrame([])

    #handle infiltration water as such, that it is prepared to take part in the standard operation
    #a particle is about 1 mm diameter at ks of 10-4m/s a time step of about 10 seconds is maybe a good start
    #maybe time step should be controlled through the actual maximal conductivity?
    return particles_infilt


# !!!
# calculate capacity of macropore at timestep > no; this will be solved dynamically
# check against redistribution capacity > means high intensity will clogg, low intensities will drain...
# ??redistribution capacity > slope, roughness
