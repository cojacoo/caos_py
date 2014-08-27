# coding=utf-8

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.constants as const
# http://docs.scipy.org/doc/scipy/reference/constants.html#module-scipy.constants


# MODEL INI & DATA READ
# import mcini as mc

def cdf(x):
    scol = np.sum(x)
    if scol>0:
        result = x/scol
    else:
        result = x*0
    return result

def minloc(x):
    result=np.min(np.where(x==0))
    return result


def waterdensity(T,P):
    '''
       Calc density of water depending on T [°C] and P [Pa]
       defined between 0 and 40 °C and given in g/m3
       Thiesen Equation after CIPM
       Tanaka et al. 2001, http://iopscience.iop.org/0026-1394/38/4/3

       NOTE: the effect of solved salts, isotopic composition, etc. remain
       disregarded here. especially the former will need to be closely
       considerd in a revised version! DEBUG.
       
       INPUT:  Temperature T in °C as numpy.array
               Pressure P in Pa as numpy.array (-9999 for not considered)
       OUTPUT: Water Density in g/m3
       
       EXAMPLE: waterdensity(np.array((20,21,42)),np.array(-9999.))
       (cc) jackisch@kit.edu
    '''
    
    import numpy as np
    
    # T needs to be given in °C
    a1 = -3.983035  # °C        
    a2 = 301.797    # °C        
    a3 = 522528.9   # °C2       
    a4 = 69.34881   # °C        
    a5 = 999974.950 # g/m3
    
    dens=a5*(1-((T+a1)**2*(T+a2))/(a3*(T+a4)))
    
    # P needs to be given in Pa
    # use P=-9999 if pressure correction is void
    if P.min()>-9999:
        c1 = 5.074e-10   # Pa-1     
        c2 = -3.26e-12   # Pa-1 * °C-1  
        c3 = 4.16e-15    # Pa-1 * °C-2  .
        Cp = 1 + (c1 + c2*T + c3*T**2) * (P - 101325)
        dens=dens*Cp
    
    # remove values outside definition bounds and set to one
    if (((T.min()<0) | (T.max()>40)) & (T.size>1)):
        idx=np.where((T<0) | (T>40))
        dens[idx]=100000.0  #dummy outside defined bounds 
    
    return dens

def mc_diffs(mc):
    '''Calculate diffs for D calculation
    '''
    import numpy as np
    D=np.empty((100,mc.soilmatrix.no.size))
    ku=np.empty(101)
    kumx=np.empty((101,mc.soilmatrix.no.size))
    psi=np.empty(101)
    psimx=np.empty((101,mc.soilmatrix.no.size))
    theta=np.empty(101)
    thetamx=np.empty((101,mc.soilmatrix.no.size))
    cH2O=np.empty(101)
    cH2Omx=np.empty((101,mc.soilmatrix.no.size))
    

    for i in np.arange(mc.soilmatrix.no.size):
        for j in np.arange(101):
            thetaS=float(j)/100.
            ku[j]=mc.soilmatrix.ks[i]*thetaS**0.5*(1.-(1.-thetaS**(1./mc.soilmatrix.m[i]))**mc.soilmatrix.m[i])**2.
            psi[j]= -1 * ( (1-thetaS**(1./mc.soilmatrix.m[i])) / (thetaS**(1./mc.soilmatrix.m[i]) ))**(1./mc.soilmatrix.n[i]) / mc.soilmatrix.alpha[i]
            psi[0]=-1.0e+10
            theta[j]=thetaS*(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])+mc.soilmatrix.tr[i]
            dummy=-mc.soilmatrix.m[i]*(1./(1.+np.abs((psi[j])*mc.soilmatrix.alpha[i])**mc.soilmatrix.n[i]))**(mc.soilmatrix.m[i]+1) *mc.soilmatrix.n[i]*(abs(psi[j])*mc.soilmatrix.alpha[i])**(mc.soilmatrix.n[i]-1.)*mc.soilmatrix.alpha[i]
            cH2O[j]=-(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])*dummy

        DI=(ku[:-1]+(np.diff(ku)/2.))*np.diff(psi)/np.diff(theta)
        DI[0]=0 #define very low diffusion at zero
        D[:,i]=DI
        psimx[:,i]=psi
        thetamx[:,i]=theta
        kumx[:,i]=ku
        cH2Omx[:,i]=cH2O

#DEBUG: Diffusive Flux is overestimated!
#       THIS NEEDS THROUGOUT TESTING
#       SET D SMALL ENOUGH FOR NOW.

    mc.D=D#/100.
    mc.psi=psimx
    mc.theta=thetamx
    mc.ku=kumx
    mc.cH2O=cH2Omx

#DEBUG: This is thetaS based. 
#       However, we may need a psi based approach since this 
#       is establishing the respective gradient.
    ku=np.empty(121)
    p_kumx=np.empty((121,mc.soilmatrix.no.size))
    theta=np.empty(121)
    p_thetamx=np.empty((121,mc.soilmatrix.no.size))
    cH2O=np.empty(121)
    p_cH2Omx=np.empty((121,mc.soilmatrix.no.size))

    for i in np.arange(mc.soilmatrix.no.size):
        for j in np.arange(121):
            psi=-10**((float(j)/10.)-2.)
            v = 1. + (mc.soilmatrix.alpha[i]* np.abs(psi))**mc.soilmatrix.n[i]
            ku[j] = (mc.soilmatrix.ks[i] * (1. - ((mc.soilmatrix.alpha[i]*np.abs(psi))**(mc.soilmatrix.n[i]-1))*(v**(-mc.soilmatrix.m[i])) )**2. / (v**(mc.soilmatrix.m[i]*0.5)))/3600.
            thetaS = (1./(1.+(psi*mc.soilmatrix.alpha[i])**mc.soilmatrix.n[i]))**mc.soilmatrix.m[i]
            theta[j]=thetaS*(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])+mc.soilmatrix.tr[i]
            dummy=-mc.soilmatrix.m[i]*(1./(1.+np.abs(psi*mc.soilmatrix.alpha[i])**mc.soilmatrix.n[i]))**(mc.soilmatrix.m[i]+1.) *mc.soilmatrix.n[i]*(np.abs(psi)*mc.soilmatrix.alpha[i])**(mc.soilmatrix.n[i]-1.)*mc.soilmatrix.alpha[i]
            cH2O[j]=-(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])*dummy

        p_thetamx[:,i]=theta
        p_kumx[:,i]=ku
        p_cH2Omx[:,i]=cH2O

    mc.p_th=p_thetamx
    mc.p_ku=p_kumx
    mc.p_cH2O=p_cH2Omx

    #get FC at psi = -0.33 bar
    idx = (mc.psi>-0.35)
    idy = (mc.psi<-0.30)
    idz = np.where(idx*idy)
    idf = np.zeros(np.shape(mc.psi)[1],dtype=int)
    for i in np.unique(idz[1]):
        idf[i] = idz[0][np.where(idz[1]==i)[0][0]]
    mc.FC=idf

    return mc


def dataread_caos(mc):
    macbase=pd.read_csv(mc.macbf, sep=' ')
    tracerbase=pd.read_csv(mc.tracerbf, sep='\t')
    soilmatrix=pd.read_csv(mc.matrixbf, sep=' ')

    #calculate missing van genuchten m
    soilmatrix['m'] = 1-1/soilmatrix.n

    #covert tracer profile into advective velocity distribution
    #tracer concentration is used as proxy :: columnwise normalisation
    tracerbase=pd.read_csv(mc.tracerbf, sep='\t')

    t_cdf=tracerbase.apply(cdf,axis=0)
    #this is the cdf of normalised tracer concentrations

    #FIGURE
    #fig, ax = plt.subplots()
    #heatmap = ax.pcolor(t_cdf, cmap=pylab.cm.Blues, alpha=0.8)
    #ax.invert_yaxis()

    mc.a_velocity=np.arange(-mc.tracer_vertgrid/2,-mc.tracer_vertgrid*(len(tracerbase)+0.5),-mc.tracer_vertgrid)/mc.tracer_t_ex
    #this is the vector of velocities

    ## DEVISION INTO FAST AND SLOW HUMP
    cutrow=np.min(t_cdf.apply(minloc,axis=0))
    #debug!!!
    cutrow=0

    #slow hump
    mc.t_cdf_slow=t_cdf[0:cutrow].apply(cdf,axis=0)

    #fast hump
    mc.t_cdf_fast=t_cdf[cutrow+1:].apply(cdf,axis=0)

    #GET MACROPORE INI FUNCTIONS
    import macropore_ini as mpo

    if mc.nomac==False:
        #READ MACROPORE DATA FROM IMAGE FILES
        mac=pd.read_csv(mc.macimg, sep=',')
        patch_def=mpo.macfind_g(mac.file[0],mac.threshold[0])

        for i in np.arange(len(mac)-1)+1:
            patch_def=patch_def.append(mpo.macfind_g(mac.file[i],mac.threshold[i]))

        #join macropore definitions
        patch_def=patch_def.set_index(np.arange(len(mac)))
        mac=mac.join(patch_def)

        #get macropore share at advection vector
        z_centroids=np.arange(-mc.tracer_vertgrid/2,-mc.tracer_vertgrid*(len(tracerbase)+0.5),-mc.tracer_vertgrid)
        share_idx=(z_centroids*0.).astype(int)
        for zc in np.arange(len(z_centroids)):
            if -z_centroids[zc]>=mac.depth.iloc[0]:
                share_idx[zc]=np.where(mac.depth<=-z_centroids[zc])[0][-1]+1
                if -z_centroids[zc]>mac.depth.iloc[-2]:
                    share_idx[zc]=len(mac.depth)-1
            else:
                share_idx[zc]=0
        mc.macshare=mac.share[share_idx]
    elif mc.nomac==True:
        mc.macshare=pd.Series(0.0)
        mac=pd.DataFrame([dict(no=1, share=0.00001, 
                                 minA=0., maxA=0., meanA=0., medianA=0.,
                                 minP=0., maxP=0., meanP=0., medianP=0.,
                                 minDia=0., maxDia=0., meanDia=0., medianDia=0.,
                                 minmnD=1., maxmnD=1., meanmnD=1., medianmnD=1.,
                                 minmxD=1., maxmxD=1., meanmxD=1., medianmxD=1.,
                                 depth=mc.soildepth), ])  
    else:
        mc.macshare=pd.Series(0.001)
        mac=pd.DataFrame([dict(no=1, share=0.001, 
                                 minA=0.0001, maxA=0.0001, meanA=0.0001, medianA=0.0001,
                                 minP=0.01*np.pi, maxP=0.01*np.pi, meanP=0.01*np.pi, medianP=0.01*np.pi,
                                 minDia=0.01, maxDia=0.01, meanDia=0.01, medianDia=0.01,
                                 minmnD=0.5, maxmnD=0.5, meanmnD=0.5, medianmnD=0.5,
                                 minmxD=0.5, maxmxD=0.5, meanmxD=0.5, medianmxD=0.5,
                                 depth=mc.soildepth), ])  


    #calculate cumulative velocity in macropores only
    a_velocity_real=-np.cumsum(np.cumsum(mc.tracer_vertgrid/mc.macshare)/((np.arange(len(mc.macshare))+1)*mc.tracer_t_ex))
    mc.a_velocity_real=a_velocity_real.values

    mc=mpo.mac_matrix_setup(mac,mc)
    mc.soilmatrix=soilmatrix
    if mc.nomac==False:
        mpo.mac_plot(mc.macP)
    mc=mc_diffs(mc)

    print 'MODEL SETUP READY.'
    return mc #[mc,soilmatrix,macP,mac,macid,macconnect,soilgrid,matrixdef,mc.mgrid]


def particle_setup(mc):
    import numpy as np
    import pandas as pd
    # read ini moist
    inimoistbase=pd.read_csv(mc.inimf, sep=',')

    # check for psi/theta definition - convert to theta if necessary
    # this is left for later...

    moistdomain=np.zeros((mc.mgrid.vertgrid,mc.mgrid.latgrid))
    npartdomain=np.zeros((mc.mgrid.vertgrid,mc.mgrid.latgrid))
    # assign moisture to grid
    for i in np.arange(len(inimoistbase)):
        idx0=int(round(inimoistbase.zmin[i]/mc.mgrid.vertfac))
        idx1=int(round(inimoistbase.zmax[i]/mc.mgrid.vertfac))
        moistdomain[idx0:idx1,:]=inimoistbase.theta[i]
        
    # plt.imshow(moistdomain)

    # define particle size
    # WARNING: as in any model so far, we have a volume problem here. 
    #          we consider all parts of the domain as static in volume at this stage. 
    #          however, we will work on a revision of this soon.
    mc.gridcellA=mc.mgrid.vertfac*mc.mgrid.latfac
    mc.particleA=abs(mc.gridcellA.values)/(2*mc.part_sizefac) #assume average ks at about 0.5 as reference of particle size
    mc.particleD=2.*np.sqrt(mc.particleA)/np.pi
    mc.particleV=3./4.*np.pi*(mc.particleD/2.)**3.
    mc.particlemass=waterdensity(np.array(20),np.array(-9999))*mc.particleV #assume 20°C as reference for particle mass
                                                                            #DEBUG: a) we assume 2D=3D; b) change 20°C to annual mean T?

    # define macropore capacity based on particle size
    # we introduce a scale factor for converting macropore space and particle size
    mc.maccap=np.round(mc.md_area/((mc.particleD**2)*np.pi*mc.macscalefac)).astype(int)

    # convert theta to particles
    npart=moistdomain*(2*mc.part_sizefac)

    # setup particle domain
    particles=pd.DataFrame(np.zeros(int(np.sum(npart))*8).reshape(int(np.sum(npart)),8),columns=['lat', 'z', 'conc', 'temp', 'age', 'flag', 'fastlane', 'advect'])
    particles['cell']=pd.Series(np.zeros(int(np.sum(npart)),dtype=int), index=particles.index)
    # distribute particles
    k=0
    npartr=npart.ravel()
    cells=len(npartr)

    for i in np.arange(cells):
        j=int(npartr[i])
        particles.cell[k:(k+j)]=i
        rw,cl=np.unravel_index(i,(mc.mgrid.vertgrid,mc.mgrid.latgrid))
        particles.lat[k:(k+j)]=(cl+np.random.rand(j))*mc.mgrid.latfac.values
        particles.z[k:(k+j)]=(rw+np.random.rand(j))*mc.mgrid.vertfac.values
        k+=j

    particles.fastlane=np.random.randint(len(mc.t_cdf_fast.T), size=len(particles))

    mc.mgrid['cells']=cells
    return [mc,particles,npart]

