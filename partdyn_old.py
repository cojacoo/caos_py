# coding=utf-8

import numpy as np
import scipy as sp

#particle dynamics
#macropore mc.soilmatrix interaction
def cellgrid(lat,z,mc):
    import numpy as np
    rw=np.floor(z/mc.mgrid.vertfac.values)
    cl=np.floor(lat/mc.mgrid.latfac.values)
    cell=rw*mc.mgrid.latgrid.values + cl
    return cell.astype(np.int64)


def gridupdate(particles,mc):
    '''Calculates grid state from particle density
    '''
    npart=np.zeros((mc.mgrid.vertgrid,mc.mgrid.latgrid), dtype=int64)*(2*mc.part_sizefac)
    npart[np.unravel_index(np.arange(mc.mgrid.cells.astype(int64)),(mc.mgrid.vertgrid.astype(int64),mc.mgrid.latgrid.astype(int64)))] = np.bincount(particles.cell)
    #theta=npart*mc.particleA/mc.mgrid.gridcellA
    return npart


def thetaSid(npart,mc):
    '''Calculates thetaS id from npart
       Use the result as: mc.D[thetaSid(npart,mc.soilmatrix,mc,mc.soilgrid).astype(int),mc.soilgrid.ravel()-1]
       on prepared D, ku, theta, psi which are stored in mc
    '''
    thetaS=npart.ravel()/(mc.soilmatrix.ts[mc.soilgrid.ravel()-1]*(2*mc.part_sizefac))
    return thetaS*100

def thetaSidx(npart,mc):
    '''Calculates thetaS id from npart and cuts at max.
       WARNING: oversaturation and draining will simply be accepted!!!
       Use the result as: mc.D[thetaSid(npart,mc.soilmatrix,mc,mc.soilgrid).astype(int),mc.soilgrid.ravel()-1]
       on prepared D, ku, theta, psi which are stored in mc
    '''
    thetaS=npart.ravel()/(mc.soilmatrix.ts[mc.soilgrid.ravel()-1]*(2*mc.part_sizefac))
    thetaS[thetaS>1.]=1.
    thetaS[thetaS<0.]=0.
    return thetaS*100


def gridupdate_thS(lat,z,mc):
    '''Calculates thetaS from particle density
    '''
    import numpy as np
    import scipy as sp
    import scipy.ndimage as spn
    npart=np.zeros((mc.mgrid.vertgrid,mc.mgrid.latgrid), dtype=np.int64)*(2*mc.part_sizefac)
    lat1=np.append(lat,mc.onepartpercell[0])
    z1=np.append(z,mc.onepartpercell[1])
    cells=cellgrid(lat1,z1,mc)
    trycount = np.bincount(cells)
    trycount=trycount-1 #reduce again by added particles
    npart[np.unravel_index(np.arange(mc.mgrid.cells.astype(np.int64)),(mc.mgrid.vertgrid.values.astype(np.int64),mc.mgrid.latgrid.values.astype(np.int64)))] = trycount
    npart_s=spn.filters.median_filter(npart,size=mc.smooth)
    thetaS=npart_s.ravel()/(mc.soilmatrix.ts[mc.soilgrid.ravel()-1]*(2*mc.part_sizefac))
    thetaS[thetaS>0.99]=0.99
    return [(thetaS.values*100).astype(np.int),npart]


def npart_theta(npart,mc):
    '''Calculates theta from npart
    '''
    import numpy as np
    import scipy as sp
    import scipy.ndimage as spn
    npart_s=spn.filters.median_filter(npart,size=mc.smooth)
    theta=(mc.particleA/(-mc.gridcellA))[0]*npart_s.ravel()
    return theta

def theta_D(th,soilmatrix,mc):
    ks=soilmatrix.ks[soil-1]
    m=soilmatrix.m[soil-1]
    thr=soilmatrix.tr[soil-1]
    ths=soilmatrix.ts[soil-1]
    alpha=soilmatrix.alpha[soil-1]
    the=(th-thr)/(ths-thr)
    Dd=(ks*(1-m)*(the**(0.5-(1/m)))) / (alpha*m*(ths-thr)) *( (1-the**(1/m))**(-1*m) + (1-the**(1/m))**m -2 )
    return Dd

def boundcheck(lat,z,mc):
    #cycle bound:
    if any(lat<0.0):
        lat[lat<0.0]=mc.mgrid.width[0]+lat[lat<0.0]
    if any(lat>mc.mgrid.width[0]):
        lat[lat>mc.mgrid.width[0]]=lat[lat>mc.mgrid.width[0]]-mc.mgrid.width[0]
    #topbound - DEBUG: set zero for now, may interfere with infilt leftover idea
    if any(z>0.0):
        z[z>0.0]=-0.00001
    #lowbound - leave to drain DEBUG: may be a case parameter!!!
    #DEBUG: maybe allow ku defined flux
    nodrain=(z>=mc.mgrid.depth[0])
    z[-nodrain]=mc.mgrid.depth[0]+0.000000000001
    return [lat,z,nodrain]

def macredist(lat,mc,activem):
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

def assignadvect(no,mc,dummy=-99):
    import numpy as np
    if any(dummy==-99): #fast lane not given
        dummy=np.random.randint(len(mc.t_cdf_fast.T), size=no)

    dummx=np.random.rand(no)
    cum_cdf=np.array(mc.t_cdf_fast.cumsum(axis=0))
    l_cdf=cum_cdf.shape[0]
    idx=abs(np.floor(cum_cdf[:,dummy]-dummx.repeat(l_cdf).reshape(cum_cdf[:,dummy].T.shape).T).sum(axis=0)).astype(np.int)
    adv=mc.a_velocity[idx]
    return adv

def mac_advection(particles,mc,thS,dt,maccol):
    '''Calculate Advection in Macropore
       Advective particle movement in macropore with retardation of the advective momentum through drag at interface,
       plus check for each macropore's capacity and possible overload.
    '''
    import numpy as np
    import scipy as sp
    import dataread as dr
    import scipy.constants as const
    import matplotlib.pyplot as plt

#DEBUG: the arrays are not yet fully specified!!!
#       no iteration for overfilling yet!!!
#       Transition with the soil mc.soilmatrix needs to be solved elsewhere!!!
#       Multicore to handle many macropores at once (as they are independent)
    macstate=[]
    if not particles.empty:
        #project advection step
        #macropore is divided in grid of particle diameter steps
        z_proj=particles.z.values+particles.advect.values*dt

        nodrain=(z_proj>=mc.soildepth)
        if any(-nodrain):
            z_proj[-nodrain]=mc.soildepth
        particles_zid=np.floor(-particles.z.values/mc.particleD).astype(int)
        
        #DEBUG:
        particles_zid[particles_zid>=len(mc.macgrid)]=len(mc.macgrid)-1
        macgrid=np.zeros((-mc.soildepth/mc.particleD).astype(np.int64))
        filling=sp.stats.itemfreq(particles_zid).astype(np.int)
        macgrid[filling[:,0].astype(np.int64)]=filling[:,1]
        proj_zid=np.floor(abs(z_proj/mc.particleD)).astype(int)
        particles_zid=np.floor(abs(particles.z.values/mc.particleD)).astype(int)
        proj_id=np.floor(z_proj/mc.mgrid.vertfac.values).astype(int)
        #check free slots to mc.soilmatrix along path
        def contactcount(idx,idy):
            return np.count_nonzero(macgrid[idx:idy]==0)
        vcontactcount=np.vectorize(contactcount)
        dragweight=vcontactcount(particles_zid,proj_zid)
        #check macropore capacity (clogging)
        def findincr(x):
            idx=np.where(mc.md_depth>x)
            if len(idx)>1:
                idxx=idx[0][0]
            else:
                idxx=len(mc.md_depth)
            return idxx
        vfindincr = np.vectorize(findincr)
        idx=vfindincr(-z_proj)
        def clogpos(idx,idy):
            x=-1
            if (idx!=idy):
                t = macgrid[idx:idy]-mc.maccap[maccol,mc.macgrid[idx]]
                if any(t<0.):
                    x=idx+np.nonzero(t<0.)[0][0] #zid of clogged cell
            return x
        vclogpos=np.vectorize(clogpos)
        
        #DEBUG: (put runaway particles back into domain)
        particles_zid[particles_zid>=len(mc.macgrid)]=len(mc.macgrid)-1
        proj_zid[proj_zid>=len(mc.macgrid)]=len(mc.macgrid)-1

        clog=vclogpos(particles_zid,proj_zid)
        # #(projected) translational kinetic energy of particle
        # E_transkin=0.5*mc.particlemass*particles.advect**2
        #experienced psi
        idx=(np.arange(mc.mgrid.vertgrid.values.astype(np.int))*mc.mgrid.latgrid.values+mc.maccols[maccol]).astype(np.int)
        psi=mc.psi[thS[idx],mc.soilgrid.ravel()[idx]-1]
        #E_psi=np.mean((psi[particles_zid],psi[proj_id]),axis=0)*mc.particlemass*const.g
        contactfac=np.ones(len(particles_zid),dtype=np.float64)
        ia=dragweight>0.
        ib=(proj_zid-particles_zid)>0.
        if any(ia & ib):
            contactfac[ia & ib]=1.-(dragweight[ia & ib]/(proj_zid[ia & ib]-particles_zid[ia & ib]))

        # #perform advection
        # particles_znew=particles.z+retard_fac*particles.advect*dt
        #DEBUG: proj_zid or proj_id???
        psib=np.mean((psi[particles_zid],psi[proj_zid]),axis=0)*0.0980665 #matric head conversion into bars
        pm=mc.particlemass/1000 #particle mass conversion into kg
        s_red=(pm*particles.advect.values**2)/(2*((psib*mc.particleA*contactfac)+((pm*particles.advect.values)/(2*dt))))
        #s_red=(dt*particles.advect.values)/(psib*mc.particleA*contactfac)
        particles_znew=particles.z.values+s_red
        #print s_red/particles.advect.values*dt

        #cut advection at clogging
        cid=(clog>=0)
        if any(cid):
            #update z_proj to center of last free cell before clog
            z_proj[cid]=-mc.particleD*(clog[cid]-0.5)
            particles_znew[cid]=np.amax([particles_znew[cid],z_proj[cid]],axis=0)

        nodrain=(particles_znew>=mc.soildepth)
        #DEBUG: calculate phi >> especially if macs get filled
        #check for continuous clog from bottom

        particles_zid=np.floor(particles_znew/mc.mgrid.vertfac.values).astype(int)
        particles_zid[particles_zid>=len(mc.macgrid)]=len(mc.macgrid)-1
        filling=sp.stats.itemfreq(particles_zid)
        macstate=np.zeros(mc.mgrid.vertgrid.values)
        #print particles.z.values
        #print particles_znew
        macstate[filling[:,0].astype(np.int64)]=filling[:,1]/(-mc.mgrid.vertfac.values/mc.particleD)

        #macropore matrix interaction -> infiltration
        particles_mzid=np.floor(particles_znew/-mc.particleD).astype(int)
        #DEBUG:
        particles_mzid[particles_mzid>=len(mc.macgrid)]=len(mc.macgrid)-1

        idx=(particles_zid*mc.mgrid.latgrid.values+np.floor(mc.md_pos[maccol]/mc.mgrid.latfac.values).astype(int)).astype(int)
        N=len(idx)
        macstatex=(macstate*100).astype(int)
        macstatex[macstatex>99]=99
        mpsi=mc.psi[macstatex[particles_mzid],mc.soilgrid.ravel()[idx]-1]
        mxpsi=mc.psi[thS[idx],mc.soilgrid.ravel()[idx]-1]
        gradient=mpsi-mxpsi #use to skew xi
        gradxi=np.tanh(gradient/20.)+np.random.rand(N) # *2.-1. not needed here
        D=mc.D[macstatex[particles_mzid],mc.soilgrid.ravel()[idx]-1]
        step_proj=(gradxi*((6*D*dt)**0.5))
        idx=(step_proj>=mc.particleD/2.)
        if any(idx):
            particles.flag[idx]=0 #particle infiltrated
            particles.lat[idx]=mc.md_pos[maccol]+mc.md_contact[maccol,mc.macgrid[particles_zid[idx]]]*(np.random.rand(sum(idx))-0.5)

        particles.z=particles_znew
        #particles.ix[:]['cell']=cellgrid(particles.lat.values,particles.z.values,mc).astype(np.int64)
        particles.cell=cellgrid(particles.lat.values,particles.z.values,mc).astype(np.int64)
        if any(-nodrain):
            particles.loc[-nodrain,'flag']=len(mc.maccols)+1


#    return particles #[particles,macstate]
    return(particles)
    #DEBUG: macstate needs revision, dropped for now


def mx_mp_interact(particles,npart,thS,mc,dt):
    idx=np.where(thS>mc.FC[mc.soilgrid-1].ravel())
    if len(idx[0])>0:
        #a) exfiltration into macropores
        idy=mc.macconnect.ravel()[idx[0]]>0
        if any(idy):
            idc=np.in1d(particles.cell.values.astype(np.int),idx[0][idy]) #get index vector which particles are in affected cells
            #DEBUG: maybe check if any idc is true?
            #we assume diffusive transport into macropore - allow diffusive step and check whether particeD/2 is moved -> then assign to macropore
            N=np.sum(idc)
            xi=np.random.rand(N)
            D=mc.D[thS[particles.cell[idc].values],mc.soilgrid.ravel()[particles.cell[idc].values]-1]
            step_proj=(xi*((6*D*dt)**0.5))
            ida=(step_proj>=mc.particleD/2.)
            if any(ida):
                particles.loc[idc[ida],'flag']=mc.macconnect.ravel()[particles.cell[idc[ida]].values    ]
        #b) bulk flow advection
        idb=np.in1d(particles.cell.values.astype(np.int),idx[0].astype(np.int))
        # allow only one particle in appropriate cell to move advectively - DEBUG: this should be an explicit parameter
        dummy, idad = np.unique(particles.cell[idb].values.astype(np.int), return_index=True)
        ida=np.where(idb)[0][idad]
        z_new=particles.z.values[ida]+particles.advect.values[ida]*dt
        z_new[z_new<mc.soildepth]=mc.soildepth+0.0000001
        particles.z.iloc[ida]=z_new
        particles.cell.iloc[ida]=cellgrid(particles.lat.values[ida],particles.z.values[ida],mc).astype(np.int64)

    return particles


def part_diffusion(particles,npart,thS,mc,dt,uffink_corr=True):
    '''Calculate Diffusive Particle Movement
       Based on state in grid use diffusivity as foundation of 2D random walk.
       Project step and check for boundary conditions and further restrictions.
       Update particle positions.
    '''
    N=len(particles.z) #number of particles handled
    
    # 1D Random Walk function with additional correction term for
    # non-static diffusion after Uffink 1990 p.15 & p.24ff
    xi=np.random.rand(N,2)*2.-1.
    psi_id=((np.log10(-mc.psi[thS,mc.soilgrid.ravel()-1].astype(np.float64)+mc.zgrid.ravel()-mc.mgrid.depth.values)+2)*10.).astype(int)
    u=mc.p_ku[psi_id,mc.soilgrid.ravel()-1]
    D=u/mc.cH2O[thS,mc.soilgrid.ravel()-1]

    #step_proj=dt*u[particles.cell.values.astype(np.int)].repeat(2).reshape((N,2)) + (xi*((2*D[particles.cell.values.astype(np.int)].repeat(2).reshape((N,2))*dt)**0.5))
    step_proj=(xi*((2*D[particles.cell.values.astype(np.int)].repeat(2).reshape((N,2))*dt)**0.5))

    if (uffink_corr==True):
        dx=map(np.linalg.norm, step_proj)
        # project step and updated state
        # new positions
        lat_proj=particles.lat.values+step_proj[:,1]
        z_proj=particles.z.values+step_proj[:,0]
        [lat_proj,z_proj,nodrain]=boundcheck(lat_proj,z_proj,mc)

        [thS,npart]=gridupdate_thS(lat_proj,z_proj,mc) #DEBUG: externalise smooth parameter)
        #cut thS>1 and thS<0 for projection
        thSx=thS
        thSx[thSx>100.]=100.
        thSx[thSx<0.]=0.
        psi_id=((np.log10(-mc.psi[thSx.astype(int),mc.soilgrid.ravel()-1].astype(np.float64)+mc.zgrid.ravel()-mc.mgrid.depth.values)+2)*10.).astype(int)
        u_proj=mc.p_ku[psi_id,mc.soilgrid.ravel()-1]
        D_proj=u_proj/mc.cH2O[thSx,mc.soilgrid.ravel()-1]

        corrD=np.abs(D[particles.cell.values.astype(np.int)]-D_proj[particles.cell.values.astype(np.int)])/dx
        corru=np.sqrt(u[particles.cell.values.astype(np.int)]*u_proj[particles.cell.values.astype(np.int)])
        #DEBUG: probably reconsider if lateral advection is reasonable
        # corrected step
        step_proj=np.append(corrD,corrD).reshape((N,2))*dt + (xi*((2*D[particles.cell.values.astype(np.int)].repeat(2).reshape((N,2))*dt)**0.5))

    # new positions
    lat_new=particles.lat.values+step_proj[:,1]
    z_new=particles.z.values+step_proj[:,0]
    [lat_new,z_new,nodrain]=boundcheck(lat_new,z_new,mc)

    # saturation check
    [thS,npart]=gridupdate_thS(lat_new[nodrain],z_new[nodrain],mc) #DEBUG: externalise smooth parameter

    phi_mx=mc.psi[thS,mc.soilgrid.ravel()-1]+mc.mxdepth_cr

    particles.ix[:]['z']=z_new
    particles.ix[:]['lat']=lat_new
    particles.ix[:]['cell']=cellgrid(lat_new,z_new,mc).astype(np.int64)
    if any(-nodrain):
        particles.loc[-nodrain,'flag']=len(mc.maccols)+1

    return [particles,thS,npart,phi_mx]


#DEBUG: multicore for diffusion - split particle sample for handling and rejoin...?
#       drawback: matrix update needs to be external to routine
#       one may also think of using multicore rather at next hierarchical level, so a hillslope takes some time as a single column


def mac_advection_mc(particles,mc,thS,dt):
    #run mac_advection over all macropores parallel
    from multiprocessing import Process, Queue, Pool
    #create a process Pool with a number processes (maccol)
    pool = Pool(processes=len(mc.maccols))
    def macadv_wrapper(maccol):
        return mac_advection(particles[particles.flag==maccol],mc,thS,dt,maccol) #[particles,macstate] dropped for now
    # map to pool processes
    results = pool.map(macadv_wrapper, tuple(np.arange(len(mc.maccols))+1))
    return pd.concat(results)
    #DEBUG. macstate is not forwarded here









