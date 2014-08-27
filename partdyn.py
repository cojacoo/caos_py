# coding=utf-8

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#particle dynamics
#macropore mc.soilmatrix interaction
def cellgrid(lat,z,mc):
    import numpy as np
    rw=np.floor(z/mc.mgrid.vertfac.values)
    cl=np.floor(lat/mc.mgrid.latfac.values)
    cell=rw*mc.mgrid.latgrid.values + cl
    #if len(np.unique(cell))!=mc.mgrid.cells.values:
    #    print 'runaway particles'
    #if any(np.isnan(cell)):
    #    A=np.where(np.isnan(cell))
    #    print 'R/C',rw[A],cl[A]
    #    print 'z/lat',z[A],lat[A]

    cell[cell<0.]=0.
    cell[cell>=mc.mgrid.cells.values[0]]=mc.mgrid.cells.values[0]-1.

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
    #DEBUG:
    cells[cells<0]=0
    trycount = np.bincount(cells)
    trycount=trycount-1 #reduce again by added particles
    npart[np.unravel_index(np.arange(mc.mgrid.cells.astype(np.int64)),(mc.mgrid.vertgrid.values.astype(np.int64),mc.mgrid.latgrid.values.astype(np.int64)))] = trycount
    npart_s=spn.filters.median_filter(npart,size=mc.smooth)
    #do not smooth at macropores centroids
    npart_s[np.unravel_index(mc.maccells,(mc.mgrid.vertgrid.values.astype(np.int64),mc.mgrid.latgrid.values.astype(np.int64)))]=npart[np.unravel_index(mc.maccells,(mc.mgrid.vertgrid.values.astype(np.int64),mc.mgrid.latgrid.values.astype(np.int64)))]
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
        reffac=np.floor(lat[lat>mc.mgrid.width[0]]/mc.mgrid.width[0])
        lat[lat>mc.mgrid.width[0]]=lat[lat>mc.mgrid.width[0]]-mc.mgrid.width[0]*reffac
    if any(lat<0.0):
        reffac=np.floor(np.abs(lat[lat<0.0]/mc.mgrid.width[0]))
        lat[lat<0.0]=lat[lat<0.0]+mc.mgrid.width[0]*reffac
    #topbound - DEBUG: set zero for now, may interfere with infilt leftover idea
    if any(z>-0.00001):
        z[z>-0.00001]=-0.00001
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

def assignadvect(no,mc,dummy=-99,realcrosssec=True):
    import numpy as np
    if any(dummy==-99): #fast lane not given
        dummy=np.random.randint(len(mc.t_cdf_fast.T), size=no)

    dummx=np.random.rand(no)
    cum_cdf=np.array(mc.t_cdf_fast.cumsum(axis=0))
    l_cdf=cum_cdf.shape[0]
    idx=abs(np.floor(cum_cdf[:,dummy]-dummx.repeat(l_cdf).reshape(cum_cdf[:,dummy].T.shape).T).sum(axis=0)).astype(np.int)
    if realcrosssec:
        adv=mc.a_velocity_real[idx]
    else:
        adv=mc.a_velocity[idx]
        
    return adv

def mac_advection(particles,mc,thS,dt,clog_switch=True):
    '''Calculate Advection in Macropore
       Advective particle movement in macropore with retardation of the advective momentum through drag at interface,
       plus check for each macropore's capacity and possible overload.
       Updated version 20/05/2014
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
    for maccol in np.arange(len(mc.maccols)):
        macstate=[]
        if not particles.loc[particles.flag==(maccol+1)].empty:
            #project advection step
            midx=np.where(particles.flag==(maccol+1))[0]
            #macropore is divided in grid of particle diameter steps
            s_proj=particles.loc[particles.flag==(maccol+1),'advect'].values*dt
            z_proj=particles.loc[particles.flag==(maccol+1),'z'].values+s_proj

            nodrain=(z_proj>=mc.soildepth)
            if any(-nodrain):
                z_proj[-nodrain]=mc.soildepth
            
            #id and filling in macropore grid
            mxgridcell=np.floor(mc.md_macdepth[maccol]/mc.particleD).astype(np.int64)[0]
            def macfil(z,z_p,mxgridcell):
                p_mzid=np.floor(-z/mc.particleD).astype(int)
                pp_mzid=np.floor(-z_p/mc.particleD).astype(int)
                Ax=(p_mzid>mxgridcell)
                Bx=(pp_mzid>mxgridcell)
                exfilt=(Ax | Bx)
                p_mzid[p_mzid>mxgridcell]=mxgridcell
                pp_mzid[pp_mzid>mxgridcell]=mxgridcell
            
                pp_mzid_plus1=np.append(pp_mzid,np.arange(mxgridcell))
                mfilling=np.bincount(pp_mzid_plus1)-1
                return [p_mzid.astype(np.int), pp_mzid.astype(np.int), mfilling]

            [particles_mzid,proj_mzid,mfilling]=macfil(particles.loc[particles.flag==(maccol+1),'z'].values,z_proj,mxgridcell)

            #id in soil grid
            particles_zid=np.floor(-particles.loc[particles.flag==(maccol+1),'z'].values/mc.mgrid.vertfac.values).astype(int)
            proj_zid=np.floor(-z_proj/mc.mgrid.vertfac.values).astype(int)

            #check free slots in macropore along path
            def contactcount(idx,idy):
                return np.count_nonzero(mfilling[idx:idy]==0)
            vcontactcount=np.vectorize(contactcount)
            dragweight=vcontactcount(particles_mzid,proj_mzid)

            #experienced psi at start point (assume to be valid for total stretch)
            idx=particles.loc[particles.flag==(maccol+1),'cell'].values
            psi=mc.psi[thS[idx],mc.soilgrid.ravel()[idx]-1]
            #E_psi=np.mean((psi[particles_zid],psi[proj_id]),axis=0)*mc.particlemass*const.g
            contactfac=np.ones(len(particles_zid),dtype=np.float64)
            ia=dragweight>0.
            ib=(proj_zid-particles_zid)>0.
            if any(ia & ib):
                contactfac[ia & ib]=1.-(dragweight[ia & ib]/(proj_zid[ia & ib]-particles_zid[ia & ib]))

            # #perform advection
            psib=psi*9806.65
            pm=mc.particlemass/1000. #particle mass conversion into kg
            #pV=mc.particlemass/dr.waterdensity(np.array(20),np.array(-9999.))
            pV=mc.particleV
            ts=mc.soilmatrix.ts.values[mc.soilgrid.ravel()[idx]-1]
            thS_projraw=(thS[idx]*0.01*ts*contactfac*abs(s_proj)*mc.particleA + mc.particleV)/(contactfac*mc.particleA*abs(s_proj)*ts)
            thS_projraw[thS_projraw>1.]=1.
            thS_proj=np.round(100.*thS_projraw).astype(int)
            th_x=(mc.soilmatrix.ts.values-mc.soilmatrix.tr.values)[mc.soilgrid.ravel()[idx]-1]
            psi_proj=mc.psi[thS_proj,mc.soilgrid.ravel()[idx]-1]
            s_red=pV*(dr.waterdensity(np.array(20),np.array(-9999.))*0.001*particles.loc[particles.flag==(maccol+1),'advect'].values**2 - psi_proj) / ((psi_proj*thS_proj-psi*thS[idx])*th_x*mc.particleA)
            particles_znew=particles.loc[particles.flag==(maccol+1),'z'].values-s_red
            #runaway particles at lower bound
            nodrain=(particles_znew<=mc.soildepth)
            particles_znew[particles_znew<=mc.soildepth]=mc.soildepth+0.00001

            #update index
            [particles_mzid,proj_mzid,mfilling]=macfil(particles.loc[particles.flag==(maccol+1),'z'].values,particles_znew,mxgridcell)

            #find macropore increment of particles
            def findincr(x):
                idx=np.where(mc.md_depth>-x)[0]-1
                if len(idx)>1:
                    idxx=idx[0]
                else:
                    idxx=len(mc.md_depth)-1
                return idxx
            vfindincr = np.vectorize(findincr)
            macincr=vfindincr(-z_proj)

            #check clogging of macropore
            if (clog_switch==True):
                #cut advection at clogging
                
                #check macropore capacity (clogging)
                #check if clogging occurs and where
                def clogpos(idx,idy,idz):
                    x=-1
                    if (idx!=idy):
                        t = mfilling[idx:idy]-mc.maccap[maccol,idz]
                        if any(t<0.):
                            x=idx+np.nonzero(t<0.)[0][0] #mzid of clogged cell
                        else:
                            x=idy
                    return x
                vclogpos=np.vectorize(clogpos)
                
                clog=vclogpos(particles_mzid.astype(np.int),proj_mzid.astype(np.int),macincr.astype(np.int))

                #cut advection at clogging
                cid=(clog>=0)
                if any(cid):
                    #update z_proj to center of last free cell before clog
                    z_proj[cid]=-mc.particleD*(clog[cid]-0.5)
                    particles_znew[cid]=np.amax([particles_znew[cid],z_proj[cid]],axis=0)

            #assign new z into data frame:
            particles.loc[particles.flag==(maccol+1),'z']=particles_znew
            
            particles_zid=np.floor(particles_znew/mc.mgrid.vertfac.values).astype(int)
            particles_zid[particles_zid>=mc.mgrid.vertgrid.values[0]]=mc.mgrid.vertgrid.values[0]-1
            #assume all particles at pore wall > film flow
            #assume positive phi if filled by more than one
            idx=np.where(mfilling>1.)[0]
            idpres=np.where(np.in1d(particles_zid,idx,True))[0]
            #soil grid cellid
            cellid=(particles_zid*mc.mgrid.latgrid.values+np.floor(mc.md_pos[maccol]/mc.mgrid.latfac.values).astype(int)).astype(int)
            #psi in matrix
            mxpsi=mc.psi[thS[cellid],mc.soilgrid.ravel()[cellid]-1]
            gradxi=np.tanh(-mxpsi/20.)+np.random.rand(len(particles_znew)) #skewes to zero at saturation
            gradxi[idpres]=np.sqrt(gradxi[idpres]) #elevate gradxi for more populated cells

            D=mc.D[99,mc.soilgrid.ravel()[cellid]-1]
            step_proj=(gradxi*((6*D*dt)**0.5))
            exfilt=(step_proj>=mc.particleD/2.)
            #exfiltration if projected step is larger than half the particle diameter
            #DEBUG: check if that ever appears
            if any(exfilt):
                idy=midx[exfilt]
                particles.flag.iloc[idy]=0
                particles.lat.iloc[idy]=mc.md_pos[maccol]+mc.md_contact[maccol,macincr[exfilt]]*(np.random.rand(sum(exfilt))-0.5)
                
            particles.loc[particles.flag==(maccol+1),'cell']=cellgrid(particles.loc[particles.flag==(maccol+1),'lat'].values,particles.loc[particles.flag==(maccol+1),'z'].values,mc).astype(np.int64)
            
            #handle draining particles if any
            if any(-nodrain):
                particles.flag.iloc[midx[-nodrain]]=len(mc.maccols)+1
                particles.z.iloc[midx[-nodrain]]=mc.soildepth-0.0001

    return(particles)


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
    #psi_id=((np.log10(-mc.psi[thS,mc.soilgrid.ravel()-1].astype(np.float64)+mc.zgrid.ravel()-mc.mgrid.depth.values)+2)*10.).astype(int)
    #u=mc.p_ku[psi_id,mc.soilgrid.ravel()-1]
    u=mc.ku[thS,mc.soilgrid.ravel()-1]
    #u=mc.soilmatrix.ks.values[mc.soilgrid.ravel()-1] #use ksat for advection
    #D=u/mc.cH2O[thS,mc.soilgrid.ravel()-1]
    D=mc.D[thS,mc.soilgrid.ravel()-1]

    vert_sproj=dt*u[particles.cell.values.astype(np.int)] + (xi[:,0]*((2*D[particles.cell.values.astype(np.int)]*dt)**0.5))
    lat_sproj=(xi[:,1]*((2*D[particles.cell.values.astype(np.int)]*dt)**0.5))

    if (uffink_corr==True):
        #Itô Scheme after Uffink 1990 and Kitanidis 1994
        #DEBUG: linalg.norm needed too long.
        #dx=map(np.linalg.norm, step_proj)
        dx=np.sqrt(vert_sproj**2+lat_sproj**2)
        # project step and updated state
        # new positions
        lat_proj=particles.lat.values+lat_sproj
        z_proj=particles.z.values-vert_sproj
        [lat_proj,z_proj,nodrain]=boundcheck(lat_proj,z_proj,mc)

        [thS,npart]=gridupdate_thS(lat_proj,z_proj,mc) #DEBUG: externalise smooth parameter)
        #cut thS>1 and thS<0 for projection
        thSx=thS
        thSx[thSx>100.]=100.
        thSx[thSx<0.]=0.
        #psi_id=((np.log10(-mc.psi[thSx.astype(int),mc.soilgrid.ravel()-1].astype(np.float64)+mc.zgrid.ravel()-mc.mgrid.depth.values)+2)*10.).astype(int)
        u_proj=mc.ku[thSx,mc.soilgrid.ravel()-1]
        #D_proj=u_proj/mc.cH2O[thSx,mc.soilgrid.ravel()-1]
        D_proj=mc.D[thSx,mc.soilgrid.ravel()-1]

        corrD=np.abs(D[particles.cell.values.astype(np.int)]-D_proj[particles.cell.values.astype(np.int)])/dx
        corru=np.sqrt(u[particles.cell.values.astype(np.int)]*u_proj[particles.cell.values.astype(np.int)])
        #corru=u[particles.cell.values.astype(np.int)]
        
        # corrected step
        vert_sproj=(corru+corrD)*dt + (xi[:,0]*((2*D[particles.cell.values.astype(np.int)]*dt)**0.5))
        lat_sproj=corrD*dt + (xi[:,1]*((2*D[particles.cell.values.astype(np.int)]*dt)**0.5))

    # new positions
    lat_new=particles.lat.values+lat_sproj
    z_new=particles.z.values-vert_sproj

    [lat_new,z_new,nodrain]=boundcheck(lat_new,z_new,mc)

    # saturation check
    [thS,npart]=gridupdate_thS(lat_new[nodrain],z_new[nodrain],mc) #DEBUG: externalise smooth parameter

    phi_mx=mc.psi[thS,mc.soilgrid.ravel()-1]+mc.mxdepth_cr

    particles['z']=z_new
    particles['lat']=lat_new
    particles['cell']=cellgrid(lat_new,z_new,mc).astype(np.int64)
    if any(-nodrain):
        particles.loc[-nodrain,'flag']=len(mc.maccols)+1

    return [particles,thS,npart,phi_mx]

def part_diffusion_split(particles,npart,thS,mc,dt,uffink_corr=True,splitfac=5):
    '''Calculate Diffusive Particle Movement
       Based on state in grid use diffusivity as foundation of 2D random walk.
       Project step and check for boundary conditions and further restrictions.
       Update particle positions.
    '''
    N_tot=len(particles.z) #number of particles

    #splitsample particles randomly
    #splitfac=5
    splitref=np.floor(N_tot/splitfac).astype(int)
    sampleset=np.random.permutation(N_tot)
    for subsample in np.arange((splitfac-1)):
        samplenow=sampleset[(subsample*splitref):((subsample+1)*splitref-1)]

        N=len(samplenow) #number of particles handled

        # 1D Random Walk function with additional correction term for
        # non-static diffusion after Uffink 1990 p.15 & p.24ff and Kitanidis 1994
        xi=np.random.rand(N,2)*2.-1.
        u=mc.ku[thS,mc.soilgrid.ravel()-1]/mc.theta[thS,mc.soilgrid.ravel()-1]
        D=u*((mc.psi[thS,mc.soilgrid.ravel()-1]-mc.psi[thS-1,mc.soilgrid.ravel()-1])/((mc.theta[thS,mc.soilgrid.ravel()-1]-mc.theta[thS-1,mc.soilgrid.ravel()-1])*9810.))
        #D=mc.D[thS,mc.soilgrid.ravel()-1]*mc.theta[thS,mc.soilgrid.ravel()-1]**2

        vert_sproj=dt*u[particles.cell.values[samplenow].astype(np.int)] + (xi[:,0]*((2*D[particles.cell.values[samplenow].astype(np.int)]*dt)**0.5))
        lat_sproj=(xi[:,1]*((2*D[particles.cell.values[samplenow].astype(np.int)]*dt)**0.5))

        if (uffink_corr==True):
            #Itô Scheme after Uffink 1990 and Kitanidis 1994 for vertical step
            #modified Stratonovich Scheme after Kitanidis 1994 for lateral step
            dx=np.sqrt(vert_sproj**2+lat_sproj**2)
            # project step and updated state
            # new positions
            lat_proj=particles.lat.values
            z_proj=particles.z.values
            lat_proj[samplenow]=particles.lat.values[samplenow]+lat_sproj
            z_proj[samplenow]=particles.z.values[samplenow]-vert_sproj
            [lat_proj,z_proj,nodrain]=boundcheck(lat_proj,z_proj,mc)
            [thSx,npartx]=gridupdate_thS(lat_proj,z_proj,mc) #DEBUG: externalise smooth parameter)
            #cut thSx>1 and thSx<0 for projection - in case they appear
            thSx[thSx>100]=100
            thSx[thSx<0]=0
            u_proj=mc.ku[thSx,mc.soilgrid.ravel()-1]/mc.theta[thSx,mc.soilgrid.ravel()-1]
            #D_proj=mc.D[thSx,mc.soilgrid.ravel()-1]*mc.theta[thSx,mc.soilgrid.ravel()-1]**2
            #D_proj=u_proj/(mc.cH2O[thSx,mc.soilgrid.ravel()-1]*mc.theta[thSx,mc.soilgrid.ravel()-1])
            D_proj=u_proj*((mc.psi[thSx,mc.soilgrid.ravel()-1]-mc.psi[thSx-1,mc.soilgrid.ravel()-1])/((mc.theta[thSx,mc.soilgrid.ravel()-1]-mc.theta[thSx-1,mc.soilgrid.ravel()-1])*9810.))

            corrD=np.abs(D_proj[particles.cell.values[samplenow].astype(np.int)]-D[particles.cell.values[samplenow].astype(np.int)])/dx
            corrD[dx==0.]=0.
            D_mean=np.sqrt(D_proj[particles.cell.values[samplenow].astype(np.int)]*D[particles.cell.values[samplenow].astype(np.int)])
            corru=np.sqrt(u[particles.cell.values[samplenow].astype(np.int)]*u_proj[particles.cell.values[samplenow].astype(np.int)])
            #corrD[corrD>corru]=corru[corrD>corru]
            # corrected step
            vert_sproj=(corru-corrD)*dt + (xi[:,0]*((2*D[particles.cell.values[samplenow].astype(np.int)]*dt)**0.5))
            lat_sproj=(xi[:,1]*((2*D_mean*dt)**0.5))

        # new positions
        lat_new=particles.lat.values
        z_new=particles.z.values
        lat_new[samplenow]=particles.lat.values[samplenow]+lat_sproj
        z_new[samplenow]=particles.z.values[samplenow]-vert_sproj
        [lat_new,z_new,nodrain]=boundcheck(lat_new,z_new,mc)

        # saturation check
        [thS,npart]=gridupdate_thS(lat_new[nodrain],z_new[nodrain],mc) #DEBUG: externalise smooth parameter

        phi_mx=mc.psi[thS,mc.soilgrid.ravel()-1]+mc.mxdepth_cr

        particles['z']=z_new
        particles['lat']=lat_new
        particles['cell']=cellgrid(lat_new,z_new,mc).astype(np.int64)
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


def plotparticles(runname,t,particles,npart,mc):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.gridspec as gridspec
    
    f_name=''.join([runname,str(t),'.pdf'])
    pdf_pages = PdfPages(f_name)
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
    ax2.text(0.1, 0.8, 'Particles @ t='+str(t)+'s', fontsize=20)
    
    ax3 = plt.subplot(gs[2])
    plt.imshow(sp.ndimage.filters.median_filter(npart,size=mc.smooth),vmin=1, vmax=mc.part_sizefac, cmap='jet')
    #plt.imshow(npart)
    plt.colorbar()
    plt.xlabel('Width [cells a 5 mm]')
    plt.ylabel('Depth [cells a 5 mm]')
    plt.title('Particle Density')
    plt.tight_layout()

    ax4 = plt.subplot(gs[3])
    ax41 = ax4.twiny()
    z1=np.append(particles.loc[((particles.age>200.)),'z'].values,mc.onepartpercell[1][:mc.mgrid.vertgrid.values.astype(int)])
    advect_dummy=np.bincount(np.round(-100.0*z1).astype(np.int))-1
    old_dummy=np.bincount(np.round(-100.0*particles.loc[((particles.age<200.)),'z'].values).astype(np.int))
    ax4.plot(advect_dummy,(np.arange(0,len(advect_dummy))/-100.),'b-',label='new particles')
    ax41.plot(old_dummy,(np.arange(0,len(old_dummy))/-100.),'g-',label='old particles')
    ax41.set_xlabel('Old Particle Count', color='g')
    ax4.set_xlabel('New Particle Count', color='b')
    ax4.set_ylabel('Depth [m]')
    #ax4.set_title('Number of Particles')
    ax4.set_ylim([mc.mgrid.depth.values,0.])
    ax4.set_xlim([0.,np.max(advect_dummy)])
    ax41.set_xlim([0.,np.max(old_dummy[1:])])
    ax41.set_ylim([mc.mgrid.depth.values,0.])
    #handles1, labels1 = ax4.get_legend_handles_labels() 
    #handles2, labels2 = ax41.get_legend_handles_labels() 
    #ax4.legend(handles1+handles2, labels1+labels2,loc=4)
    #    ax41.legend(loc=4)
    
    plt.show()
    pdf_pages.savefig(fig)
    pdf_pages.close()
    print ''.join(['wrote graphic to ',f_name])






