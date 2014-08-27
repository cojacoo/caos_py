# MACROPORE INITIALISATION
macbase['area']=macbase.Macropore_Share*mc.refarea
rem_area=[]
np_tot=[]
lcircum=[]
mr=[]
mxd=[]

for i in np.arange(len(macbase.area)):
    Ax=0
    r_r=[]
    if macbase.area[i] > 0:
        while Ax < macbase.area[i]:
            r=np.random.rand(1)
        
            if r<=0.5:
                r_r.append(macbase.r_Min[i]+r*2*(macbase.r_Mean[i]-macbase.r_Min[i]))
            else:
                r_r.append(macbase.r_Min[i]+r*(macbase.r_Max[i]-macbase.r_Min[i]))
        
            Ax += const.pi*(r_r[-0]**2)
            
        rem_area.append((Ax/len(r_r))[0])
        np_tot.append(len(r_r))
        lcircum.append(((pd.DataFrame(r_r)*const.pi*2.).mean())[0])
        mr.append((pd.DataFrame(r_r).mean())[0])
        mxd.append(np.sqrt((mc.refarea/len(r_r))/const.pi))
        
    else:
        rem_area.append(0)
        np_tot.append(0)
        lcircum.append(0)
        mr.append(0)
        mxd.append(0)
        
rem_share=pd.DataFrame(rem_area)/mc.refarea
            
macbase['mxd']=pd.DataFrame(mxd)
macbase['rem_area']=pd.DataFrame(rem_area)
macbase['rem_share']=rem_share
macbase['np_tot']=np_tot
macbase['lcircum']=lcircum
macbase['mr']=mr      


#macbase

#setup macropore domain

#  dummi=int(minloc(mac_ini%rem_share,mask=mac_ini%share>0.0)) !get idx of pore with largest mdist to span domain
#  i=dummi(1)
#  
#  macref=mac_ini(i)%rem_share
#  mxd=mac_ini(i)%mdist
#  mac_ini(i)%reffac=1
#  
#  do j=1, size(mac_ini)
#    mac_ini(j)%reffac=ceiling(mac_ini(i)%rem_area/mac_ini(j)%rem_area)
#    if (j>1 .AND. mac_ini(j)%class==mac_ini(j-1)%class) mac_ini(j)%reffac=min(mac_ini(j)%reffac,mac_ini(j-1)%reffac)   ! check for monoton-decreasing in class with depth
#  end do

