import numpy as np
from matplotlib import pyplot as plt
from pyGeneralRoutines.span import span
from pyGeneralRoutines.fn_add_subfix import fn_add_subfix
from pyProfile.profile import *
from pyProfile.psd import *
from pyProfile.power_law import *
import pdb

"""2014/12/29 from slumpingPSD.ipynb
Module containing routines for slumping of profiles.
xg,yg,xm,ym are expected to be defined globally.
Plotting routines assume that globals pslumped, ci, ct have values calculated by slump routine.

pslumpded, ci, ct
"""

# Beam Routines

#profiles are managed in um, X in mm.
d=0.4 #thickness mm
rho=2.38e-3 #g/cm^3 -> g/mm^3
g=9800 #mm/s^2
E=73.6e9   #Eagle glass 73.6 GPa ->  10^6 g/mm/s**2 
nu=0.23   #poisson ratio

def beamtime_um(x,alpha):
    """Return the beam profile in um at effective time alpha (in units of gravitational sag)
    e.g. alpha0=1 at t=0."""
    L=span(x,size=1)
    x=x-x[0]
    gravsag=-rho*g/2/d**2/E*x**2*(L-x)**2   #beam with fixed ends
    return alpha*gravsag*1000


def fit_beamsag(x,y,range=None):
    """Given a profile with fixed ends, returns the sagged profile that minimizes least
    square differences."""
    L=span(x,size=1)
    x=x-x[0]
    
    if not(range is None):
        i=np.where((logx>range[0]) & (logx< range[1]))
        x=x[i]
        y=y[i]
    gravsag=-x**2*(L-x)**2   #beam with fixed ends     
    def scale(s):
        return np.sqrt(np.sum((s*gravsag-y)**2))
    minimize_scalar(scale)
    return alpha*gravsag*1000 

"""
def fit_power_law(x,y,range=None):
    '''fit a PSD according to a model y=K_N*x**N
    where x is in mm-1 and y in mm*3'''
    logx = np.log10(x)
    logy = np.log10(y)
    #logyerr = yerr / ydata
    if not(range is None):
        i=np.where((logx>range[0]) & (logx< range[1]))
        logx=logx[i]
        logy=logy[i]
    out =np.polyfit(logx,logy,1)  
"""
    
def find_contact(x,delta):
    """calculate contact position and time for a flat beam and a profile delta in micron.
    X of Profile in mm must start by 0 and must be leveled.
    2014/12/28 return 0,None if x is made of two points only."""
    
    if len(x)==2:
        return 0,None
    delta=delta-line(x,delta)
    assert delta[0]==delta[-1]==0
    Pl=beamtime_um(x,1)
    i=np.argmin((delta/Pl)[1:-1])+1 #index of contact point fraction must be >=1 everywhere
    alpha0=(delta/Pl)[i]#instant of contact in arbitrary units of time
    if alpha0<1: print('Warning! alpha0<1. It touches for gravitational sag.')
    
    return alpha0,i


#slumping routines
#remove plot parameters from slump, but add them in corresponding _slump routine of SlumpingModelProfile class

def slump(x,ym,yg,nsteps=5,toll=1e-5,plot=True,outfolder=None,delay=0.01):
    # input : d, yg, 
    # parameters: nsteps, toll, delay 
    # flag: plot, outfolder
    # output : contactTimes, pslumped

    # delta difference between glass and mandrel.
    # yg glass profile
    #pdb.set_trace()
    
    delta=ym-yg
    contactTimes=x*0   #array with all contact times for each point
    contactTimes[1:-1]=np.NAN
    pp=delta*0  #array partial profile (profile variation during step)
    pslumped=[yg[:]]  #[delta*0]  #list of profiles of the beam slumped at each step
    stepTime=[]
    stepPoint=[]
    
    #determine first contact 
    alpha=0   #effective time (slumping amount in units of gravitational sag)
    dalpha,inext=find_contact(x,delta)
    contactTimes[inext]=dalpha
    
    for j in range(nsteps):
        
        #alpha and inext are set to time and index of next contact point
        #Get the next segment to slump.
        inext=np.nanargmin(np.where(contactTimes>alpha,contactTimes,np.NAN))
        dalpha=contactTimes[inext]-alpha  #time difference in this step
        alpha=contactTimes[inext]  #time at next contact point
        stepPoint.append(inext)
        stepTime.append(alpha)
        
        # Slump for a time dalpha on all segments.
        ## Segments are defined by icp, indices of contact points at present time. 
        ## all contact times are stored, be careful to use only the ones in the past.
        icp=np.where(np.invert(np.isnan(contactTimes)))[0]  
        icp=icp[np.where(contactTimes[icp]<alpha)]
        ## iterate over intervals.
        for i1,i2 in zip(icp[:-1],icp[1:]): 
            pp[i1:i2+1]=beamtime_um(x[i1:i2+1],dalpha)
        
        #update profile and residuals
        pslumped.append(pslumped[-1]+pp) #sum last profile to the variation
        delta=delta-pp
        delta[np.where(np.abs(delta)<toll)]=0   #points closer than toll are considered in contact
        if len(np.where((np.abs(delta)<toll)&(delta!=0))[0])!=0:
            print(j,np.where((np.abs(delta)<toll)&(delta!=0)))
        
        #calculate touch times for the 2 new subsegments   
        icpnext=np.where((icp-inext)<0)[0][-1]
        i1,i2=icp[icpnext:icpnext+2]
        tt=[find_contact(x[i1:inext+1],delta[i1:inext+1]),find_contact(x[inext:i2+1],delta[inext:i2+1])]
        tt=[t for t in tt if not(t is None)] 
        ct,ci=[t[0] for t in tt],[t[1] for t in tt]
        assert len(ct) == len (ci) ==2
        ## need to add offsets for first and second interval
        for cci, cct, ioffset in zip(ci, ct,[i1,inext]):
            if not(cci is None):
                contactTimes[np.array(cci+ioffset,dtype=int)]=cct+alpha

    return  pslumped,np.array(stepTime),np.array(stepPoint)

def interpolate_profile(pslumped,ct,a0):
    """returns the profile at an arbitrary time assume all global input variables.
    alpha is scalar and larger than ct[0]."""
    nstep=np.where((ct-a0)>0)[0][0] #index of first step above a0
    frac=(a0-ct[nstep-1])/(ct[nstep]-ct[nstep-1]) #ct is contact time of NEXT point 
    result=pslumped[nstep]+frac*(pslumped[nstep+1]-pslumped[nstep])
    return result

def plot_steps(pslumped,ct,ci,steps,x,ym=None,yg=None,outfolder=None,delay=0.05):
    """Plot a step described by pslumped, ct,ci. 
    x,yg,ym profiles of glass and mandrel on common x.
    TODO: adapt to any time."""
    
    for ns in steps:
        assert ns>0
        profile=pslumped[ns] #glass profile at step ns  ##change these for fractional steps
        pp=pslumped[ns+1]-profile  #profile slumped in step
        delta=ym-profile   #residuals at the end
        ip=ci[ns-1] #index of last point that came in contact
        ipn=ci[ns] #index of next point to come in contact
        alpha=ct[ns-1] #index at time of profile   ##
        dalpha=ct[ns]-ct[ns-1]    ##
        
        plt.clf()

        plt.subplot(311)
        plt.title(r'contacts #: %i,$\alpha=$ %s, $\Delta\alpha=$%s'%(ns,alpha,dalpha))

        plt.plot(x,ym,'r',label='Mandrel',lw=3)
        if not(yg is None):plt.plot(x,yg,'g:',label='Glass Initial')
        plt.plot(x,profile,'b',label='Glass')
        plt.plot(x[ip:ip+1],ym[ip:ip+1],'oy',label='Last',markersize=8)
        plt.plot(x[ipn:ipn+1],ym[ipn:ipn+1],'sg',label='Next',markersize=8)
        plt.xlabel('X(mm)')
        plt.ylabel('Z profile($\mu$m)')
        plt.grid(1)
        plt.legend(loc=0)

        plt.subplot(312)  #residuals
        plt.grid(1)
        plt.xlabel('X(mm)')
        plt.ylabel('Z variation in timestep($\mu$m)')
        plt.plot(x,delta,'c',label='residuals',lw=3)
        plt.plot(x,pp,c='black',label='variation')
        plt.plot(x[ip:ip+1],delta[ip:ip+1],'oy',label='Last',markersize=8)
        plt.plot(x[ipn:ipn+1],delta[ipn:ipn+1],'sg',label='Next',markersize=8)
        plt.show()
        plt.legend(loc=0)

        plt.subplot(313) 
        plt.grid(1)
        plt.xlabel('X(mm)')
        plt.ylabel(r'Time for contact $\alpha_0$ (grav. sag.)')
        plt.plot(x,delta/pp*dalpha)   
        #last has no sense, it touched and pp is 0
        #plt.plot(x[ip:ip+1],(delta/pp)[ip:ip+1],'oy',label='Last',markersize=8)
        plt.plot(x[ipn:ipn+1],(delta/pp*dalpha)[ipn:ipn+1],'sg',label='Next',markersize=8)
        plt.semilogy()
        plt.legend(loc=0)
        if outfolder:
            plt.savefig(os.path.join(outfolder,'%03i'%ns+'.png'))
        plt.pause(delay)

        
def plot_steps_PSD(pslumped,ct,ci,steps,x=None,ym=None,yg=None,outfolder=None,delay=0.05):
    """Plot PSD of a step described by pslumped, ct,ci. 
    x,yg,ym profiles of glass and mandrel on common x.
    TODO: adapt to any time."""
    
    offset=0.1  #offset for plotting of marker
    for ns in steps:
        profile=pslumped[ns] #glass profile at step ns
        alpha=ct[ns-1] #index at time of profile
        dalpha=ct[ns]-ct[ns-1]
        ip=ci[ns-1] #index of last point that came in contact
        ipn=ci[ns] #index of next point to come in contact

        f1,gpsd0=psd(x,yg)
        f2,mpsd=psd(x,ym)
        f3,gpsdf=psd(x,pslumped[-1])
        assert np.all(f1==f2)
        assert np.all(f1==f3)
        
        plt.clf()
        plt.subplot(211)
        plt.title(r'contacts #: %i,$\alpha=$ %s, $\Delta\alpha=$%s'%(ns,alpha,dalpha))
        plt.plot(x,ym,'r',label='Mandrel',lw=3)
        plt.plot(x,yg,'g:',label='Glass Initial')
        plt.plot(x,profile,'b',label='Glass')
        plt.plot(x[ip:ip+1],ym[ip:ip+1],'oy',label='Last',markersize=8)
        plt.plot(x[ipn:ipn+1],ym[ipn:ipn+1],'sg',label='Next',markersize=8)
        plt.xlabel('X(mm)')
        plt.ylabel('Z profile($\mu$m)')
        plt.grid(1)
        plt.legend(loc=0)
        plt.subplot(212)
        plt.title('PSD')
        plt.xlabel('Spatial freq. (mm$^-1$)')
        plt.ylabel('PSD (mm$^3$)')
        plt.plot(f1,gpsd0,label='initial glass')
        plt.plot(f1,mpsd,label='mandrel')
        plt.plot(f1,gpsdf,label='final')
        plt.plot(*(psd(x,profile)),label='Glass')
        plt.loglog()
        plt.grid()
        plt.legend(loc=0)
        if outfolder:
            plt.savefig(os.path.join(outfolder,'PSD_%03i'%ns+'.png'))
        plt.pause(delay)

def load_test_profiles():
    #2015/02/25 modified on the base of slumpingPSD v4
    gfile='test/16_op2s06sub_yscan_20140707.txt'
    mfile='test/01_OP2_yscan_20140425.txt'
    cgfile='test/02_mandrel3_yscan_20140706.txt'
    cmfile='test/04_mandrel3_yscan_20131223.txt'
    xg,yg=np.genfromtxt(gfile,delimiter=',',unpack=1)
    xm,ym=np.genfromtxt(mfile,delimiter=',',unpack=1)    
    xcg,ycg=np.genfromtxt(cgfile,delimiter=',',unpack=1)
    xcm,ycm=np.genfromtxt(cmfile,delimiter=',',unpack=1)
    xg_raw=xg.copy()
    
    ##glass
    plt.figure(1)
    plt.clf()
    plt.plot(xg,yg-yg.mean(),'r',label='original')

    #glass profile has peaks, remove them
    for i in range(len(yg[1:])):
        if abs(yg[i]-yg[i-1])>3:
            yg[i]=yg[i-1]

    plt.plot(xg,yg-yg.mean(),'b',label='peak removed')
    yg=yg-ycg
    plt.plot(xg,yg-yg.mean(),'g',label='calibrated')

    #smooth 
    ##note: removing 3.3 leaves 6.6 !)
    yg=movingaverage(yg,66)
    i=np.where((xg>=25)& (xg<=125))
    xg=xg[i]
    yg=yg[i]
    plt.plot(xg,yg-yg.mean(),c='purple',label='smoothed and cut')
    plt.legend(loc=0)

    yg=yg-line(xg,yg)
    yg=-yg
    
    ##mandrel
    #xm has 9 um steps
    plt.figure(2)
    plt.clf()
    plt.plot(xm,ym-ym.max(),label='orig')
    ym=movingaverage(ym,6)
    ym=np.interp(xg_raw,xm,ym)
    xm=xg_raw.copy()
    plt.plot(xm,ym-ym.max(),label='downsam.')
    ym=ym-np.interp(xm,xcm,ycm)
    plt.plot(xm,ym-ym.max(),label='downsam. calib.')
    ym=movingaverage(ym,66)

    #cut profiles between 25 and 125, level.
    i=np.where((xm>=25)& (xm<=125))
    ym=ym[i]
    xm=xm[i]
    ym=ym-line(xm,ym)

    plt.plot(xm,ym-ym.max(),label='smoothed leveled cut')
    plt.legend(loc=0)
    plt.ylim([-18,0])
    
    return xm,ym,xg,yg

if __name__=='___main__':    
    # glass profile:   xg, yg
    # mandrel profile: xm, ym 
    xm,ym,xg,yg=load_test_profiles()
    
    #SLUMPING
    pslumped,ct,ci = slump(xg,ym,yg,nsteps=300,toll=1e-5)
    print('slumped!')
    
    plotIndices=list(range(1,5))

    plt.figure(1)
    for ns in plotIndices:
        plot_step(pslumped,ct,ci,ns,xg,ym,outfolder=outfolder)
    plt.figure(2)
    for ns in plotIndices:
        plot_step_PSD(pslumped,ct,ci,ns,xg,ym,yg,outfolder=outfolder)
	
    a0=100.
    plt.figure(3)
    clf()
    plt.plot(x,pslumped[2])
    plt.plot(x,pslumped[3])
    for i in [86,88,90,95,100,105,108,108.2,108.4,108.6,109]:
        plt.plot(x,interpolate_profile(i))