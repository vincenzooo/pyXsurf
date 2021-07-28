# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 16:06:48 2016
v1 Saved 2018/08/27 after radical change to raimondi-spiga routine
without saving previous version.
It might be worth to recover it from backups.

@author: Vincenzo Cotroneo
@email: vcotroneo@cfa.harvard.edu
"""
from dataIO.span import span
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


## profile creation

def line(x,y=None):
    """return line through end points of x,y.
	x and y are vectors, can be of different length (e.g. y can be a 2-elements vector).
    If only one vector is provided, it is assumend as equally spaced points.
"""
    x=np.array(x)
    if y is None:
        y=x
        x=np.arange(len(y))
    y=np.array(y)
    
    #account for nan
    sel=~np.isnan(y)
    if sel.any():
        y0=[y[sel][0],y[sel][-1]]
        if len(x)==len(y):
            x0=x[sel]
        else:
            x0=x
        L=span(x0,size=1)
        return (x-x0[0])*(y0[-1]-y0[0])/L+y0[0]
    else:
        return y
        
def make_signal(amp,L,N,nwaves,ystartend=(0,0),noise=0.,minus_one=False):
    """Build a signal of length L and number of points N, as a sum of a sinusoid, a line and a noise. minus_one remove last point (just a convenience e.g. 
    for a periodic profile)."""
    
    x=np.arange(N,dtype=float)/(N-1)*L
    l=line(x,ystartend)
    y=amp*np.sin(2*np.pi*x/L*nwaves) + noise*np.random.random(N)+l
    if minus_one:
        x,y=x[:-1],y[-1]
    return x,y

def make_circle(x,c,r,sign=1):
    """plot positive part if sign is positive, negative if negative."""
	
    y=np.sqrt(R**2-(x-c[0])**2)+c[1]
    return x,y*np.sign(sign)
    

#profile fitting

def polyfit_profile (x,y=None,degree=1):
    """return polynomial of degree that fits the profile.
"""
    if y is None:
        y=x
        x=np.arange(len(y))
    
    sel=~np.isnan(y)
    if sel.any():
        y0=y[sel]
        x0=x[sel]
        coeff=np.polyfit(x0,y0,degree)
        return np.polyval(coeff,x)
    else:
        return y    

        
##PROFILE I/O
def save_profile(filename,x,y,**kwargs):
    
    np.savetxt(filename,np.vstack([x,y]).T,**kwargs)
        
#PROFILE OPERATIONS        
def level_profile (x,y=None,degree=1):
    """return profile after removal of a polynomial component.
"""
    if y is None:
        y=x
        x=np.arange(len(y))
    
    return x,y-polyfit_profile(x,y,degree) 
        

def remove_profile_outliers(y,nsigma=3,includenan=True):
    """remove outliers from a profile by interpolation.
    y is modified in place. Mask of outliers is returned. """
    
    mask=np.abs(y-np.nanmean(y))<(nsigma*np.nanstd(y)) #points to keep
    if includenan:
        mask[np.isnan(y)]=False
        
    if mask.any()==False:
        return y*np.nan  #all invalid values, transform to nan so it can be
            #detected on call. numeric values cannot be all out of nsigma, unless small.

    if not mask.all():
        x=np.arange(len(y))
        y[mask==False] = np.interp(x[mask==False], x[mask], y[mask])
        
    return mask==False
	

    
"""
def movingaverage(values,window,mode='full'):
    weigths = np.repeat(1.0, window)/window
    #including valid will REQUIRE there to be enough datapoints.
    #for example, if you take out valid, it will start @ point one,
    #not having any prior points, so itll be 1+0+0 = 1 /3 = .3333
    smas = np.convolve(values, weigths, mode)
    return smas # as a numpy array
"""   

def movingaverage(values,window):
    """for now, only odd windows. Full returns"""
    assert int(window/2.)*2==(window-1)
    weigths = np.repeat(1.0, window)/window
    #including valid will REQUIRE there to be enough datapoints.
    #for example, if you take out valid, it will start @ point one,
    #not having any prior points, so itll be 1+0+0 = 1 /3 = .3333
    smas = np.convolve(values, weigths, 'same')
    for i in np.arange(int(window/2.)+1):
        smas[i]=np.mean(values[:2*i+1])
        smas[-i]=np.mean(values[-2*i-1:])
    return smas # as a numpy array

def rebin_profile(x,y,*args,**kwargs):
    ss=stats.binned_statistic(x,y,statistic='mean',*args,**kwargs)
    x2=np.array([(x+y)/2. for x,y in zip(ss[1][:-1],ss[1][1:])])  #centra su punto centrale
    y2=ss[0]
    return x2,y2  

def crop_profile(x,y,xrange=None,*args,**kwargs):
    
    if xrange is None: 
        xrange={None,None}
    if xrange[0] is None:
        xrange[0]=np.min(x)
    if xrange[1] is None:
        xrange[1]=np.max(x)
    
    sel=(x>=xrange[0])&(x<=xrange[1])

    return x[sel],y[sel]  
        
def subtract_profiles(x1,y1,x2,y2,*args,**kwargs):
    return x1,y1-np.interp(x1,x2,y2)  

def calculate_barycenter(x,y):
    """return the x of barycenter using y as weight function."""
    return np.sum(x*y)/len(x)
    
def mirror_profile(xx,yy,center=None,split=False,xout=None):
    """Return the profile x, y mirrored around x=center.
    If split is set return a list of the two profiles >,< 
    as [[xpos,ypos],[xneg,yneg]].
    If xout is passed it is used to interpolate the double
    profiles. If split is not selected and the initial profile
    is not symmetrical can give unexpected results. 
    In case split is selected and center exactly matches one of the input points,
    center point is added to both profiles.
    """
    x=xx.copy()
    y=yy.copy()
    mpos = (x >= center) #mask
    x[~mpos]=2*center-x[~mpos]
    
    if split:
        if center in x:
            mneg=np.logical_not(mpos)
            mneg[x==center]=True
        x0,y0=[x[mneg],y[mneg]],[x[mpos],y[mpos]] #note that x0 and y0 are no more x and y, rather two profiles returned (no sorting)
        if xout is not None:
            x0,y0=[[xout,np.interp(xout,np.sort(xx),yy[xx.argsort()])] for xx,yy in [x0,y0]]
            #x0,y0=[[xout,np.interp(xout,xx,yy)] for xx,yy in [x0,y0]]
    else:
        i0=x.argsort()
        x0=x[i0]
        y0=y[i0]        
        if xout is not None:
            y0=np.interp(xout,x0,y0)
            x0=xout
    
    return x0,y0
        
        
    
def calculate_HEW(x,y,center=None,fraction=0.5):
    """calculate HEW around center from profile x and y by integrating the two sides. If center is None, barycenter is calculated.
    Radius is returned that gives integrated height equal to `fraction`
    of total integral."""
    
    x2 = np.array(x)
    y2 = np.array(y)
    
    if center is None:
        center=calculate_barycenter(x,y)
    
    xe,ye=mirror_profile(x,y,center=center)
    intprof=[np.trapz(ye[:i],xe[:i]) for i in np.arange(len(xe))]
    he=intprof[-1]*fraction
    return xe[np.count_nonzero(intprof<=he)]
    
    
    

#PROFILE STATS AND DERIVED QUANTITIES
def PSF_spizzichino(x,y,alpha=0,xout=None,energy=1.,level=True):
    """Try to use spizzichino theory as in PR notes to calculate Hthe PSF,
    return a vector of same length as xout.
    alpha is incidence angle from normal in degrees, alpha= 90 - shell slope for tilt removed profiles.
        Tilt can be included in the profile, in that case alpha is left to 0
       (total slope must be <0 and alpha>0, this means that profile with tilt is expected to be <0).
    xout can set the output intervals in theta on the focal plane (from specular angle), if not set 512 points are used.
    Lambda is wavelength in keV."""
    
    lambda_mm=12.398425/energy/10**7
    
    if xout is None:
        lout=1001
    else:
        lout=len(xout)
    L=span(x,size=True)
    deltax=L/(len(x))
    
    #calculate and remove slope as alpha. Profile tilt removed is yl
    if level: 
        slope=line(x,y)
        yl=y-slope
        #adjust incidence angle to include the slope removed from profile leveling. 
        # Increasing profile is positive slope angle:
        alpha=alpha*np.pi/180-np.arctan2(y[-1]-y[0],x[-1]-x[0])
    else:
        yl=y
        
    if alpha<=0: raise ValueError
    
    thmax= lambda_mm/(2*deltax*(np.pi/2-alpha))
    #xout is the array of theta for the output
    if xout is None:
        xout=np.linspace(alpha-thmax,alpha+thmax,lout)
    else:
        xout=xout+alpha
        
    scale=np.sqrt(2.)
    I=np.array([np.abs((deltax/L*(np.exp(2*np.pi*1.j/lambda_mm*(x*(np.sin(alpha)-np.sin(theta))-scale*yl*(np.cos(alpha)+np.cos(theta)))))).sum())**2 for theta in xout])
    """
    The above is equivalent to (iterate on all theta in xout):
    I[theta]=np.abs((deltax/L*(np.exp(2*np.pi*1.j/lambda_mm*(x*(np.sin(alpha)-np.sin(theta))-scale*yl*(np.cos(alpha)+np.cos(theta)))))).sum())**2
    """
    return xout-alpha,I

def PSF_raimondiSR(x,y,alpha=0,xout=None,energy=1.):
    """Try to use theory from Raimondi and Spiga A&A2015 to calculate the PSF for single reflection,
    return a vector of same length as xout.
    alpha is incidence angle from normal in degrees, alpha= 90 - shell slope for tilt removed profiles.
    Tilt can be included in the profile, in that case alpha is left to 0
       (total slope must be <0 and alpha>0, this means that profile with tilt is expected to be <0).
    xout can set the output intervals in theta on the focal plane (from specular angle), if not set 512 points are used.
    Lambda is wavelength in keV.
    """
    
    """
    R0 e' necessario ma solo per normalizzare la PSF.  In realta' se nella formula all'ultima riga sostituisci dr1 = L*sin(alpha) = L*R0/2f (usi la singola riflessione, giusto?) vedrai che R0 se ne va e non serve saperlo. f a questo punto sarà semplicemente la distanza alla quale si valuta il campo, non necessariamente la focale.
    
    dr1 = L*sin(alpha) = L*R0/2f
    
    """
    
    lambda_mm=12.398425/energy/10**7
    
    if xout is None:
        lout=1001
    else:
        lout=len(xout)
    L=span(x,size=True)
    deltax=L/(len(x))
    
    #calculate and remove slope as alpha. Profile after tilt removal is yl
    slope=line(x,y)
    yl=y-slope
    #adjust incidence angle to include the slope removed frorm profile leveling. 
    # Increasing profile is positive slope angle:
    alpha=alpha*np.pi/180-np.arctan2(y[-1]-y[0],x[-1]-x[0]) 
    if alpha<=0: raise ValueError
    
    thmax= lambda_mm/(2*deltax*(np.pi/2-alpha))
    #xout is the array of theta for the output
    if xout is None:
        xout=np.linspace(alpha-thmax,alpha+thmax,lout)
    else:
        xout=xout+alpha
    
    dR1=L*np.sin(alpha)  #L1
    R0=1  #F*np.tan(4*alpha)
    d20=1  #randdomly set things to 1
    z1=1
    
    
    
    PSF=1/2*F/(L*lambda_mm)*np.abs(
    [np.sqrt(y/d20)*np.exp(-2*np.pi*1.j/lambda_mm*(d20-z1+x**2/(2*(S-x))))].sum()*deltax
    )**2  #L1
    
    
    
    scale=np.sqrt(2.)
    I=[np.abs((deltax/L*(np.exp(2*np.pi*1.j/lambda_mm*(x*(np.sin(alpha)-np.sin(theta))-scale*yl*(np.cos(alpha)+np.cos(theta)))))).sum())**2 for theta in xout]
    """
    The above is equivalent to (iterate on all theta in xout):
    I[theta]=np.abs((deltax/L*(np.exp(2*np.pi*1.j/lambda_mm*(x*(np.sin(alpha)-np.sin(theta))-scale*yl*(np.cos(alpha)+np.cos(theta)))))).sum())**2
    """
    return xout-alpha,I    
 
'''
#P.S. I suspect this is the same as removing average tilt. Proof??
def autotilt(x,y):
    """Transform a profile by tilting in a way that two contact points are on same horizontal line, return removed line."""
    if delta[0]==0 and delta[-1]==0
    L=span(x,size=1)
    line=x*(delta[-1]+delta[0])/L-delta[0]
    delta=delta-line  #start from endpoints
    i,j=delta.argsort()[:2]   #index of two maxima
'''

##TESTS AND USE

def test_mirror(xx=None,yy=None,center=None,outfolder=None):
    
    print ("mirror a profile about a center coordinate")
    xx=np.arange(30)
    yy=-0.03*xx**2+0.2*xx-5
    center=15
    
    
    plt.close('all')
    plt.figure()
    plt.plot(xx,yy,label='starting profile')
    plt.plot(*mirror_profile(xx,yy,center=center),label='reflect about 15')
    plt.plot(*mirror_profile(xx,yy,center=center),'o')
    for i,p in enumerate(mirror_profile(xx,yy,center=center,split=1)):
        plt.plot(*p,'-.',label='reflect about 15, split#%i'%i) 
    plt.legend(loc=0)

    plt.figure()
    xout=15+np.arange(5)*2
    plt.title('test interpolated output')
    plt.plot(xx,yy,label='starting profile')
    plt.plot(*mirror_profile(xx,yy,center=center,xout=xout),label='reflect about 15')
    plt.plot(*mirror_profile(xx,yy,center=center,xout=xout),'o')
    for i,p in enumerate(mirror_profile(xx,yy,center=center,split=1,xout=xout)):
        plt.plot(*p,'-.',label='reflect about 15, split#%i'%i) 
    plt.legend(loc=0)
    
    plt.show()
    print('done!')
    return xx,yy

def test_HEW():
    #datafile=r'test\01_mandrel3_xscan_20140706.txt'
    print ("uses Spizzichino's formula to predict PSF on sinusoidal ")
    x=np.linspace(0,300.,100)
    y=np.cos(6*np.pi*x/span(x,size=True))/60000.
    xout=np.linspace(-5,5,1000)/206265.
    #x,y=np.genfromtxt(datafile,unpack=True,delimiter=',')
    #y=y-line(x,y)
    plt.figure('profile')
    plt.clf()
    plt.title('profile')
    plt.plot(x,y)
    plt.plot(x,line(x,y))
    plt.xlabel('Axial Position (mm)')
    plt.ylabel('Profile height (mm)')
    plt.figure('PSF')
    plt.clf()
    alpha=89.79
    plt.title('PSF, alpha=%f6.2'%alpha)
    xout,yout=PSF_spizzichino(x,y,alpha=alpha,xout=xout)
    plt.xlabel('angular position around alpha (arcsec)')
    plt.ylabel('Intensity (a.u.)')
    plt.plot(xout*206265.,yout)
    plt.show()
    print('done!')
    return xout,yout
    
if __name__=="__main__":
    test_HEW()
    