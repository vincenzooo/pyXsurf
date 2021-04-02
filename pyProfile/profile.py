# -*- coding: utf-8 -*-
"""
It is a library of functions for manipulation or related to optics, acting on a couple of lists x and y, of same length, representing a profile. 

It has only a simple reader (a thin wrapper around `np.genfromtxt`) and a function `register_profile` to align and rescale x and y. Since all functions act on single vectors x and y, more complex readers are not needed here and are left to profile_class. 

Created on Sun Mar 06 16:06:48 2016

@author: Vincenzo Cotroneo
@email: vincenzo.cotroneo@inaf.it
"""
from dataIO.span import span
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pdb


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
        
def make_signal(amp,x=None,L=None,N=None,nwaves=None,phase=0,ystartend=(0,0),noise=0.,minus_one=False):
    """
    Build a signal of length L and number of points N, as a sum of a cosinusoid, a line and a noise. minus_one remove last point (just a convenience e.g. for a periodic profile), note that in 
    this case the returned x corresponds to values returned by
    np.arange e .linspace, however it needs to be called with N+1
    points, so the last one can be excluded and N points returned,
    keeping intervals consistent (this might change in future versions).
    Signal is generated on `x` if this is provided.
    Otherwise it is generated on length `L` (can be range).
    Phase adds a phase in radians (the armonic component of signal is defined as `amp*np.sin(2*np.pi*x/L*nwaves+phase)`).
    
    VC 2020/09/11 added option for L as range and x on which to generate signal.
    2020/07/17 make signal a cosine (it was a sine), because more consistent with real part of imaginary number.
    VC 2020/06/27 horrible interface with args, replace with kwargs with defaults.
    Added phase.
    OLD CODE NEEDS TO be UPDATED! TODO: file search
    """
    
    #pdb.set_trace()
    
    # L is the length (starting from zero) or the interval,
    #    on which the signal is generated (phase and nwaves are applied wrt this interval). `x` are the datapoints on which the signal is calculated (defaults to L).
    
    #Note that L and nwaves (togheter with phase) determine the design parameters of the cosinusoid, while x and N the coordinates on which to calculate values.
    
    if x is None:
        if N is None:
            raise TypeError ("N is needed if x is not provided.")
        elif L is None: #N but not L
            L = [0, N-1]      
    else:
        if N is not None:
            raise TypeError ("N is not needed if x is provided.")            
        if L is None:
            L = span(x)        
    
    #after this, if L was not None, it is set to (2-el) interval start and end for the signal generation.
    #pdb.set_trace()
    
    if np.size(L) == 1:
        assert L is not None  #E' stata aggiustata in base a x e a N.
        L = [0,np.ravel(L)[0]] #this is to get scalar element in every case L =[5], L =[[5]], etc.. not sure it is needed and if this is a good way of doing it.
    elif np.size(L) != 2:
        raise ValueError ("Invalid size for L")

    # generate x cocrdinates if not provided
    if x is None:
        x = np.linspace(L[0],L[1],N,dtype=float) #set to N integers
        #x = np.linspace(0,L,N,dtype=float) #set to N integers
        #x=np.arange(N,dtype=float)/(N-1)*L #set to N integers
#x=L[0] + np.arange(N,dtype=float)/(N-1)*(L[1]-L[0]) 
        
    l=line(x,ystartend)
    if nwaves is not None:
        y=amp*np.cos(2*np.pi*(x-L[0])/span(L,size=True)*nwaves+phase) 
    y = y + noise*np.random.random(N)+l  #apply noise and line at once.
    if minus_one:
        x,y=x[:-1],y[:-1]
    return x,y

def test_make_Signal():# test make_signal
    
    x = np.linspace(-0.5,0.5,100)
    plt.clf()

    print("X span: %s N: %i"%(span(x),len(x)))
    #plt.plot(*make_signal(2.,x,nwaves=3))

    #plt.plot(*make_signal(1.,x,nwaves=3,phase=np.pi/2),'o')
    plt.plot(*make_signal(1.,nwaves=3,L=10,N=100),'o')
    plt.plot(*make_signal(2.,nwaves=3,L=[1,11],N=100),'x')
    plt.plot(*make_signal(2.,nwaves=3,L=[1,11],N=100,phase=np.pi/2))
    plt.plot(*make_signal(2.,x,nwaves=3,L=100,phase=np.pi/2))
    #plt.plot(*make_signal(2.,nwaves=3,N=100,phase=np.pi/2))

def make_circle(x,c,r,sign=1):
    """plot positive part if sign is positive, negative if negative."""
	
    y=np.sqrt(R**2-(x-c[0])**2)+c[1]
    return x,y*np.sign(sign)
    
def find_internal_interval(x,y):
    """find the largest interval of profile with all valid point.
    See pySurf.data2D.find_internal_rectangle."""
    raise NotImplementedError
    #return x,y
    
def remove_nan_ends(x,y=None,internal=False):
    """remove all y points at the two ends of profile that contains only nans in y. If internal is set, return the
    internal crop (largest internal segments without nans)."""
    
    if y is None:
        y = x
        x = np.arange(np.size(y))
    
    #resuls are obtained by slicing,
    nanpts=np.where(~np.isnan(y))[0] #indices of columns containing all nan
    if np.size(nanpts)>0:  
    
        try:
            istart=nanpts[nanpts==np.arange(len(nanpts))][-1]+1
        except IndexError:
            istart=0
            
        try:
            istop=nanpts[np.arange(data.shape[1])[-len(nanpts):]==nanpts][0]
        except IndexError:
            istop=np.size(y)+1
        x=x[istart:istop]
        y=y[istart:istop]

    if internal:
        r=find_internal_interval(x,y)
        x,y=crop_profile(x,y,*r)

    return x,y
    
#profile fitting

def polyfit_profile (x,y=None,degree=1,P=np.polynomial.Legendre):
    """Use one of the poynomial bases in 
   (default to Legendre), and return polynomial of given `degree` that fits the input profile `x`.
    
    Temporary function used to separate the numpy layer from the profile, as preferred implementation switched from np.polynomial functions to classes in np.polynomial.Polynomial, which is more flexible, but also has a more complex descriptor (*).
    This function is thinly wrapped by level_profile, the two will eventually merge, maybe with a flag `fit` to return fit or residuals. 
    
    (*) Note that np coefficients are not directly the polynomial coefficient, e.g. [2,4] doesn't necessarily mean 2 + 4 * x 
    There are two poorly documented parameters `window` and `domain` (both default to -1,1) which regulate the scaling of polynomial result. This means that the polynomial coefficients are first calculated on `window` (default [-1,1]) and the result returned shifted to match `domain` (equal to range of x).
    
    Most common usage is to set window to an interval on which the polynomial
    base is well behaving, and domain to the range of x.
    
    see: https://stackoverflow.com/questions/52339907/numpy-polynomial-generation/52469490
    https://github.com/numpy/numpy/issues/9533
    A detailed example of why using window and domain is here:
    https://numpy.org/doc/stable/reference/routines.polynomials.classes.html
    
    Some of the Polynomials bases are well-behaving, or orthonormal only on a given `window` of x (e.g. Legendre on [-1,1]), we usually want

In using Chebyshev polynomials for fitting we want to use the region where x is between -1 and 1 and that is what the window specifies. However, it is unlikely that the data to be fit has all its data points in that interval, so we use domain to specify the interval where the data points lie. When the fit is done, the domain is first mapped to the window by a linear transformation and the usual least squares fit is done using the mapped data points. The window and domain of the fit are part of the returned series and are automatically used when computing values, derivatives, and such. If they aren’t specified in the call the fitting routine will use the default window and the smallest domain that holds all the data points.

(that seems in contradiction with numpy documentation that states both default to one).    
    
"""    

    if y is None:
        y=x
        x=np.arange(len(y))
    
    goodind=~np.isnan(y) #isolate valid data
    if goodind.any():
        y0=y[goodind]
        x0=x[goodind]
    else:
        x0,y0 = x,y    
    
    res = P.fit(x0,y0,degree) #fit only valid data
    #coeff=np.polyfit(x0,y0,degree)
    
    result = res(x) #result = np.polyval(coeff,x)
    result[~goodind] = np.nan #set back nans if any
    
    return result
 
        
##PROFILE I/O
    
def register_profile(x,y,scale=(1,1),
    strip=False,crop=None,center=None,*args,**kwargs):
    """get x,y and register them using usual set of parameters.

    registering operation are performed in the following order and are:
        scale: scale the two axis of the `scale` factor, if sign is changed, reorder.
        strip: if True, strip all nans at the border of the data.
        crop: list of ranges (2 axis) to pass to profile.crop_profile
        center: the final position of profile center, if 1 element, only x is centered, if 2 elements, y is centered also.
    This means e.g. that data are first cropped (or cleaned of invalid data) than centered. This means that the value puts in the provided coordinate(s) the center of points after cropping operations.

    unexpected parameters passed to register_data are ignored (*args and **kwargs are not used, just suppress error).

    `load_profile` doesn't automatically call `register_data`,
        this might be different from pySurf.
    """

    x=x*scale[0]
    y=y*scale[1]
    
    #if x is inverted, invert data and orient as cartesian.
    #this maybe can be move somewhere else, maybe in class set_data,x,y
    # but be careful to avoid double reflection.
    
    if x[-1]<x[0]:
        x=x[::-1]
        data=np.fliplr(data)
        x=x-min(x)

    #adjust crop and scales
    if strip:  #messes up x and y
        #print("WARNING: strip nans is not properly implemented and messes up X and Y.")
        x,y = remove_nan_ends(x,y)

    #doing cropping here has the effect of cropping on cartesian orientation,
    #coordinates for crop are independent on center and dependent on scale and x bin (pixel size).
    #center is doing later, resulting in center of cropped profile only is placed in the suitable poition.
    if crop is None:
        crop=[None,None]
    x,y=crop_profile(x,y,*crop)

    #center data if center is None, leave unchanged
    if center is not None:
        assert len(center)>=1
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        if len(center)==2:
            y=y-(np.nanmax(y)+np.nanmin(y))/2.+center[1]

    return x,y

def load_profile(file,*args,**kwargs):
    """The simplest file loader using np.genfromtxt.
        Returns x and y."""
        
    return np.genfromtxt(file,unpack=True,*args,**kwargs)

def merge_profile(x1,y1,x2,y2):
    """stitch profiles"""
    raise NotImplementedError("see example of psd merging in G:\My Drive\progetti\read_nid.ipynb")

def save_profile(filename,x,y,**kwargs):
    """Use np.savetxt to save x and y to file """
    np.savetxt(filename,np.vstack([x,y]).T,**kwargs)
        
#PROFILE OPERATIONS        

def get_stats(x,y=None,units=None):
    
    """"""
    if units is None:
        u = ["",""]
    elif np.size(units) == 1:
        u=[units,units]
    else:
        u=units
    assert np.size(u) == 2
    
    stats=['RMS:%3.3g %s'%(np.nanstd(y),u[1]),'PV_X:%3.3g %s, PV_Y:%3.3g %s'%(span(x,size=1),u[0],span(y,size=1),u[1])]
    return stats
    
def level_profile (x,y=None,degree=1):
    """return profile after removal of a (Legendre) polynomial component.
    polyfit_profile has an option to change the polynomial base if Legendre is not the desired base.
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

def crop_profile(x,y=None,xrange=None,yrange=None,*args,**kwargs):
    """Crop a profile to range (can be set to None) in x and y.
    Experimental, crop on y leaves holes in profile."""
    #qui sarebbe utile impostare gli argomenti come x e y necessariamente args e i ranges kwargs.
    #Si potrebbe anche implementare una funzione piu' potente come clip in IDL
    
    if y is None:
        y=x
        x=np.arange(np.size(y))
    
    if xrange is None: 
        xrange=[None,None]
    if xrange[0] is None:
        xrange[0]=np.min(x)
    if xrange[1] is None:
        xrange[1]=np.max(x)

    sel=(x>=xrange[0])&(x<=xrange[1])
    x=x[sel]
    y=y[sel]
    
    if yrange is None: 
        yrange=[None,None]
    if yrange[0] is None:
        yrange[0]=np.nanmin(y)
    if yrange[1] is None:
        yrange[1]=np.nanmax(y)

    sel=(y>=yrange[0])&(y<=yrange[1]) 
    

    return x[sel],y[sel]  

def resample_profile(x1,y1,x2,y2=None):
    """resample y1 (defined on x1) on x2.
    Both x1 and y1 need to be set for input data,
        x2 is returned together with interpolated values as a tuple.
    y2 is not used and put for consistency (can be omitted).
    
    """
    if np.any(np.diff(x1) <= 0): raise ValueError ('x1 must be increasing for interpolation')
    if np.any(np.diff(x2) <= 0): raise ValueError ('x2 must be increasing for interpolation')    
    
    y2 = np.interp(x2,x1,y1)
     
    return x2,y2

def sum_profiles(x1,y1,x2,y2,*args,**kwargs):
    return x1,y1+resample_profile(x2,y2,x1)[1]  
        
def subtract_profiles(x1,y1,x2,y2,*args,**kwargs):
    return x1,y1-resample_profile(x2,y2,x1)[1]  

def calculate_barycenter(x,y):
    """return the x of barycenter using y as weight function."""
    return np.sum(x*y)/len(x)
    
def reflect_profile(xx,yy,center=None,split=False,xout=None):
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
    
    if center is None:
        center=calculate_barycenter(x,y)
    
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
        
        
    
def calculate_HEW(x,y,center=None,fraction=0.5,profile=False):
    """calculate HEW around center from profile x and y by integrating the two sides. If center is None, barycenter is calculated.
    Radius is returned that gives integrated height equal to `fraction`
    of total integral.
    If `profile` is set, a profile x0,y0 of the integrated intensity is returned, where x0 is distance from `center` and y0 is integrated y."""
    
    x2 = np.array(x)
    y2 = np.array(y)
    
    if center is None:
        center=calculate_barycenter(x,y)
    
    xe,ye=reflect_profile(x,y,center=center)
    intprof=[np.trapz(ye[:i],xe[:i]) for i in np.arange(len(xe))]
    he=intprof[-1]*fraction
    if profile:
        return xe,np.array(intprof)
    else:
        return xe[np.count_nonzero(intprof<=he)]
    
    
def plot_HEW(xout,yout,center=None,fraction=0.5):
    """Plots details of HEW calculation."""
    
    cs=' barycenter' if center is None else  'x=%.3g'%center
    #plt.title('HEW centered about'+cs)

    xi,yi=calculate_HEW(xout,yout,profile=True,center=center)
    plt.xlabel('angular distance from barycenter (arcsec)')
    plt.ylabel('Intensity (a.u.)', color='b')

    plt.plot(*reflect_profile(xout*206265.,yout,center=center),
             label='Radial intensity')
    #ax1=plt.gca()
    ax2 = plt.gca().twinx()
    plt.plot(xi*206265.,yi/np.nanmax(yi),'r',label='Normalized integral')

    ax2.set_ylabel('Integrated PSF, normalized')
    #ax2.tick_params('y', colors='r')

    plt.legend(loc=7)
    
    hew=calculate_HEW(xout,yout,center=center)
    plt.axhline(yi[np.count_nonzero(yi<np.nanmax(yi)*fraction)]/np.nanmax(yi),
                ls='--',c='y')
    plt.axvline(hew*206265.,ls='--',c='y')
    
    plt.show()   
    return hew

#PROFILE STATS AND DERIVED QUANTITIES
def PSF_spizzichino(x,y,alpha=0,xout=None,energy=1.,level=True, HEW=True):
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
    
    #if HEW:
    #    calculate_HEW(xout-alpha,I,center=None,fraction=0.5)
        
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
    R0 è necessario ma solo per normalizzare la PSF.  In realtà se nella formula all'ultima riga sostituisci dr1 = L*sin(alpha) = L*R0/2f (usi la singola riflessione, giusto?) vedrai che R0 se ne va e non serve saperlo. f a questo punto sarà semplicemente la distanza alla quale si valuta il campo, non necessariamente la focale.
    
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
    
    
    
    PSF=1/2*F/(L*lambda_mm)*np.abs([np.sqrt(y/d20)*np.exp(-2*np.pi*1.j/lambda_mm*(d20-z1+x**2/(2*(S-x))))].sum()*deltax)**2  #L1
    
    
    """ spizzichino original:
    scale=np.sqrt(2.)
    I=[np.abs((deltax/L*(np.exp(2*np.pi*1.j/lambda_mm*(x*(np.sin(alpha)-np.sin(theta))-scale*yl*(np.cos(alpha)+np.cos(theta)))))).sum())**2 for theta in xout]
    """
    
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

def test_reflect(xx=None,yy=None,center=None,outfolder=None):
    
    print ("specularly reflect a profile about a center position on x axis")
    xx=np.arange(30)
    yy=-0.03*xx**2+0.2*xx-5
    center=15
    
    
    plt.close('all')
    plt.figure()
    plt.plot(xx,yy,label='starting profile')
    plt.plot(*reflect_profile(xx,yy,center=center),label='reflect about 15')
    plt.plot(*reflect_profile(xx,yy,center=center),'o')
    for i,p in enumerate(reflect_profile(xx,yy,center=center,split=1)):
        plt.plot(*p,'-.',label='reflect about 15, split#%i'%i) 
    plt.legend(loc=0)

    plt.figure()
    xout=15+np.arange(5)*2
    plt.title('test interpolated output')
    plt.plot(xx,yy,label='starting profile')
    plt.plot(*reflect_profile(xx,yy,center=center,xout=xout),label='reflect about 15')
    plt.plot(*reflect_profile(xx,yy,center=center,xout=xout),'o')
    for i,p in enumerate(reflect_profile(xx,yy,center=center,split=1,xout=xout)):
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
    