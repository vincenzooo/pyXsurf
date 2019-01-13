"""Similar to points, but manage matrix data with x and y. All functions accept arguments in form data,x,y where x and y can be optional in some routines.

# 2016/10/09 copied from PSDanalysis.py, quite rough, it contains also psd functions that should go in psd2d.
"""

import matplotlib.pyplot as plt
import numpy as np
#from pySurf.points import *
#from pySurf.psd2d import *
from pyProfile.profile import polyfit_profile
from plotting.multiplots import compare_images 
from IPython.display import display
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import span
import logging
import os
#from astropy.io.fits import fits

## SEMI-OBSOLETE LEVELING WITH 1D AUXILIARY FUNCTIONS
## STILL ONLY WAY TO HANDLE GENERIC (non-legendre) leveling by line, 
##   code needs review for the part where the leveling function needs x,
##   see example 
"""This is the old leveling. level_by_line """
    
def removelegendre(x,deg):
    """remove degree polyomial, a possible leveling function for leveldata.
    Note: this is superseded by levellegendre"""
    print ("removelegendre is obsolete, use levellegendre instead.")
    lfit=np.polynomial.legendre.legfit
    lval=np.polynomial.legendre.legval
    def rem(y):
        return y-lval(x,lfit(x,y,deg),tensor=0)   
    return rem
  
def removesag(y):
    """remove second degree polyomial, a possible leveling functiondata for leveldata."""
    return y-polyfit_profile (y,degree=2)

def removept(y):
    """remove piston and tilt, a possible leveling functiondata for leveldata."""
    #never tested
    y=y-line (y)
    return y-np.nanmean(y)
    
def level_by_line(data,function=None,axis=0,**kwargs):
    """remove line through extremes line by line (along vertical lines).
    The returned array has 0 at the ends, but not necessarily zero mean.
    If fignum is set, plot comparison in corresponding figure.
    Function is a function of profile vector y that returns a corrected profile.
    
    Completely useless, can be replaced by np.apply_along_axis"""
    
    print ("level_by_line is completely useless function, use np.apply_along_axis.")

    def remove_line(y): #default line removal function
        return y-line(y) 
    
    if function is None:
        function=remove_line
        
    ldata=np.apply_along_axis( function, axis=axis, arr=data, **kwargs )
        
    return ldata

## 2D FUNCTIONS

def save_data(filename,data,x=None,y=None,fill_value=np.nan,addaxis=True,**kwargs):
    """save points on a file. If matrix is true write in matrix form (in this case you have to 
    provide the values for axis). Otherwise write as points in columns."""
    #2016/10/21 rewritten routine to use points_find_grid
    #2014/08/08 default fill_value modified to nan.
    #20140420 moved filename to second argument for consistency.

    #fix x and y according to input
    #calculate x and y from shape if provided, then overwrite if other options 
    shape=data.shape
    if x is None:
        x=np.linspace(0,shape[1],shape[1])
    if y is None:
        y=np.linspace(0,shape[0],shape[0])
    #x and y are automatically calculated, then overwrite if xgrid and ygrid provided

    #save output
    if os.path.splitext(filename)[-1]=='.fits':
        print("creating fits..")
        hdu=fits.PrimaryHDU(data)
        hdu.writeto(filename,overwrite=1,**kwargs)
        return
    else:
        if addaxis:
            #add first column and row with x and y coordinates, unless flag addaxis is set false
            data=np.vstack([x,data])
            data=np.hstack([np.concatenate([[np.nan],y])[:,None],data])
    
    np.savetxt(filename,data,**kwargs)

def get_data(filename,x=None,y=None,xrange=None,yrange=None,matrix=False,addaxis=False,center=None,skip_header=None,delimiter=' ',autocrop=False):
    """read matrix from text file. Return data,x,y
    center is the position of the center of the image in final coordinates (changed on 2016/08/10, it was '(before any scaling or rotation) in absolute coordinates.') If None coordinates are left unchanged.
        Set to (0,0) to center the coordinate system to the data.
    addaxis (if matrix is set) can be set to read values for axis in first row and column
        (e.g. if points were saved with default addaxis=True in save_data.
    autocrop remove frame of nans (external rows and columns made all of nans),
        note it is done before centering. To include all data in centering, crop data
        at a second time with remove_nan_frame
    """

    #2014/04/29 added x and y as preferred arguments to xrange and yrange (to be removed).
    if skip_header is None: 
        skip=0 
    else: 
        skip=skip_header
    
    mdata=np.genfromtxt(filename,skip_header=skip,delimiter=delimiter)
    if addaxis:
        y,mdata=np.hsplit(mdata,[1])
        y=y[1:].flatten() #discard first corner value
        x,mdata=np.vsplit(mdata,[1])
        x=x.flatten() #remove first dimension
    else:
        x=np.arange(mdata.shape[1])
        y=np.arange(mdata.shape[0])

    if autocrop:
        mdata,x,y=remove_nan_frame(mdata,x,y)
        
    if center is not None:
        assert len(center)==2
        x=x-(((np.nanmax(x)+np.nanmin(x))/2)-center[0])
        y=y-(((np.nanmax(y)+np.nanmin(y))/2)-center[0])
    
    return mdata,x,y


    
def crop_data(data,x,y,xrange=None,yrange=None,zrange=None):
    """return data,x,y """
    
    if xrange is None: 
        xrange=span(x)
    if yrange is None: 
        yrange=span(y)
    if zrange is None: 
        zrange=span(data)
    import pdb
    #spdb.set_trace()
    data=data[:,(x>=xrange[0]) & (x<=xrange[1])]
    data=data[(y>=yrange[0]) & (y<=yrange[1]),:]
    x=x[(x>=xrange[0]) & (x<=xrange[1])]
    y=y[(y>=yrange[0]) & (y<=yrange[1])]
    data[((data>zrange[1]) | (data<zrange[0]))]=np.nan 
    return data,x,y

def remove_nan_frame(data,x,y):
    """remove all external rows and columns that contains only nans"""
    nancols=np.where(np.sum(~np.isnan(data),axis=0)==0)[0] #indices of columns containing all nan
    if len(nancols)>0:
        istart=nancols[nancols==np.arange(len(nancols))][-1]+1
        istop=nancols[np.arange(data.shape[1])[-len(nancols):]==nancols][0]
        data=data[:,istart:istop]
        x=x[istart:istop]
    #repeat for rows
    if len(nancols)>0:
        nanrows=np.where(np.sum(~np.isnan(data),axis=1)==0)[0] #indices of columns containing nan
        istart=nanrows[nanrows==np.arange(len(nanrows))][-1]+1
        istop=nanrows[np.arange(data.shape[0])[-len(nanrows):]==nanrows][0]
        data=data[istart:istop,:]
        y=y[istart:istop]    
    return data,x,y
    
def projection(data,axis=0,span=False,expand=False):

    """return average along axis. default axis is 0, profile along x.
    keywords give extended results:
    span: if set, return  3 vectors [avg, min, max] with min and max calculated pointwise along same direction.
    expand: instead of a single vector with point-wise minima, returns lists of all vectors having at least one point 
        that is minimum (maximum) between all vectors parallel to axis. Overrides span.
    ex:
    a=array([[21, 16,  3, 14],
             [22, 17,  6, 15],
             [ 0,  3, 21, 16]])
       
    In [62]: projection(a)
    Out[62]: array([ 14.33333333,12.,10.,15.])
    
    In [73]: projection(a,span=True)
    Out[73]:
    [array([ 14.33333333,12.,10.,15.]),
     array([ 0,  3,  3, 14]),
     array([22, 17, 21, 16])]
    
    In [71]: projection(a,expand=True)
    Out[71]:
    [array([ 14.33333333,12.,10.,15.]),
     array([[21, 16,  3, 14],[ 0,  3, 21, 16]]),
     array([[22, 17,  6, 15],[ 0,  3, 21, 16]])]
    """
    
    pm=np.nanmean(data,axis=axis) 
    
    if expand:
        imin=np.sort(list(set(np.nanargmin(data,axis=axis))))
        pmin=np.take(data,imin,axis)
        imax=np.sort(list(set(np.nanargmax(data,axis=axis))))
        pmax=np.take(data,imax,axis)
        pm=[pm,pmin,pmax]
    elif span:
        pmin=np.nanmin(data,axis=axis) 
        pmax=np.nanmax(data,axis=axis) 
        pm=[pm,pmin,pmax]
    
    return pm
    
def matrix_project(data,axis=1):

    """project a matrix along an axis and return, min, mean and max. 
    For backward compatibility, replaced by projection
    (note different default axis)."""
    
    #return np.nanmin(data,axis=axis),np.nanmean(data,axis=axis),np.nanmax(data,axis=axis)
    return projection(data,span=True,axis=axis)

def fitlegendre(x,y,deg,nanstrict=False,fixnans=False,byline=True):
    """Return a legendre fit of degree deg. Work with 1 or 2D y (if 2D, each column is independently fit
    and x is the coordinate of first axis.
    if nanstrict is True, every column containing nan (or Inf) is considered invalid and a column of nan is returned, if False, nan are excluded and fit is calculated on valid points only (note that since columns
    are slices along first index, the option has no effect on 1D data (nans are always returned as nans)."""
    
    '''
    note this was working with nan 
    def removelegendre(x,deg):
    """remove degree polyomial, a possible leveling function for leveldata."""
    lfit=np.polynomial.legendre.legfit
    lval=np.polynomial.legendre.legval
    def rem(y):
        return y-lval(x,lfit(x,y,deg),tensor=0)   
    return rem
    '''
    
    #this works for 1D or 2D and fits 
    lfit=np.polynomial.legendre.legfit
    lval=np.polynomial.legendre.legval

    import pdb
    #pdb.set_trace()
            
    ## goodind definition can be adjusted for > 2D , datarec definition is already good thanks to [...,] slice
    goodind=np.where(np.sum(~np.isfinite(y),axis=0)==0)[0] #indices of columns non containing any nan
        
    #fit legendre of degree. lval transpose data (!) so it must be transposed back if 2D
    result=y*np.nan
    if len(goodind)>0:
        #this applies only to 2D data
        datarec=lval(x,lfit(x,y[...,goodind],deg))
        if len(y.shape)==2:
            datarec=datarec.T    
        result[...,goodind]= datarec
    
    if len(y.shape)==2:
        if not nanstrict:
            nancols=np.isin(np.arange(y.shape[-1]),goodind,invert=True) #boolean
            if nancols.any():
                datarec=np.zeros((y.shape[0],y.shape[1]-len(goodind)))*np.nan
                for i,col in enumerate(y[...,nancols].T):
                    datarec[:,i]=fitlegendre(x,col,deg,fixnans=fixnans)
                result[:,nancols]=datarec
            # recursively fit columns containing nans one by one to filter nans
            """
            nancols=np.isin(np.arange(y.shape[-1]),goodind,invert=True) #boolean 
            datarec=np.zeros(len(x)*np.nan
            datarec=datarec.T 
            for i,(xx,col) in enumerate(zip(x[nancols],y[...,nancols])):
                datarec[]
            """
    elif len(y.shape)==1:
        mask=np.isfinite(y)
        if np.logical_not(mask).all():
            result=y
        else:
            #if nan need to be fixed coefficients are used on all x,
            # if not fixed only good points are replaced
            coeff=lfit(x[mask],y[mask],deg)
            if fixnans:
                result=lval(x,coeff)
            else:
                result=y*np.nan
                result[mask]=lval(x[mask],coeff)
        
    return result

    
def levellegendre(x,y,deg,nanstrict=False):
    """remove degree polyomial by line, evolution of leveldata using legendre functions
    that work also 2D. nr. of terms fitted in legendre is deg+1 (piston->deg=0).
    For 2D data, data are passed as second argument (y) and y coordinates passed as first (x) (legendre are leveled along columns)"""
    
    datarec=fitlegendre(x,y,deg,nanstrict=nanstrict)

    #result[...,goodind]= y[...,goodind]-datarec #remove poly by line 
    result=y-datarec
    return result #y-datarec 

    
def calculate_slope_2D(wdata,x,y,scale=(1.,1.,1.)):
    """calculate slope maps in x and y.
    return slope in x and y respectively.
    Set scale to (dx,dy,1000.) for z in micron, x,y in mm."""    
    
    dx=np.diff(x)#[:,np.newaxis]
    dy=np.diff(y) [:,np.newaxis]  #original
     
    grad=np.gradient(wdata)  #0->axial(along y), 1->az (along x)
    slopeax=grad[0][:-1,:]/scale[-1]*206265/dy  #in arcseconds from um  #original
    slopeaz=grad[1][:,:-1]/scale[-1]*206265/dx  
    #slopeax=grad[0][:,:-1]/scale[-1]*206265/dy  #in arcseconds from um
    #slopeaz=grad[1][:-1,:]/scale[-1]*206265/dx    
    
    return slopeax,slopeaz
    


## PLOT FUNCTIONS

def plot_data(data,x,y,title=None,outfile=None,units=None,*args,**kwargs):
    """Plot data using imshow and modifying some default properties.
    Units for x,y,z can be passed as 3-el array or scalar, None can be used to ignore unit.
    Broadcast all imshow arguments."""
    
    aspect=kwargs.pop('aspect','equal')
    origin=kwargs.pop('origin',"lower")
    interpolation=kwargs.pop('interpolation',"none")
    if np.size(units)==1:
        units=np.repeat(units,3)
        
    axim=plt.imshow(data,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),
        aspect=aspect,origin=origin,**kwargs)
    plt.xlabel('X'+(" ("+units[0]+")" if units[0] is not None else ""))
    plt.ylabel('Y'+(" ("+units[1]+")" if units[1] is not None else ""))
    
    cb=plt.colorbar()
    if units[2] is not None:
        cb.ax.set_title(units[2])
        #cb.ax.set_xlabel(units[2],rotation=0)
        #cb.ax.xaxis.set_label_position('top')
        #cb.ax.xaxis.label.set_verticalalignment('top') #leave a little bit more room
    if title is not None:
        plt.title(title)
    if outfile is not None:
        plt.savefig(outfile)
    return axim

def xplot_data(data,x,y,title=None,outfile=None,*args,**kwargs):
    aspect=kwargs.pop('aspect','equal')
    origin=kwargs.pop('origin',"lower")
    axim=plt.imshow(data,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect=aspect,origin=origin,**kwargs)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if outfile is not None:
        plt.savefig(outfile)
    return axim
"""
this fails in giving plots with better scaled axis.
fig=plt.figure()
ax1=fig.add_subplot(311, adjustable='box-forced')
ax1.imshow(np.random.rand(100).reshape((10,10)),extent=(0,20,0,20),aspect='equal')
ax2=fig.add_subplot(312,sharex=ax1, adjustable='box-forced')
ax2.imshow(np.random.rand(100).reshape((10,10)),extent=(0,20,0,200),aspect='auto')
"""

def mark_data(datalist,outfile=None,deg=1,levelfunc=None,propertyname='markers',direction=0):
    """plot all data in a set of subplots. Allows to interactively put markers 
    on each of them. Return list of axis with attached property 'markers'"""
    import logging
    from matplotlib.widgets import MultiCursor
    from plotting.add_clickable_markers import add_clickable_markers2
    
    try: 
        logger
    except NameError:
        logger=logging.getLogger()
        logger.setLevel(logging.DEBUG)
    
    axes=list(compare_images([[levellegendre(y,wdata,deg=6),x,y] for (wdata,x,y) in datalist],
        commonscale=True,direction=direction))
    fig=plt.gcf()
    multi = MultiCursor(fig.canvas, axes, color='r', lw=.5, horizOn=True, vertOn=True)
    plt.title('6 legendre removed')
    
    if outfile is not None:
        if os.path.exists(fn_add_subfix(outfile,'_markers','.dat')):
            np.genfromtxt(fn_add_subfix(outfile,'_markers','.dat'))
        
    for a in axes:
        a.images[0].set_clim(-0.01,0.01)
        add_clickable_markers2(ax=a,propertyname=propertyname)

    #display(plt.gcf())

    #Open interface. if there is a markers file, read it and plot markers.
    #when finished store markers

    for a in axes:
        logger.info('axis: '+str(a))
        logger.info('markers: '+str(a.markers))
        print(a.markers)
        
    if outfile is not None:
        np.savetxt(fn_add_subfix(outfile,'_markers','.dat'),np.array([a.markers for a in axes]))
        plt.savefig(fn_add_subfix(outfile,'_markers','.png'))
        
    return axes
    

'''
def compare_2images(data,ldata,x=None,y=None,fignum=None,titles=None,vmin=None,vmax=None,
    commonscale=False):
    """plot two images from data and ldata on a two panel figure with shared zoom.
    x and y must be the same for the two data sets.
    """

    d1mean,d1std=np.nanmean(data),np.nanstd(data)
    d2mean,d2std=np.nanmean(ldata),np.nanstd(ldata)    
    std=min([d1std,d2std])
    
    if x is None:
        x=np.arange(data.shape[1])
    if y is None:
        y=np.arange(data.shape[0])
        
    plt.figure(fignum)
    
    plt.clf()
    ax1=plt.subplot(121)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    if titles is not None:
        plt.title(titles[0])    
    s=(std if commonscale else d1std)
    axim=plt.imshow(data,extent=np.hstack([span(x),span(y)]),
        interpolation='None',aspect='auto',vmin=d1mean-s,vmax=d1mean+s)
    plt.colorbar()

    ax2=plt.subplot(122,sharex=ax1,sharey=ax1)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    if titles is not None:
        plt.title(titles[1])
    s=(std if commonscale else d2std)
    axim=plt.imshow(ldata,extent=np.hstack([span(x),span(y)]),interpolation='None',
        aspect='auto',vmin=d2mean-s,vmax=d2mean+s)
    plt.colorbar()

    return ax1,ax2
'''


def compare_2images(data,ldata,x=None,y=None,fignum=None,titles=None,vmin=None,vmax=None,
    commonscale=False):
    """for backward compatibility, replaced by multiplot.compare_images. plot two images from data and ldata on a two panel figure with shared zoom.
    x and y must be the same for the two data sets.
    """
    return list(compare_images([(data,x,y),(ldata,x,y)],
        interpolation='None',aspect='auto',commonscale=True))

def plot_slope_slice(wdata,x,y,scale=(1.,1.,1.),vrange=None,srange=None,filter=False):
    """
    use calculate_slope_2D to calculate slope and
    plot map and respective slope maps in x and y. Return the three axis.
    Set scale to (1,1,1000.) for z in micron, x,y in mm.
    If filter is set, data out of srange are removed and natural scale is
      used for plot. If not, srange is used for plot axis, but all data
      are used in plots and rms calculation. """
    
    #Note very important that filtering is made pointwise on the 2D slope figure:
    #  all values out of range are masked and all lines containing at least one value are 
    #  excluded.
    #  this is very different than masking on the base of line slice slope rms,
    #  that is the quantity plotted in the histogram.
    
    slopeax,slopeaz=calculate_slope_2D(wdata,x,y,scale=scale)
    if filter:
        slopeax=np.where(np.logical_and(slopeax>srange[0],slopeax<srange[1]),slopeax,np.nan)
        slopeaz=np.where(np.logical_and(slopeaz>srange[0],slopeaz<srange[1]),slopeaz,np.nan)
        psrange=None  #plot scale
    else:
        if srange is not None:
            psrange=[0,max(np.abs(srange))]
        else:
            psrange=None
        
    plt.clf()
    #plt.suptitle('PCO1S16')
    ax1=plt.subplot(221)
    #import pdb
    #pdb.set_trace()
    plt.imshow(wdata,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='equal',origin='lower')
    plt.clim(vrange)
    plt.xlabel('X(mm)')
    plt.ylabel('Y(mm)')
    plt.colorbar()
    ax2=plt.subplot(222,sharey=ax1)
    plt.plot(np.nanstd(slopeaz,axis=1),y)
    isbad=np.isnan(np.nanstd(slopeaz,axis=1)) #boolean array True on cols with nans
    nbad=len(np.where(isbad)[0])
    if nbad > 0:
        plt.plot(np.repeat(ax2.get_xlim()[1],nbad),y[isbad],'>')
    if psrange is not None: plt.xlim(psrange)
    plt.title('total slope rms %6.3f arcsec'%np.nanstd(slopeaz))
    plt.ylabel('Y(mm)')
    plt.xlabel('Slice X Slope rms (arcsec)')
    ax3=plt.subplot(223,sharex=ax1)
    plt.plot(x,np.nanstd(slopeax,axis=0))
    isbad=np.isnan(np.nanstd(slopeax,axis=0)) #boolean array True on cols with nans
    nbad=len(np.where(isbad)[0])
    if nbad > 0:
        plt.plot(x[isbad],np.repeat(ax3.get_ylim()[1],nbad),'^') #plot symbol on invalid points
    if psrange is not None: plt.ylim(psrange)
    plt.title('total slope rms %6.3f arcsec'%np.nanstd(slopeax))
    plt.xlabel('X(mm)')
    plt.ylabel('Slice Y Slope rms (arcsec)')
    plt.tight_layout()
    plt.colorbar().remove() #dirty trick to adjust size to other panels
    #display(plt.gcf())
    ax4=plt.subplot(224)
    #histogram
    plt.hist(np.nanstd(slopeax,axis=0)[~np.isnan(np.nanstd(slopeax,axis=0))],bins=100,label='AX slope',alpha=0.2,color='r',normed=True);
    plt.hist(np.nanstd(slopeaz,axis=1)[~np.isnan(np.nanstd(slopeaz,axis=1))],bins=100,label='AZ slope',alpha=0.2,color='b',normed=True);
    plt.xlabel('Slope rms (arcsec)')
    plt.ylabel('Fraction of points')
    #plt.title ('rms: %5.3f, PV: %5.3f'%(np.nanstd(tmp[:,2]),span(tmp[:2],1)))
    plt.legend(loc=0)
    #display(plt.gcf())
    
    return ax1,ax2,ax3,ax4
    
def plot_slope_2D(wdata,x,y,scale=(1.,1.,1.),vrange=None,srange=None,filter=False):
    """
    use calculate_slope_2D to calculate slope and
    plot map and slice slope rms in x and y. Return the three axis.
    Set scale to (1,1,1000.) for z in micron, x,y in mm.
    If filter is set, data out of srange are removed and automatic scale is
      used for plot. If not, srange is used for plot axis, but all data
      are used in plots and rms calculation. """
    
    slopeax,slopeaz=calculate_slope_2D(wdata,x,y,scale=scale)
    if filter:
        slopeax=np.where(np.logical_and(slopeax>srange[0],slopeax<srange[1]),slopeax,np.nan)
        slopeaz=np.where(np.logical_and(slopeaz>srange[0],slopeaz<srange[1]),slopeaz,np.nan)
        psrange=None  #plot scale
    else:
        psrange=srange
    plt.clf()
    
    #plt.suptitle('PCO1S16')
    ax1=plt.subplot(221)
    plt.imshow(wdata,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='equal',origin='lower')
    plt.xlabel('X(mm)')
    plt.ylabel('Y(mm)')
    plt.colorbar()
    plt.clim(vrange)
    ax2=plt.subplot(222,sharey=ax1)
    plt.imshow(slopeaz,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='equal',origin='lower')
    plt.colorbar().set_label('arcsec', rotation=270)
    plt.clim(psrange)
    plt.title('Azimuthal slope')
    ax3=plt.subplot(223,sharex=ax1)
    plt.imshow(slopeax,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='equal',origin='lower')
    plt.colorbar().set_label('arcsec', rotation=270)
    plt.clim(psrange)
    plt.title('Axial slope')
    ax4=plt.subplot(224)
    #histogram
    plt.hist(slopeax[~np.isnan(slopeax)],bins=100,label='AX slope',alpha=0.2,color='r',normed=True);
    plt.hist(slopeaz[~np.isnan(slopeaz)],bins=100,label='AZ slope',alpha=0.2,color='b',normed=True);
    plt.xlabel('Slope (arcsec)')
    plt.ylabel('Fraction of points')
    #plt.title ('rms: %5.3f, PV: %5.3f'%(np.nanstd(tmp[:,2]),span(tmp[:2],1)))
    plt.legend(loc=0)
    #display(plt.gcf())
    plt.tight_layout()
    return ax1,ax2,ax3,ax4
    
def plot_surface_analysis(wdata,x,y,label="",outfolder=None,nsigma_crange=1,fignum=None,
    figsize=(9.6,6.7),psdrange=None,levelingfunc=None,frange=None):
    """create preview plots for data. This is two panels of raw and filtered data 
    (second order removed) + histogram of height distribution + PSD. 
    Label is typically the filename and is used to generate title and output filenames.
    fignum and figsize are for PSD2D figure (fignum=0 clear and reuse current figure, 
    None create new).
    nsigma_crange and levelingfunc are used to evaluate color range for plot by iteratively 
    leveling and removing outliers at nsigma until a convergence is reached.
    psdrange is used 
    frange is used for plot of PSD2D
    """
    
    units=['mm','mm','$\mu$m']
    order_remove=2 #remove sag, this is the order that is removed in leveled panel
    if np.size(units)==1:   #unneded, put here just in case I decide to take units
        units=np.repeat(units,3)  #as argument. However ideally should be put in plot psd function.
    
    if levelingfunc is None:
        levelingfunc=lambda x: x-legendre2d(x,1,1)[0]
    
    if outfolder is not None:
        os.makedirs(os.path.join(outfolder,'PSD2D'),exist_ok=True)
    else:
        print('OUTFOLDER is none')
    
    rms=np.nanstd(wdata)
    #crange=filtered_span(wdata,nsigma=nsigma_crange,itmax=1,range=True)
    crange=remove_outliers(wdata,nsigma=nsigma_crange,
        flattening_func=levelingfunc,itmax=2,span=1)
    plt.clf()
    
    #FULL DATA
    plt.subplot(221)
    #plt.subplot2grid((2,2),(0,0),1,1)
    plot_data(wdata,x,y,units=units,title='leveled data')
    plt.clim(*crange)
    
    #SUBTRACT LEGENDRE
    ldata=levellegendre(y,wdata,order_remove)
    plt.subplot(223)
    lrange=filtered_span(ldata,nsigma=nsigma_crange,itmax=1,span=True)
    #plt.subplot2grid((2,2),(1,0),1,1)
    plot_data(ldata,x,y,units=units,title='%i orders legendre removed'%order_remove)
    plt.clim(*lrange) 
    
    #HISTOGRAM
    #plt.subplot2grid((2,2),(0,1),2,1)
    plt.subplot(222)
    plt.hist(wdata.flatten()[np.isfinite(wdata.flatten())],bins=100,label='full data')
    plt.hist(ldata.flatten()[np.isfinite(ldata.flatten())],bins=100,color='r',alpha=0.2,label='%i orders filtered'%order_remove)
    #plt.vlines(crange,*plt.ylim())
    plt.vlines(crange,*plt.ylim(),label='plot range raw',linestyle='-.')
    plt.vlines(lrange,*plt.ylim(),label='plot range lev',linestyle=':')
    plt.legend(loc=0)
    
    #FREQUENCY AND PSD ANALYSIS
    if fignum==0:   #this is the figure created by psd2d_analysis
        fig=plt.clf()
    stax=plt.gca()   #preserve current axis
    f,p=psd2d_analysis(ldata,x,y,title=label,
        wfun=np.hanning,fignum=fignum,vrange=lrange,
        prange=psdrange,units=units,
        frange=frange)#,ax2f=[0,1,0,0])
    plt.ylabel('slice rms ($\mu$m)')
    ps=projection(p,axis=1,span=True) #avgpsd2d(p,span=1)      
    if outfolder:
        plt.savefig(os.path.join(outfolder,'PSD2D',
            fn_add_subfix(label,"_psd2d",'.png')))
        save_data(os.path.join(outfolder,'PSD2D',fn_add_subfix(label,"_psd2d",'.txt')),
            p,x,f)
        np.savetxt(os.path.join(outfolder,fn_add_subfix(label,"_psd",'.txt')),np.vstack([f,ps[0],ps[1],ps[2]]).T,
            header='f(%s^-1)\tPSD(%s^2%s)AVG\tmin\tmax'%(units[1],units[2],units[1]),fmt='%f')
            #pdb.set_trace()
    plt.sca(stax)
    
    plt.subplot(224)
    plt.ylabel('axial PSD ('+units[2]+'$^2$ '+units[1]+')')
    plt.xlabel('Freq. ('+units[1]+'$^{-1}$)')
    plt.plot(f,ps[0],label='AVG')
    plt.plot(f,ps[1],label='point-wise min')
    plt.plot(f,ps[2],label='point-wise max')
    plt.plot(f,p[:,p.shape[1]//2],label='central profile')
    plt.ylim(psdrange)
    plt.loglog()
    plt.grid(1)
    plt.legend(loc=0) 
    #ideally to be replaced by:
    """
    plt.subplot(224)
    plot_psd(((f,ps[0]),(f,ps[1]),(f,ps[2),(f,p[:,p.shape[1]//2])),
            ['AVG','point-wise min','point-wise max','central profile'])
    plt.ylabel('axial '+plt.ylabel)
    plt.ylim(psdrange)
    """

    plt.suptitle('%s: rms=%5.3f 10$^{-3}$'%(label,rms*1000)+units[2])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if outfolder:
        plt.savefig(os.path.join(outfolder,fn_add_subfix(label,"",'.png')))
    
    return (ldata,x,y),(f,p)
    
## CASE-SPECIFIC ANALYSIS FUNCTIONS
    
def leveldata(wdata,xwg,ywg): 
    """plot and return matrices of wdata, lwdata, lwdata2
    from data wdata. use polynomial fit, non legendre, each line is leveled on 
    extremes, not on zero piston/tilt""" 
    #import pdb
    #pdb.set_trace()
    
    #create data
    lwdata=level_by_line(wdata.copy())
    lwdata2=level_by_line(wdata.copy(),removesag)
    lwdata2=level_by_line(lwdata2)  ##why this? to level extremes, no piston. The default action of level_by_line is remove line through extremes.
    
    #make plots
    #generator of axis
    for ax in compare_images([wdata,lwdata,lwdata2],xwg,ywg,titles=['original','line lev.','sag. lev.'],fignum=0)  :
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

    return wdata,lwdata,lwdata2

def levelpoints(w0):    
    """plot and return matrices of wdata, lwdata, lwdata2
    from points w0"""
    xwg,ywg=points_find_grid(w0,'grid')[1]
    wdata=resample_grid(w0,xwg,ywg,matrix=1)
    
    return leveldata(wdata,xwg,ywg)
    
##TEST FUNCTIONS
def test_profile_legendre(nanpoints=0):
    """test my routines with profile
    """
    
    x=np.arange(12)
    y=np.arange(20)
    coeff=[[0,3,-2],[12,0.01,-0.5],[-1,0.5,0]]
    inanp=[4,7,10] # indices for point nans if nanpoints != 0
    inanl=[2,5]     #indices for nan lines 

    plt.figure("fit and reconstruction")
    plt.clf()
    for cc in coeff:
        yp=np.polynomial.legendre.legval(y,cc)
        if nanpoints>0:
            yp[inanp]=np.nan
        plt.plot(y,yp,label='build %5.4f %5.4f %5.4f'%tuple(cc))
        yrec=fitlegendre(y,yp,2)
        plt.plot(y,yrec,'x')
    plt.legend(loc=0)


def make_prof_legendre(x,coeff,inanp=[]):      
    """create a profile on x with legendre coefficients coeff.
    Add nans on inanp indices."""
    yp=np.polynomial.legendre.legval(x,coeff)
    yp[inanp]=np.nan
    return yp

def make_surf_legendre(x,y,coeff,inanp=[],inanl=[]):
       
    data=np.empty((len(y),len(x)))
    nrep=len(x)/len(coeff)
    for i,cc in enumerate(coeff):
        yp=make_prof_legendre(y,cc,)
        data[:,i*nrep:(i+1)*nrep]=yp[:,None]
    data[inanp,inanp]=np.nan
    data[:,inanl]=np.nan

    return data

def test_profile_legendre(nans=True,fixnans=True):
    """test on a list of 2D coefficients creating profile then fitting with 
        numpy routines and with data2D routine fitlegendre 
        (wrapper around numpy to handle nans)."""
    
    y=np.arange(20)
    coeff=[[0,3,-2],[12,0.01,-0.5],[-1,0.5,0]]
    if nans:
        inanp=[4,7,10] # indices for point nans if nanpoints != 0
    else:
        inanp=[]
        
    plt.figure("fit and reconstruction")
    plt.clf()
    testdata=[y]
    for cc in coeff:
        yp=make_prof_legendre(y,cc,inanp)
        testdata=testdata+[yp]
        plt.plot(y,yp,label='build %5.4f %5.4f %5.4f'%tuple(cc))
        ccfit=np.polynomial.legendre.legfit(y,yp,2)
        print('fit:%5.4f %5.4f %5.4f'%tuple(ccfit))
        yrec=np.polynomial.legendre.legval(y,ccfit)
        yrec2=fitlegendre(y,yp,2,fixnans=fixnans)
        plt.plot(y,yrec,'x',label='rec  %5.4f %5.4f %5.4f'%tuple(ccfit))
        plt.plot(y,yrec2,'o',label='data2D.fitlegendre')
    plt.legend(loc=0)
    
    return testdata
    
def test_surf_legendre(nans=True,fixnans=True):
    """test how 1D (line) routines in polynomial.legendre work on 1D and 2D data.
    If nanpoints=0 nan are not put in data. If nanpoints=1 or 2 nans are added on some points and
    some lines, with value value of nanpoints determineing the option nanstrict of levellegendre
    (1=false, 2=true).
    """
    x=np.arange(12)
    y=np.arange(20)
    coeff=[[0,3,-2],[12,0.01,-0.5],[-1,0.5,0]]
    if nans:
        inanp=[4,7,10] # indices for point nans if nanpoints != 0
        inanl=[2,5]     #indices for nan lines    x=np.arange(12)
    else:
        inanp,inanl=[]
    
    data=make_surf_legendre(x,y,coeff,inanp,inanl)
    
    #test on surface
    ccfit=np.polynomial.legendre.legfit(y,data,2)
    print("shape of data array: ",data.shape)
    print("shape of coefficient array:",ccfit.shape)
    #print 'fit:%5.4f %5.4f %5.4f'%tuple(ccfit)
    #xx,yy=np.meshgrid(x,y)
    
    datarec=np.polynomial.legendre.legval(y,ccfit).T
    #datarec=np.polynomial.legendre.legval2d(yy,xx,ccfit)

    plt.figure("2D fit and reconstruction with numpy")
    plt.clf()    
    plt.subplot(131)
    plt.title('data')
    plt.imshow(data,origin='lower',interpolation='none')
    plt.colorbar()
    plt.subplot(132)
    plt.title('reconstruction')
    plt.imshow(datarec,origin='lower',interpolation='none')
    plt.colorbar()
    plt.subplot(133)
    plt.title('difference')
    plt.imshow(data-datarec,origin='lower',interpolation='none')
    plt.colorbar()
    plt.tight_layout()
    
    datarec=fitlegendre(y,data,2,fixnans=fixnans)
    plt.figure("2D fit and reconstruction with fitlegendre")
    plt.clf()    
    plt.subplot(131)
    plt.title('data')
    plt.imshow(data,origin='lower',interpolation='none')
    plt.colorbar()
    plt.subplot(132)
    plt.title('reconstruction')
    plt.imshow(datarec,origin='lower',interpolation='none')
    plt.colorbar()
    plt.subplot(133)
    plt.title('difference')
    plt.imshow(data-datarec,origin='lower',interpolation='none')
    plt.colorbar()
    plt.tight_layout()
    
    #test my routine now
    plt.figure("legendre removal with levellegendre")
    plt.imshow(levellegendre(y,data,2),origin='lower',interpolation='none')
    plt.colorbar()
    plt.show()
    
    #compare with old remove legendre
    plt.figure("compare removal routines")
    plt.clf()
    
    data=data+np.random.random(data.shape)
    datarec=levellegendre(y,data,2)
    rem4=removelegendre(y,2)
    rldata=level_by_line(data,rem4)
    
    plt.subplot(131)
    plt.imshow(datarec,origin='lower',interpolation='none')
    plt.title('levellegendre')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(rldata,origin='lower',interpolation='none')
    plt.title('level_by_line')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(datarec-rldata,origin='lower',interpolation='none')
    plt.title('difference')
    plt.colorbar()
    plt.tight_layout()
    
def test_fails_leveling():
    """reproduce warning about fit"""
    from pySurf.instrumentReader import matrixZygo_reader
    f='G:\\My Drive\\Shared by Vincenzo\\Slumping\\Coating Metrology\\newview\\20180330_slumped\\07_PCO1.3S04.asc'
    wdata,x,y=matrixZygo_reader(f,scale=(1000.,1000,1.),center=(0,0)) 
    #wdata,x,y=a[0]-legendre2d(a[0],2,1)[0],a[1],a[2]  #simple way to remove plane
    rem4=removelegendre(y,4)  #remove first 4 legendre components
    ldata=level_by_line(wdata,rem4)
    
    
    
if __name__=="__main__":
    
    #make a test array
    d=np.random.random(20).reshape((5,4))
    d[[3,2],[1,0]]=25
    compare_2images(d,remove_outliers2d(d))
    
    #test avg routines
    p=((np.random.rand(20)*25).astype(int)).reshape(4,5)
    