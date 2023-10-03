from pyProfile.psd import plot_sig_psd, normPSD
import matplotlib.pyplot as plt
import numpy as np
import os
from dataIO.span import span
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.dicts import strip_kw
# from pyProfile.profile import line
from pyProfile.psd import plot_psd
from pyProfile.psd import psd_units
from pySurf.data2D import plot_data, projection
from pySurf.outliers2d import remove_outliers2d
from matplotlib.colors import LogNorm
from utilities.imaging import fitting as fit
# from IPython.display import display
from plotting.fignumber import fignumber
from plotting.backends import maximize
from pySurf.data2D import levellegendre, level_by_line
from pySurf.data2D import get_data

import pdb


def psd2d(data, x, y, wfun=None, norm=1, rmsnorm=False, axis=1,includezerofreq=False):
        """Calculate the 2d psd by lines along axis. return freq and psd.
        
        doesnt work with nan.
        use 2d function for psd np.fft.rfft2 for efficiency and mimics
            what done in pySurf.psd.psd
        norm defines the normalization, see function psd.normPSD.
        2017/01/11 broken interface from (x,y,data..),
        added check to correct on the base of sizes."""
        #2017/08/01 complete refactoring, this was internal _psd2d,
        # now is made analogous of 1d pySurf.psd.psd
        # The previous psd2d was including a lot of plotting and output,
        #    it was renamed in plot_psd2d

        #attention, normalization doesn't really work with
        # window, the rms is the rms of the windowed function.
        
        #assert data.shape[0]==data.shape[1]
        
        if axis == 0: # do psds along lines, but note that the result will not be transposed ()        
            data=data.T
            
        if wfun is None:
            win=np.ones(data.shape[0])[:,None]
        else:
            win=wfun(data.shape[0])[:,None]
        N=data.shape[0]
        L=span(y,True)
        yfft=np.fft.rfft2(data*win,axes=[0])

        normfactor=normPSD(N,L,form=norm)
        if rmsnorm:  #normalize to rms of non windowed function
            normfactor=normfactor*np.nanstd(data,axis=0)**2/np.nanstd(data*win,axis=0)**2

        psd=2*normfactor*(np.abs(yfft))**2
        psd[0,:]=psd[0,:]/2.

        freqs = np.fft.rfftfreq(N,float(L)/(N-1))
        
        if not includezerofreq:
            if freqs[0]==0:
                freqs=freqs[1:]
                psd=psd[1:,...]

        return freqs,psd

#COMPUTATION
def avgpsd2d(psddata,axis=1,span=False,expand=False):

    """ wrapper for backward compatibilty and tutorial,
    use directly pySurf.data2D.projection() with axis=1 to get same effect.
    return psd average along axis from 2d psd computed with same
    orientation. default axis is 1, because axial profiles
    are along axis 0, sum on axis 1.
    """
    from pySurf.data2D import projection
    return projection(psddata,axis=axis,span=span,expand=expand)

def rms_power(f,p,rmsrange=None,squeeze=True):
    """integrate `p` (2dpsd) to calculate rms slice power in a range of freq. f is the  frequency axis (vertically oriented in plots) for p. Accepts None as extreme of rmsrange. frequencies are assumed to be equally spaced.
    Return a vectors rms with one element for each range in rmsrange. If one element, extra dimension is
        removed unless squeeze is set (useful e.g. to call from wrapper function and get consistent behavior).
    Note that total rms is calculated as rms of column rms, calculated from PSD for each column. If f[0] is zero, the component is excluded, unless first component of rmsrange is explicitly set to zero. If you want to include all frequencies but zero, set rms range first component to None (or rmsrange itself to None).
    Values can then differ from surface rms in case of invalid points (e.g. as consequence of the fact
    that all lines are weighted equally in line average and also invalid points are excluded).
    2020/07/21 corrected handling of zero frequency and added `includezerofreq` Flag. 
    """
    
    #20180404 renamed _rms_power -> rms_power and rms_power -> plot_rms_power
    if rmsrange is None:
        rmsrange=[None,None]
    if rmsrange[0] is None:
        if f[0] == 0:
            rmsrange[0]=f[1]
        else:
            rmsrange[0]=np.min(f,axis=0)
    if rmsrange[1] is None:
        rmsrange[1]=np.max(f,axis=0)

    mask=(f>=rmsrange[0]) & (f<=rmsrange[1])
    df=f[1]-f[0]

    rms=np.sqrt(np.nansum(p[mask,:],axis=0)*df)  #integrate psd for rms
    rms[np.all(np.isnan(p[mask,:]),axis=0)]=np.nan  #set nan where invalid
    #pdb.set_trace()
    if squeeze:
        if len(rms.shape)==1:
            rms.flatten()

    return rms

def plot_rms_power(f,p,x=None,rmsrange=None,ax2f=None,units=None,*args,**kwargs):
    """Plot curves of slice rms from integration of PSD 2D over one or more ranges of frequency `rmsrange`.
    units of x,y,z (scalar or 3-element) can be provided for axis label (as unit of length, not of f,psd),
        None can be used to exclude units. Units of y are irrelevant because integrate, but kept as 3 vector for
        consistency with other routines.
    ax2f can be an array with same length of rmsrange set to True for frequency ranges to be plotted on 2nd
        y (right) axis. Total rms is always plotted on left axis.
    Return a set of slice rms array (of size x), however if rmsrange is not passed returns only a single array. Zero freq is excluded.
    2020/07/21 corrected bug on wrong rms total calculation in title. 
    """
    #20180404 renamed _rms_power -> rms_power and rms_power -> plot_rms_power
    #added units
    if np.size(units)==1:
        units=np.repeat(units,3)

    if rmsrange is not None:
        if len(rmsrange)==2:
            if np.size(rmsrange[0])==1:  #create a nested list if only a value for range
                rmsrange=[rmsrange]

    loc1,loc2=(2 if np.any(ax2f) else 0),1  #these are the position for legends, set first to auto if only one needed, assigned as plt.legend(loc=loc)
    if x is None:
        x=np.arange(p.shape[1])
        
    if f[0] == 0:
        f=f[1:]
        p=p[1:,...]
        
    #set label if not passed in kwargs:
    l='full freq. range [%4.2g : %4.2g]'%(f[0],f[-1])+((" "+units[1] if units[1] is not None else " [Y]")+"$^{-1}$")
    kwargs['label'] = kwargs.pop('label',l)
    
    #calculate and plot rms
    rms=rms_power(f,p,rmsrange=None)
    plt.plot(x,rms,*args,**kwargs)
    
    
    ax3=plt.gca()
    tit1,tit2=(['Left y axis','Right y axis'] if np.any(ax2f) else [None,None]) #legend title headers
    l1=ax3.legend(loc=loc1,title=tit1)

    #plt.title('Total rms power=%6.3g'%(np.sqrt((rms**2).sum()))+((" "+units[2]) if units[2] is not None else ""))  #wrong math, forgets average?
    plt.title('Total rms power=%6.3g'%(np.sqrt(np.nansum(rms**2)/(np.sum(~np.isnan(rms)))))+((" "+units[2]) if units[2] is not None else ""))
    c=plt.gca()._get_lines.prop_cycler

    rms_v=rms
    if rmsrange is not None:
        rms_v=[rms_v]
        if ax2f is None:
            ax2f=np.zeros(len(rmsrange))
        elif np.size(ax2f)==1:   #any sense in having a scalar input?
            ax2f=[ax2f]*len(rmsrange)
        if np.any(ax2f):
            ax4=ax3.twinx()

        #plt.title(plt.title()+' rms=')
        for fr,a in zip(rmsrange,ax2f):
            ax,tit,loc=((ax4,tit2,loc2) if a else (ax3,tit1,loc1))
            #pdb.set_trace()
            rms=rms_power(f,p,rmsrange=fr,squeeze=False)
            if fr[0] is None: fr[0]=min(f)
            if fr[1] is None: fr[1]=max(f)
            sty=next(c)
            l='freq. range [%4.2g:%4.2g]'%(fr[0],fr[1])
            #print(l)
            ax.plot(x,rms,label=l,**sty)
            plt.sca(ax3)
            #plt.title(ax3.get_title()+' [%4.2g : %4.2g]:%4.2g '%(fr[0],fr[1],np.sqrt((rms**2).sum())))
            plt.title(ax3.get_title()+' [%4.2g : %4.2g]:%4.2g '%(fr[0],fr[1],np.nanmean(rms)))
            rms_v.append(rms)
            ax.legend(loc=loc,title=tit)

    ax3.set_xlabel('X'+ (" ("+units[0]+")" if units[0] is not None else ""))
    ax3.set_ylabel('slice rms'+( ' ('+units[2]+')' if units[2] is not None else ' [units of Z]'))
    #ax3.add_artist(l1)
    return rms_v



#PROCESS MULTIPLE DATA AT ONCE
def multipsd2(datalist,wfun=None):
    """return two lists of vectors respectively freq and avg psd2
    each list has vectors respectively for y and x for data must be passed as list of tuples, each one in form (wdata,xwg,ywg)"""
    flist=[]
    psdlist=[]
    for i,(wdata,xwg,ywg) in enumerate(datalist):
        fw1y,pw1y=psd2d(wdata,xwg,ywg) #psds along y
        fw1x,pw1x=psd2d(wdata.T,ywg,xwg) #psds along x
        pw1yavg=avgpsd2d(pw1y)  #y
        pw1xavg=avgpsd2d(pw1x)  #x
        flist.extend([fw1y,fw1x])
        psdlist.extend([pw1yavg,pw1xavg])
        #result.append(((fw1,pw1avg),(fw1m,pw1mavg)))

    return flist,psdlist

def multipsd3(wdata,xwg,ywg,wfun=None):
    """return two lists respectively freq and psd2d
    each list has vectors respectively for y and x for each data set passed.
    data must be passed as list of tuples, each one in form (wdata,xwg,ywg)"""

    fw1y,pw1y=psd2d(wdata,xwg,ywg) #psds along y
    fw1x,pw1x=psd2d(wdata.T,ywg,xwg) #psds along x

    return [fw1y,fw1x],[pw1y,pw1x]

def multipsd(datalist):
    """return a tuple of two couples ((yfreq.,ypsd),(xfreq.,xpsd))
    for each of the data passed, respectively for y and x.
    data must be passed as list of tuples (wdata,xwg,ywg)"""
    result=[]
    for (wdata,xwg,ywg) in datalist:
        fw1,pw1=psd2d(xwg,ywg,wdata) #psds along y
        fw1m,pw1m=psd2d(ywg,xwg,wdata.T) #psds along x
        pw1avg=avgpsd2d(pw1)  #y
        pw1mavg=avgpsd2d(pw1m)  #x
        result.append(((fw1,pw1avg),(fw1m,pw1mavg)))
    return result
    
def plot_psd2d(f,p,x,prange=None,includezerofreq=False,units=None,*args,**kwargs):
    """plots a 2d psd as a surface with logaritmic color scale on the current axis. Return axis.
    If a zero frequency is present it is excluded by default from plot, unless includezerofreq is set to True.
    Units (3-el array) is units of lengths for data (not of PSD), can be None.
    2020/07/10 uncommented call to `pySurf.data2D.plot_data` (in "functions" module `data2D`).
    The "old" code segment called instead directly `plt.imshow`.  
    In some sense, replicates code in `plot_data` and was not necessarily in sync
    (e.g.: stats flag is not included here). 
    This seems to fix the issue. """
    #2018/04/05 critical renaming of plot_psd2d(wdata,x,y..) -> psd2d_analysis
    # and _plot_psd2d(f,p,x) -> plot_psd2d consistently with other routines.

    if len(f.shape)==2:
        print ("""Wrong size detected in call to plot_psd2d. Functions were renamed,
                you probably want to call psd2d_analysis(wdata,x,y..).
                I will try to correct automatically for this time, but you need to
                update calling code URGENTLY, it will be discontinued.""")
        return psd2d_analysis(f,p,x,*args,**kwargs)

    if np.size(units)==1:
        units=np.repeat(units,3)
    #pdb.set_trace()

    if not includezerofreq:
        if f[0]==0:
            f=f[1:]
            p=p[1:,...]

    if prange is None:
        prange=[None,None]

    # new:
    
    cbunits = psd_units(units)
    #pdb.set_trace()
    #plt.yscale('log')
    plt.title('2D PSD')
    ax = plot_data(p,x,f,norm=LogNorm(vmin=prange[0],vmax=prange[1]),
    units=cbunits,aspect='auto',*args,**kwargs)
    plt.ylabel('freq. ('+cbunits[1]+')')
    #pdb.set_trace()
    cb =  plt.gca().images[-1].colorbar
    
    #after introducing psd_units, cbunits is always defined,
    # conditions can be modified, but leaave like this for generality.
    #cb.ax.set_title("PSD"+(" ("+cbunits[2]+")") if cbunits[2] else "")
    
    #plt.ylabel('Frequency ('+cbunits[1]+")")
    #plt.xlabel('X'+ (" ("+cbunits[0]+")" if cbunits[0] else ""))
    plt.grid(1)

    return ax

#PLOTTING

def psd2d_analysis(wdata,x,y,title=None,wfun=None,vrange=None,
    rmsrange=None,prange=None,fignum=5,rmsthr=None,includezerofreq=False,
    aspect='auto', ax2f=None, units=None,outname=None,norm=1,rmsnorm=True):
    """ Calculates 2D PSD as image obtained combining all profile PSDS calculated along vertical slices of data. Resulting image has size 
    If title is provided rms slice power is also calculated and plotted on three panels with figure and PSD.
    
    Return PSD as PSD2D object. 

    uses plot_rms_power(f,p,rmsrange=None) to calculate rms power.
    fignum window where to plot, if fignnum is 0 current figure is cleared,
    if None new figure is created. Default to figure 5.
    rmsrange is one or a list of frequency ranges for plotting integrated rms. Can contain None to use max or min.
    If axf2 is set to boolean or array of boolean, plot slice rms for the frequencies
    associated to rmsrange on second axis. rmsrange must be set accordingly (same number of elements).
    
    rmsthr sets a threshold for data inclusion. If rms is above the value, the line is considered to contain invalid data and is removed from output PSD.
    This makes it easy to average the returned PSDs.
    Corresponding data are still visualized in central panel, but are marked with a red cross at top of y axes.
    If multiple rms range intervals are provided, line is removed if any of them is over the threshold, but this might change in future, e.g. TODO: make it possible to add vector threshold with as many elements as rms range intervals.

    Set outname to empty string to plot without generating output.

    typical values:
    rmsrange=[[None,0.1],[0.1,0.2],[0.2,1],[1,None]]  #list of frequency intervals for rms
    ax2f=[0,1,1,0,0]             #   axis where to plot rms corresponding to intervals in rmsrange: 1 right, 0 left
    wfun  = np.hanning           #   type of windows for fourier transform
    units = ("mm","mm","$\mu$m") #   units of surface data from which PSD is calculated

    vrange_surf=([-0.5,0.5])  #color scale of surface map
    vrange_leg=([-0.05,0.05])   #color scale of legendre removed map
    prange=np.array((1e-8,1.e-1))#np.array((5e-8,1.e-5))  #color scale of 2d psd plot

    fs,ps=psd2d_analysis(levellegendre(y,wdata,2),x,y,outname="",wfun=np.hanning,
    rmsrange=rmsrange,prange=prange,ax2f=ax2f,units=units)
    """
    #2018/04/05
    # -critical renaming of plot_psd2d(wdata,x,y..) -> psd2d_analysis
    # and _plot_psd2d(f,p,x) -> plot_psd2d consistently with other routines.
    # -added title to determine if generating plot, with the intent of replacing outname.
    plt.pause(1)
    #print('cane',fignum) 
    if outname is not None:  #handle backcompatibility
        print ('psd2d_analysis WARNING: `title` replaced `outname` and output figure will be no more generated.'+
            'OUTNAME will be removed in next version, use title and save the plot after returning from routine.')
        if title is not None:
            print('outname should be not used together with title, it will be ignored.')
        else:
            title=outname
    
    if vrange is not None:
        if prange is None:
            print("""WARNING:  older versions of psd2d use vrange as range of psdvalue.
                    In updated version prange is range of psd and vrange is range for surface
                    plots. Update your code.""")

    #check compatibility with old interface
    if len(wdata.shape)==1:
        if len(y.shape)==2:
            print("WARNING: your code calling psd2d uses old call")
            print("psd2d(x,y,data), automaticly adjusted, but")
            print("update IMMEDIATELY your code to call")
            print("psd2d(data,x,y)")
            x,y,wdata=wdata,x,y

    f, p = psd2d(wdata, x, y, wfun=wfun, norm=norm, rmsnorm=rmsnorm, includezerofreq=includezerofreq)  # calculate PSD 2D
    
    if not includezerofreq:
        if f[0]==0:
            f=f[1:]
            p=p[1:,...]
    
    # GENERATE OUTPUT
    if title is not None:

        if np.size(units)==1:
            units=np.repeat(units,3)

        if prange is None:
            if f[0] == 0:
                prange=span(p[1:,:])
            else:
                prange=span(p)
                if prange[0]<1e-20:  #arbitrary small value
                    print("WARNING: low limit detected in prange, can cause problems with log color scale.")
        if vrange is None:
            vrange=span(wdata)

        #plot surface map, who knows why on figure 5       
        fig=fignumber(fignum)
        #pdb.set_trace()
        #
        plt.clf()
        maximize()
        plt.draw()

        plt.suptitle(title)

        ################
        #plot Surface
        ax1=plt.subplot(311)
        plot_data(wdata,x,y,aspect=aspect,vmin=vrange[0],vmax=vrange[1],units=units,title='Surface',stats=True)
        plt.xlabel(None)
        plt.grid(1)

        ################
        #plot 2D PSD
        ax2=plt.subplot(312,sharex=ax1)
        plot_psd2d(f,p,x,prange=prange,units=units)
        if rmsrange is not None:      #plot horizontal lines to mark frequency ranges for rms extraction,
        #rms is tranformed to 2D array
            if len(np.array(rmsrange).shape)==1:  #if only one interval, make it 2D anyway
                rmsrange=[rmsrange]
            rmsrange=np.array(rmsrange)
            for fr in np.array(rmsrange):  #I want fr to be array type, even if I don't care about rmsrange this does the job
                ax2.hlines(fr[fr != None],*((span(x)-x.mean())*1.1+x.mean()))
        ax2.set_xlabel("")
        plt.xlabel(None)
        #plt.subplots_adjust(top=0.85)

        ################
        #plot rms slice
        ax3=plt.subplot(313,sharex=ax1)
        if f[0]==0: #ignore component of 0 frequency in rms calculation if present
            ff,pp=f[1:],p[1:]
        else:
            ff,pp=f,p

        rms=plot_rms_power(ff,pp,x,rmsrange=rmsrange,ax2f=ax2f,units=units)
        #pdb.set_trace()
        mask=np.isfinite(rms) #True if good. it is an array
        if rmsthr is not None:
            mask = mask & (rms  < rmsthr)
            ax3.hlines(rmsthr,*ax2.get_xlim()) #plot markers on rms chart
            ax2.plot(x[~mask],np.repeat(ax2.get_ylim()[1],len(x[~mask])),'rx') #plot markers on psd2d chart
        
        #pdb.set_trace()
        #questa era qui, ma dava errore perche' mask e' lineare
        #mask = np.all(mask,axis =0) # if any of mask is False -> False
        #p[:,~mask]=np.nan  #questo da errore se mask e' piu' di una dimensione
        ax3.grid(1)
        #plt.tight_layout(rect=(0, 0.03, 1, 0.95) if title else (0, 0, 1, 1))
        
        # replaced with more evolved interactive resize:
        #  plt.colorbar().remove() #dirty trick to adjust size to other panels
        def resize(event):
            box1 = ax1.get_position()
            box2 = ax2.get_position()
            box3 = ax3.get_position()
            ax2.set_position([box1.x0, box2.y0, box1.width , box2.height])
            ax2.set_adjustable("box",share=True)
            ax3.set_position([box1.x0, box3.y0, box1.width , box3.height])
            ax3.set_adjustable("box",share=True)

        cid = fig.canvas.mpl_connect('draw_event', resize)
        cid2 = fig.canvas.mpl_connect('resize_event', resize)

        resize(None)

        #plt.tight_layout()

        if outname:    #kept for compatibility, will be removed in next version
            plt.savefig(fn_add_subfix(outname,'_2dpsd','.png'))

    return f,p

def psd_analysis(*args,**kwargs):
    """
    Convenience function to plot average psd.
    Wrapper around `psd2d analysis` analysis. Adds a plot of average PSD and return it. 
    
    Accept parameters for psd2d_analysis and plot_psd,
    including `analysis=True`.
    Call it directly to obtain 2d psd.
    Use 
    Use `plot_psd` from `pySurf.psd` to plot. 
    """
    
    #pdb.set_trace()
    tmp = strip_kw(kwargs,psd2d_analysis,title="",exclude=['units'])
    f,p2=psd2d_analysis(*args,**tmp)
    p=projection(p2,axis=1)
    plt.figure()
    #u = kwargs.get('units',None)
    #t = kwargs.get('title',None)
    kw = strip_kw(kwargs,plot_psd)
    #u = kw['units'] if 'units' in kw else None
    
    #pdb.set_trace()
    plot_psd(f,p,**kw)

    #plot_psd(f,p,units=[u[0],u[1]] if u else u, title= t, **kwargs)
    return f,p


def compare_2dpsd(data,ldata,x=None,y=None,fignum=None,titles=None,vmin=None,vmax=None):
    """plot a four panel data and psd for the two figures from data and ldata.
    x and y must be the same for the two data sets.
    Calculate PSD. Return freqs, psd and lpsd."""

    if x is None or y is None:
        raise NotImplementedError
    freqs,p=psd2d(x,y,data)
    freqs,pl=psd2d(x,y,ldata)

    #ldata=remove_outliers2d(data,nsigma=3,fignum=fignum+1)
    if fignum is not None:
        plt.figure(fignum)
        plt.clf()
        ax1=plt.subplot(221)
        if titles is not None:
            plt.title(titles[0])
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        axim=plt.imshow(data,extent=(x[0],x[-1],y[0],y[-1]),
                        interpolation='none',aspect='equal')
        plt.colorbar()

        ax2=plt.subplot(222,sharex=ax1)

        plt.title('PSD')

        plt.imshow(p,norm=LogNorm(vmin=vmin,vmax=vmax),interpolation='none',
                   extent=(x[0],x[-1],freqs[0],freqs[-1]),aspect='auto')
        plt.xlabel('X (mm $\mu$m^2)')
        plt.ylabel('Freq (mm^-1)')
        plt.colorbar()
        plt.draw()

        ax3=plt.subplot(223,sharex=ax1,sharey=ax1)
        if titles is not None:
                    plt.title(titles[1])
        plt.xlabel('X (mm $\mu$m^2)')
        plt.ylabel('Y (mm)')
        axim=plt.imshow(ldata,extent=(x[0],x[-1],y[0],y[-1]),interpolation='none',aspect='equal')
        plt.colorbar()

        plt.subplot(224,sharex=ax1)

        plt.title('PSD outliers corrected')
        plt.imshow(pl,norm=LogNorm(vmin=vmin,vmax=vmax),interpolation='none',
                   extent=(x[0],x[-1],freqs[0],freqs[-1]),aspect='auto')
        plt.xlabel('X (mm)')
        plt.ylabel('Yfreq (mm^-1)')
        plt.colorbar()
        plt.draw()

    return freqs,p,pl

#SCRIPTS
def avgpsd(fitsfile,correct=False,reader=None,level=None,nsigma=3,rotate=False,**kwargs):
    """Wrapper on avgpsd2d. Extract average psd from a fits image,
    includes reading and outliers removal.
    Reader is a function of fitsfile that returns x,y,data."""
    if reader is None:  #I don't care it's not pythonic, simplest way to avoid circular imports.
        from pySurf.readers.instrumentReader import getdata
        reader=getdata

    x,y,data=reader(fitsfile) #x and y in mm, data in um
    if rotate:
        data=data.T
        x,y=y,x
    if level is not None:
        data=level_by_line(data,**kwargs)
    if correct:
        data=remove_outliers2d(data,nsigma=nsigma)
    f,p=psd2d(x,y,data)
    pm=avgpsd2d(p,axis=1)
    return f,pm

def calculatePSD(wdata,xg,yg,outname="",wfun=None,vrange=None,rmsrange=None,prange=None,fignum=1):
    """given points w, calculate and plot surface maps with different leveling (piston, tilt, sag, 10 legendre)
    use psd2d to calculate and save x and y 2d PSDs, plots only y.
    fignum window where to plot, if fignnum is 0 current figure is cleared, if None new figure is created. Default to figure 1 ."""
    ###AWFUL CODE, partially fixed -> fix plotting window

    ## FOUR PANEL PLOT save as _lev.png, _piston.dat, _tilt.dat, _sag.dat
    ## on fignum
    if fignum==0:
        plt.clf()
    else:
        plt.figure(fignum)
    plt.clf()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    #xg,yg=points_find_grid(w,'grid')[1]
    wdata,lwdata,lwdata2=leveldata(wdata,xg,yg) #calculate and make a plot with 3 panels
    if outname:
        np.savetxt(fn_add_subfix(outname,'_piston','.dat'),wdata) #doesn't  write x and y
        np.savetxt(fn_add_subfix(outname,'_tilt','.dat'),lwdata)
        np.savetxt(fn_add_subfix(outname,'_sag','.dat'),lwdata2)

    leg10=wdata-fit.legendre2d(wdata,10,10)[0]
    #forth panel contains legendre
    plt.subplot(224)
    plt.imshow(leg10,extent=(xg[0],xg[-1],yg[0],yg[-1]),
                interpolation='None',aspect='auto')
    plt.title('10 legendre')
    plt.colorbar()
    #display(plt.gcf())
    if outname:
        plt.savefig(fn_add_subfix(outname,'_lev','.png'))

    ## DATA OUTPUT save as _psd.dat, psd2d creates
    #create packed data for txt output
    #containing x and y psds, with 3 different levelings
    #resulting in 6 couples freq/psd.
    # matches the order specified in header

    #header for final matrix output, define labels of psd sequence
    head='yfreq ypsd xfreq xpsd ylevfreq ylevpsd xlevfreq xlevpsd ysagfreq ysagpsd xsagfreq xsagpsd'

    #this assumes x and y are same xg and yg on all datalist
    datalist=[wdata,lwdata,lwdata2]
    labels=['_piston','_tilt','_sag'] #postfix for output file
    flist=[]
    psdlist=[]
    for d,l in zip(datalist,labels):
        plt.figure(3)
        #x and y psds
        fx,px=psd2d(d.T,yg,xg,wfun=wfun) #psds along x, no plot
        fy,py=plot_psd2d(d,xg,yg,outname=fn_add_subfix(outname,l),wfun=wfun,
            rmsrange=rmsrange,prange=prange,vrange=vrange) #psds along y, plot
        #display(plt.gcf())

        #note y goes first
        flist.extend([fy,fx])
        psdlist.extend([py,px])
    avgpsdlist=[avgpsd2d(p) for p in psdlist]

    #generate output array and save it with header
    outarr=np.empty((np.max([len(f) for f in flist]),len(psdlist)*2))*np.nan
    for i,(f,p) in enumerate(zip(flist,avgpsdlist)):
        outarr[:len(f),i*2]=f
        outarr[:len(p),i*2+1]=p

    if outname:
        np.savetxt(fn_add_subfix(outname,'_psd','.dat'),outarr,header=head,fmt='%f')

    ## PLOT AVG PSDs save as _psds.png
    fw1,fw1m,fw2,fw2m,fw3,fw3m=flist  #used only in plot
    pw1avg,pw1mavg,pw2avg,pw2mavg,pw3avg,pw3mavg=avgpsdlist  #average psds
    plt.figure(2)
    plt.clf()
    plt.plot(fw1,pw1avg,label='Y')
    plt.plot(fw2,pw2avg,label='Y lev.')
    plt.plot(fw3,pw3avg,label='Y sag.')
    plt.plot(fw1m,pw1mavg,'--',label='X')
    plt.plot(fw2m,pw2mavg,'--',label='X lev.')
    plt.plot(fw3m,pw3mavg,'--',label='X sag.')
    plt.loglog()
    plt.grid()
    plt.legend(loc=0)
    if outname:
        plt.savefig(fn_add_subfix(outname,'_psds','.png'))

    return outarr


def calculatePSD2(wdata,xg,yg,outname="",wfun=None,vrange=[None,None],rmsrange=None,prange=None,fignum=1,misal_deg=(1,1),leg_deg=(10,10)):
    """Updated version with subtracted 4 terms legendre."""
    """given points w, calculate and plot surface maps with different leveling (piston, tilt, sag, 10 legendre)
    use psd2d to calculate and save x and y 2d PSDs, plots only y.
    fignum window where to plot, if fignnum is 0 current figure is cleared, if None new figure is created. Default to figure 1 ."""
    
    from plotting.multiplots import compare_images
    ##STILL AWFUL CODE
    # misal_deg is useless, since data are expected to be already leveled.
    # if not, still can be applied, but is quite useless.

    ###AWFUL CODE, partially fixed -> fix plotting window

    ## FOUR PANEL PLOT save as _lev.png, _piston.dat, _tilt.dat, _sag.dat
    ## on fignum
    fig=fignumber(fignum)
    plt.clf()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    #
    wdata=-fit.legendre2d(wdata,1,1)[0]
    lwdata=wdata-fit.legendre2d(wdata,*misal_deg)[0]
    lwdata2=wdata-fit.legendre2d(wdata,*leg_deg)[0]
    #
    #make plots
    ax=compare_images([wdata,lwdata,lwdata2],xg,yg,titles=['original','cone corrected','10 legendre 2D corrected'],fignum=0,commonscale=True,vmin=vrange[0],vmax=vrange[1])  #generator of axis, not sure how it can work below, missing something?
    for im in ax:
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
    #
    ##return wdata,lwdata,lwdata2

    wdata,lwdata,lwdata2=leveldata(wdata,xg,yg) #calculate and make a plot with 3 panels
    if outname:
        np.savetxt(fn_add_subfix(outname,'_piston','.dat'),wdata) #doesn't  write x and y
        np.savetxt(fn_add_subfix(outname,'_%2i_%2i_misal'%misal_deg,'.dat'),lwdata)
        np.savetxt(fn_add_subfix(outname,'_%2i_%2i_leg'%leg_deg,'.dat'),lwdata2)

    leg10=wdata-fit.legendre2d(wdata,*leg_deg)[0]
    #forth panel contains legendre
    plt.subplot(224)
    plt.imshow(leg10,extent=(xg[0],xg[-1],yg[0],yg[-1]),
                interpolation='None',aspect='auto')
    plt.title('(%i,%i) 2D legendre removed'%leg_deg)
    plt.colorbar()
    #display(plt.gcf())
    if outname:
        plt.savefig(fn_add_subfix(outname,'_lev','.png'))

    ## DATA OUTPUT save as _psd.dat, psd2d creates
    #create packed data for txt output
    #containing x and y psds, with 3 different levelings
    #resulting in 6 couples freq/psd.
    # matches the order specified in header

    #header for final matrix output, define labels of psd sequence
    head='yfreq ypsd xfreq xpsd ylevfreq ylevpsd xlevfreq xlevpsd ysagfreq ysagpsd xsagfreq xsagpsd'

    #this assumes x and y are same xg and yg on all datalist
    #change this part, we are no more interested in doing psd on all possible leveling, only 4 terms removed by line
    # along the direction of interest, however to keep constant number of columns, replace piston, tilt, sag
    # with tilt, sag, 4 legendre
    datalist=[wdata,lwdata,lwdata2]
    labels=['_tilt','_mis','_leg'] #postfix for output file
    flist=[]
    psdlist=[]
    for d,l in zip(datalist,labels):
        plt.figure(3)
        #x and y psds
        fx,px=psd2d(d.T,yg,xg,wfun=wfun) #psds along x, no plot
        fy,py=plot_psd2d(d,xg,yg,outname=fn_add_subfix(outname,l),wfun=wfun,
            rmsrange=rmsrange,prange=prange,vrange=vrange) #psds along y, plot
        #display(plt.gcf())

        #note y goes first
        flist.extend([fy,fx])
        psdlist.extend([py,px])
    avgpsdlist=[avgpsd2d(p) for p in psdlist]

    #generate output array and save it with header
    outarr=np.empty((np.max([len(f) for f in flist]),len(psdlist)*2))*np.nan
    for i,(f,p) in enumerate(zip(flist,avgpsdlist)):
        outarr[:len(f),i*2]=f
        outarr[:len(p),i*2+1]=p

    if outname:
        np.savetxt(fn_add_subfix(outname,'_psd','.dat'),outarr,header=head,fmt='%f')

    ## PLOT AVG PSDs save as _psds.png
    fw1,fw1m,fw2,fw2m,fw3,fw3m=flist  #used only in plot
    pw1avg,pw1mavg,pw2avg,pw2mavg,pw3avg,pw3mavg=avgpsdlist  #average psds
    plt.figure(2)
    plt.clf()
    plt.plot(fw1,pw1avg,label='Y')
    plt.plot(fw2,pw2avg,label='Y lev.')
    plt.plot(fw3,pw3avg,label='Y sag.')
    plt.plot(fw1m,pw1mavg,'--',label='X')
    plt.plot(fw2m,pw2mavg,'--',label='X lev.')
    plt.plot(fw3m,pw3mavg,'--',label='X sag.')
    plt.loglog()
    plt.grid()
    plt.legend(loc=0)
    if outname:
        plt.savefig(fn_add_subfix(outname,'_psds','.png'))

    return outarr


'''
def plot_profile_outliers(data,x,y,ldata,i=1000):
    print("obsolete, this routine will be removed from psd2d")
    plt.figure(2)
    plt.clf()
    plot_sig_psd(x,data[:,i],scale=1,label='data')
    plot_sig_psd(x,ldata[:,i],scale=1.,label='outliers removed')
    plt.suptitle('Effects of outliers removal - %s'%datafile).set_size('large')

    plt.grid()
    plt.legend(loc=0)
    plt.subplot(211)
    plt.grid()

    plt.subplot(212)
    freqs,p=psd2d(x,y,data)
    pm=avgpsd2d(p,axis=1)
    plt.plot(freqs,pm,label='2D average raw')
    lpm=avgpsd2d(psd2d(x,y,ldata)[1],axis=1)
    plt.plot(freqs,lpm,label='2D average corr.')
    plt.ylabel('(mm $\mu$m^2)')
    plt.xlabel('mm^-1')
    plt.legend(loc=0)
'''

#####################################################

##TESTS

def test_slicerms():
    """test plot options for slicerms """
    wf=r'test\PSD\2dpsd\171010_Secondary_Mandrel_C3_RefSub.dat'
    outfolder=r'test\PSD\2dpsd\output'
    wdata,x,y=get_data(wf,matrix=True,addaxis=True)
    outfile=fn_add_subfix(wf,'_4leg',strip=True,pre=outfolder+os.path.sep)

    f3,psd3=psd2d(levellegendre(y,wdata,4),x,y,wfun=np.hanning,norm=1,rmsnorm=True)
    plt.figure()
    r=plot_rms_power(f3,psd3,x=None,rmsrange=None)
    plt.savefig(fn_add_subfix(outfile,'_testslicerms_01','.png'))
    plt.figure()
    plot_rms_power(f3,psd3,x=x,rmsrange=[0.1,1])
    plt.savefig(fn_add_subfix(outfile,'_testslicerms_02','.png'))
    plt.figure()
    plot_rms_power(f3,psd3,x=x,rmsrange=[0.1,1],ax2f=[1])
    plt.savefig(fn_add_subfix(outfile,'_testslicerms_03','.png'))
    plt.figure()
    plot_rms_power(f3,psd3,x=x,rmsrange=[[None,0.1],[0.1,1],[1,None]],ax2f=[0,1,1])
    plt.savefig(fn_add_subfix(outfile,'_testslicerms_04','.png'))

def test_plotpsd2d():
    """test of 3 panel psd2d plot, including 2d psd and slice rms. """
    prange=np.array((1e-10,1))#np.array((5e-8,1.e-5))  #color scale of 2d psd plot
    rmsrange=[[0.1,1],[0.01,0.1]]  #range of frequency for rms calculation
    vrange_surf=([-0.15,0.15])  #color scale of surface map
    vrange_leg=([-0.025,0.025])   #color scale of legendre removed map

    wf=r'test\PSD\2dpsd\171010_Secondary_Mandrel_C3_RefSub.dat'
    outfolder=r'test\PSD\2dpsd\output'
    wdata,x,y=get_data(wf,matrix=True,addaxis=True)
    outfile=fn_add_subfix(wf,'_4leg',strip=True,pre=outfolder+os.path.sep)

    plt.figure()
    #PSD with polynomial removal and average in f3, psd3 and wptot
    f3,psd3=plot_psd2d(levellegendre(y,wdata,2),x,y,wfun=np.hanning,
                outname=outfile,vrange=vrange_leg,prange=prange,rmsrange=rmsrange,ax2f=True)

def test_psd2d():
    """test of plotting 2dpsd alone """
    from pySurf.data2D import get_data, levellegendre

    rmsrange=[[None,0.1],[0.1,0.2],[0.2,1],[1,None]]  #list of frequency intervals for rms
    ax2f=[0,1,1,0,0]             #   axis where to plot rms corresponding to intervals in rmsrange: 1 right, 0 left
    wfun  = np.hanning           #   type of windows for fourier transform
    units = ("mm","mm","$\mu$m") #   units of surface data from which PSD is calculated

    vrange_surf=([-0.5,0.5])  #color scale of surface map
    vrange_leg=([-0.05,0.05])   #color scale of legendre removed map
    prange=np.array((1e-8,1.e-3))#np.array((5e-8,1.e-5))  #color scale of 2d psd plot
    order=2                     #highest legendre order to remove (2=sag)

    TEST_PATH=os.path.join(os.path.dirname(os.path.realpath(__file__)),r'test\PSD\2dpsd')
    wf=os.path.join(TEST_PATH,r'171010_Secondary_Mandrel_C3_RefSub.dat')
    OUTFOLDER=os.path.join(TEST_PATH,'output')
    OUTFILE=fn_add_subfix(wf,'_4leg',strip=True,pre=OUTFOLDER+os.path.sep)

    wdata,x,y=get_data(wf,matrix=True,addaxis=True)
    fs,ps=psd2d_analysis(levellegendre(y,wdata,order),x,y,outname=OUTFILE,wfun=np.hanning,
        rmsrange=rmsrange,prange=prange,ax2f=ax2f,units=units)

    f3,psd3=psd2d(levellegendre(y,wdata,order),x,y,wfun=np.hanning,norm=1,rmsnorm=True)

    plt.figure(1)
    plt.clf()
    plot_psd2d(f3,psd3,x,rmsrange=rmsrange)
    plt.figure(2)
    plt.clf()
    plot_psd2d(f3,psd3,x,rmsrange=rmsrange,prange=prange)

    """
    plt.figure()
    plt.title('2D PSD')
    plt.imshow(psd3,norm=LogNorm(),interpolation='none',
        extent=(x[0],x[-1],f3[0],f3[-1]),origin='lower',aspect='auto')
    plt.colorbar()
    """

    plt.figure(3)
    plt.clf()
    plt.title('2D PSD')
    plt.imshow(psd3,norm=LogNorm(vmin=prange[0],vmax=prange[1]),interpolation='none',
        extent=(x[0],x[-1],f3[0],f3[-1]),origin='lower',aspect='auto')
    plt.colorbar()


def mwc(psd3,prange):
    """plots a test case"""
    plt.title(str(prange))
    plt.imshow(psd3,norm=LogNorm(vmin=prange[0],vmax=prange[1]),interpolation='none',
        origin='lower',aspect='equal')
    from matplotlib.ticker import LogFormatter
    formatter = LogFormatter(10, labelOnlyBase=False)
    plt.colorbar(format=formatter)  #ticks=[1,5,10,20,50],

    #array([  5.03059356e-13,   4.11978017e-01])

def test_prange_mwc():
    from plotting.multiplots import compare_images,subplot_grid

    ptest=[(1e-9,0.1),(1e-11,0.1),(1e-13,0.1)]
    datarange=np.array([  5.0e-13,   4e-01])
    random_sample=np.random.random_sample
    psd3=random_sample((100,50))*(datarange[1]-datarange[0])+datarange[0]
    plt.figure()

    for i,p in enumerate(ptest):
        plt.subplot (1, 3, i+1)
        mwc(psd3,p)
    #compare_images(*zip(psd3,x,p),fignum=0,titles=map(str,ptest))
    plt.tight_layout()

def test_prange_mwc2():
    from plotting.multiplots import compare_images,subplot_grid

    #make data
    datarange=np.array([  5.0e-13,   4e-01])
    random_sample=np.random.random_sample
    psd3=random_sample((100,50))*(datarange[1]-datarange[0])+datarange[0]

    ptest=[(1e-9,0.1),(1e-11,0.1),(1e-13,0.1)] #test values for color range
    plt.figure()
    for i,p in enumerate(ptest):
        plt.subplot (1, 3, i+1)
        plt.title(str(p))
        plt.imshow(psd3,norm=LogNorm(vmin=p[0],vmax=p[1]))
        plt.colorbar()
    plt.tight_layout()
