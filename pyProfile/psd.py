from pyProfile.profile import line, make_signal
import numpy as np
from dataIO.span import span
import matplotlib.pyplot as plt
#from pySurf.data2D import projection

def psd2prof(f,p,phase=None,N=None):
    """build a profile from PSD, if phase (in rad) is not passed use random 
    values. A real profile of npoints is assumed. Note that profiles with 
    (len(p)-1)*2 and (len(p)-1)*2+1 points give psd with same number of points,
    so both reconstructions are possible. Odd number of points is the default,
    passing N overrides, with N=(len(p)-1)*2, default (len(p)-1)*2+1"""
 
    if phase==None:
        phase=np.random.random(len(f))*np.pi*2
    
    
    #adjust norm of positive freq.
    p=p/2  
    p[0]=p[0]*2
    #add complex part
    #p[0]=p[0]/2
    yfft=np.sqrt(p)*np.exp(1j*phase)
    if N is None:
        N=(len(p)-1)*2+1
    else:
        if N!=(len(p)-1)*2 and N!=(len(p)-1)*2+1 :
            raise ValueError
    y=np.fft.irfft(yfft,N)
    y=y*len(y)
    x=np.arange(len(y))/(f[1]-f[0])/(len(y)-1)
    x=x-x[0]
    
    return x,y

def normPSD(N,L=None,form=1):
    """return a normalization factor for the PSD, calculated according to different definitions of `power`.
    The PSD is defined as PSD(f)=factor*np.abs(FFT)**2 where FFT is the fast fourier transform.
    Note that in case of rfft the doubling factor is not included in norm (because part of FFT definition).
    Possible values are (ref. to Numerical recipes when possible):
    1 - units: [Y**2] no normalization. "sum squared amplitude" in NR 13.4.1
    2 - units: [Y**2][X] This has the advantage that measurements on same range with different number of points match and it is the way it is usually plotted. The rms is the integral of PSD over frequency rather than the sum.
    3 - units: [Y**2] here rms is the sum of PSD, this is RA normalization (however it differs in RA when a window is used). 13.4.5 in NR.
    4 - units: [Y**2] not sure what this is for, "mean squared amplitude" in NR 13.4.2 and in formula 12.1.10 as discrete form of Parsifal's theorem.
    """
        
    if form==0:
        factor=1. #units: [Y**2] no normalization. "sum squared amplitude" in NR 13.4.1
    if form==1:
        factor=1./N**2*L #[Y**2][X] This has the advantage that measurements on same range with different number of points match
        #and it is the way it is usually plotted. The rms is the integral of PSD over frequency rather than the sum.
    elif form==2:
        factor=1./N**2 #[Y**2] here rms is the sum of PSD, this is RA normalization (however it differs in RA when a window is used).
        #"" 13.4.5 in NR
    elif form==3:
        factor=1./N #[Y**2] not sure what this is for, "mean squared amplitude" in NR 13.4.2
                    #and in formula 12.1.10 as discrete form of Parsifal's theorem
    return factor
    
def psd(x,y,retall=False,wfun=None,norm=1,rmsnorm=False):
    """return frequencies and PSD of a profile.
    
    If retall is set, also the phase is returned.
    PSD is squared FFT divided by step size. Profile is assumed real, so only positive freqs are used, doubling PSD values. Return value has ceil(N/2) elements (always odd).
    See `np.fft.rfftfreq` for details.
    wfun is a function that given the number of points return a vector with values for the window. Note that normalization is quite arbitrary, usually it's rescaled later with rmsnorm.
    2017/07/30 if rmsnorm is set True, PSD is normalized to (multiplied by) rms calculated from profile (must be same if no window applied). 
    2016/03/26 this psd i return to old normalizaiton "time-integral squared magnitude", multiplying by dx.
    """
    
    if wfun is None:
        win=np.ones(len(y))
    else:
        win=wfun(len(x))
    N = len(y)
    L=span(x,True)
    
    yfft  = np.fft.rfft(y*win) #note that rfft return exact values as fft (needs to be doubled after squaring amplitude)
    
    normfactor=normPSD(N,L,form=norm)
    if rmsnorm:  #normalize to rms of non windowed function. This is independent on 0 constant term, 
    # but also alter its meaning 
        normfactor=normfactor*np.nanstd(y)**2/np.nanstd(y*win)**2
   
    psd  = 2*normfactor*np.abs(yfft)**2
    psd[0]=psd[0]/2  #the first component is not doubled
    
    freqs=np.fft.rfftfreq(N,np.float(L)/(N-1))
    if retall:
        return freqs,psd,np.angle(yfft)
    else:
        return freqs,psd         
        
def psd_units(units=[None,None,None]):
    """return as a list of 3 strings (or None) for `units` of x,f,PSD
    from units of x,y,data.
    In case there is not enough information on units of data, 
        returned units are '', '[Y]', '[Y] [Z]$^2$'.
    
    This aims to keep some consistency in handling units, because
    of the risk of ambiguity, e.g. m^2 vs m**2 or m$^2$,
    or even None vs "".
    This ambiguity is also the reason why in object version `Data2D_class.PS2D` 
    units are kept from surface data, rather than being set to PSD units.
    This is probably not the best way to solve the ambiguity, but this
    function should be the preferred way to generate units for the 
    PSD and its axis, until a better one is found. 
    X is left blank rather than outputting '[X]' because it is identical
        to input units and it can be inconvenient to use the string
        in plots (e.g. "x ([X])") and the original units can be accessed
        if desired.
        
    2020/07/16 moved to `pyProfile.psd` from `pySurf.psd2d`, making it valid for 2 or 3D units."""
    
    import copy
    if units is not None:
        units = units.copy() #otherwise bizarre side effect
    flag2d = False  #flag to return 2-el vector
    if np.size(units) == 1: 
        units = np.repeat(units,3)
    elif np.size(units) == 2:
        flag2d=True
        units=[None,units[0],units[1]]

    units[1] = (units[1] if units[1] else "[Y]")+"$^{-1}$"
    if units[0] is None: units[0] = "" # cbunits[0] = units[0] if units[0] else ""
    if units[2] and units[1]:
        units[2] = units[1]+" "+units[2]+"$^2$"  #colorbar units string
    else:
        units[2]=(units[1] if units[1] else "[Y]")+" [Z]$^2$"
        #units[2]="[Y] [Z]$^2$"
    
    if flag2d: units=units[1:3]
    return units
    
def test_psd_units():
    """run tests with different combinations of defined and undefined units.
    """
    
    testval = [[None,None],
               ['mm',None],
               ['mm',''],
               ['mm','mm'], #the following fails:
               None,  
               'mm',
               ['mm']]
    
    print ('TEST FOR 2D UNITS (PROFILE)')
    for t in testval:
        print ("test for: ",t)
        print ('PSD units(x,f,PSD):\n'+'\n'.join(psd_units(t)))
        print ('----------\n')

    testval = [[None,None,None],
               ['mm',None,None],
               ['mm',None,''],
               ['mm','',''],
               ['mm','mm',''],
               ['mm','mm','mm'],  #the following fails:
               None,  
               'mm',
               ['mm']]
               
    print ('TEST FOR 3D UNITS (SURFACE)')
    for t in testval:
        print ("test for: ",t)
        print ('PSD units(x,f,PSD):\n'+'\n'.join(psd_units(t)))
        print ('----------\n')

def components(d,win=1):
    """RA Want to return Fourier components with optional window
    Application note: These components are dependent on sampling!
    This means you can *not* interpolate these components onto other
    frequency grids!
    """
    #Handle window
    if win is not 1:
        if np.size(np.shape(d)) is 1:
            win = win(np.size(d))/np.sqrt(np.mean(win(np.size(d))**2))
        else:
            win1 = win(np.shape(d)[0])
            win2 = win(np.shape(d)[1])
            win = np.outer(win1,win2)
            win = win/np.sqrt(np.mean(win**2))

    #Compute Fourier components
    return np.fft.fftn(d*win)/np.size(d)
    
def realPSD(d0,win=np.hanning,dx=1.,axis=None,nans=False,minpx=10):
    """(Ryan Allured) This function returns the PSD of a real function
    Gets rid of zero frequency and puts all power in positive frequencies
    Returns only positive frequencies
    """
    from utilities.imaging.man import stripnans
    if nans is True:
        d = stripnans(d0)
    else:
        d = d0
    if len(d) < minpx:
        return np.nan
    #Get Fourier components
    c = components(d,win=win)
    #Handle collapsing to 1D PSD if axis keyword is set
    if axis==0:
        c = c[:,0]
    elif axis==1:
        c = c[0,:]

    #Reform into PSD
    if np.size(np.shape(c)) is 2:
        f = [np.fft.fftfreq(np.shape(c)[0],d=dx)[:np.shape(c)[0]/2],\
                   np.fft.fftfreq(np.shape(c)[1],d=dx)[:np.shape(c)[1]/2]]
        c = c[:np.shape(c)[0]/2,:np.shape(c)[1]/2]
        c[0,0] = 0.
        #Handle normalization
        c = 2*c
        c[0,:] = c[0,:]/np.sqrt(2.)
        c[:,0] = c[:,0]/np.sqrt(2.)
        
    elif np.size(np.shape(c)) is 1:
        f = np.fft.fftfreq(np.size(c),d=dx)
        f = f[:np.size(c)/2]
        c = c[:np.size(c)/2]
        c[0] = 0.
        c = c*np.sqrt(2.)

    return f[1:],np.abs(c[1:])**2

   
def xrealPSD(d,win=1,dx=1.,axis=None):
    """RA This function returns the PSD of a real function
    Gets rid of zero frequency and puts all power in positive frequencies
    Returns only positive frequencies
    """
    #Get Fourier components
    c = components(d,win=win)  #np.fft.fftn(d*win)/np.size(d)
    #Handle collapsing to 1D PSD if axis keyword is set
    if axis==0:
        c = c[:,0]
    elif axis==1:
        c = c[0,:]

    #Reform into PSD
    if np.size(np.shape(c)) is 2:
        f = [np.fft.fftfreq(np.shape(c)[0],d=dx)[:np.shape(c)[0]/2],\
                   np.fft.fftfreq(np.shape(c)[1],d=dx)[:np.shape(c)[1]/2]]
        c = c[:np.shape(c)[0]/2,:np.shape(c)[1]/2]
        c[0,0] = 0.
        #Handle normalization
        c = 2*c
        c[0,:] = c[0,:]/np.sqrt(2.)
        c[:,0] = c[:,0]/np.sqrt(2.)
        
    elif np.size(np.shape(c)) is 1:
        f = np.fft.fftfreq(np.size(c),d=dx)
        f = f[:np.size(c)/2]
        c = c[:np.size(c)/2]
        c[0] = 0.
        c = c*np.sqrt(2.)

    return f,np.abs(c)**2    
import pdb

def plot_psd(f,p,units=None,label=None,span=0,psdrange=None,
    includezerofreq=False,*args,**kwargs):
    """Plot a PSD on logaritmic axis with standard labels.
    Units can be passed as profile units (2-el) or x,y,z (3-el, only last two are used). 
    If span is set, plots psd
    range according to span settings of data2D.projection."""
    #Note: title is not implemented because it can be a suptitle, so it is left to set
    # to caller routine. Take care to manage multiple plotting as list or successive calls.
    #this gets very tricky to do
    

    if label is not None:
        l = label
    else:
        l = ""
    
    if not includezerofreq:
        if f[0]==0:
            f=f[1:]
            p=p[1:,...]
            
    if units is None: units=['[X]','[Y]','[Z]']
    if len(units)==2: units = [None,units[0],units[1]]
    #print ("FIX THIS ROUTINE BEFORE USING IT, see newview_plotter")
    plt.ylabel('axial PSD ('+units[2]+'$^2$ '+units[1]+')')
    plt.xlabel('Freq. ('+units[1]+'$^{-1}$)')
    #pdb.set_trace()
    if len(p.shape)==2: #experimentally deals with 2D array automatically plotting average
        ps=projection(p,axis=1,span=span)
        if span:
            plt.plot(f,p[0],label=l + 'AVG')
            plt.plot(f,p[1],label=l + 'point-wise min')
            plt.plot(f,p[2],label=l + 'point-wise max')
        else:
            plt.plot(f,p,label=l + 'AVG')
    else:
        assert len(p.shape)==1
        plt.plot(f,p,label=l,*args,**kwargs)
    
    if psdrange is not None:
        if len(psdrange)==1: psdrange=np.repeat(psdrange,2)
        plt.ylim(psdrange)
    plt.loglog()
    plt.grid(1)
    plt.legend(loc=0) 
    #plt.suptitle('%s: rms=%5.3f 10$^{-3}$'%(label,rms*1000)+units[2])

from matplotlib import pyplot as plt
import numpy as np
from cycler import cycler
from collections import OrderedDict

def make_psd_plots(toplot,units=None,outfile=None): 

    """This was in makenicePSDplots.make_plots, needs to be integrated with psd_plot.
    
    toplot is a dictionary (or collections.OrderedDict if order is important) with filenames as keys and labels as values.
    psd is read from each file on the first two columns and plotted."""
    
    plt.figure()
    # 1. Setting prop cycle on default rc parameter
    plt.rc('lines', linewidth=1.5)
    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':', '-.'])* cycler('color', ['r', 'g', 'b', 'y']) ))
       
    for ff,ll in toplot.items():    
        f,p=np.genfromtxt(ff,delimiter='',unpack=True,usecols=[0,1])
        plot_psd(f,p,label=ll,units=units)
        """
        plt.plot(f,p,label=ll)
        plt.loglog()
        plt.xlabel('freq. (mm$^-1$)')
        plt.ylabel('mm $\mu$m$^2$')
        plt.grid(1)
        plt.legend(loc=0)
        plt.show()
        """
        
    if outfile is not None:
        plt.savefig(outfile)
        print("results saved in %s"%outfile)
        #plt.savefig(fn_add_subfix(ff,'_psdcomp','.png'))

    
def linearTrend(x,y): 
    """test linear trend"""
    yp=y-np.mean(y)
    yt=y-line(x,y)
    
    plt.clf()
    for yy,ll in zip([y,yp,yt],['signal','avg offset','line removed']):
        plot_sig_psd(x,yy,label=ll)
    for ax in plt.gcf().axes:
        plt.sca(ax)
        plt.legend(loc=0)

class PSDplot(object):
    """Contains a number of psds in form (f,psd)."""
    def __init__(self):
        self.psds=[]
        #self.labels=[]
        self.kwargs=[]
        
    def append(self,psd):
        self.psds.append(psd)
        
    def append_from_file(self,file,cols,**kwargs):
        self.psds.append(np.genfromtxt(file,skip_header=1,unpack=1)[[cols[0],cols[1]]])       
        self.kwargs.append(kwargs)
        #self.labels.append(label)
        
        
    def plot(self,*args):
        plt.figure()
        for i,(yfreq1,ypsd1) in enumerate(self.psds):
            #label=kwargs.pop('label',None)
            plt.plot(yfreq1,ypsd1,*args,**(self.kwargs[i]))
        plt.loglog()
        plt.grid(1)
        plt.legend(loc=0)
        

def addPSDs(psdfile,cols,*args,**kwargs):
    """Add to current plot a list of PSDs from a standard (12-col,1 line header) PSD file.
    """
    
    assert len(cols)==2
    
    for psdfile1,cols1 in zip(psdfiles,cols):
        yfreq1,ypsd1=np.genfromtxt(psdfile1,skip_header=1,unpack=1)[cols[0],cols[1]]
    
    #plt.figure()
    plt.plot(yfreq1,ypsd1,*args,**kwargs)
    #plt.loglog()
    #plt.grid()
    #if title is None:
    #    title=os.path.basename(psdfile1)
    #plt.title(title)
    plt.legend(loc=0)
    if outfolder:
        plt.savefig(os.path.join(outfolder,title+'.png'))    

def plotPSDs(psdfile1,psdfile2,title=None,outfolder=None):
    """Plot a PSD from a standard (12-col,1 line header) PSD file."""
    
    yfreq1,ypsd1,xfreq1,xpsd1=np.genfromtxt(psdfile1,skip_header=1,unpack=1)[8:]
    yfreq2,ypsd2,xfreq2,xpsd2=np.genfromtxt(psdfile2,skip_header=1,unpack=1)[8:]
    
    plt.figure()
    plt.plot(yfreq1,ypsd1,'b',label='before annealing')
    plt.plot(xfreq1,xpsd1,'b')
    plt.plot(yfreq2,ypsd2,'r',label='after annealing')
    plt.plot(xfreq2,xpsd2,'r')
    plt.loglog()
    plt.grid()
    plt.ylabel('(mm $\mu$m^2)')
    plt.xlabel('mm^-1')
    if title is None:
        title=os.path.basename(psdfile2)
    plt.title(title)
    plt.legend(loc=0)
    if outfolder:
        plt.savefig(os.path.join(outfolder,title+'.png'))           


def FFTtransform(x,y):
    yfft  = np.fft.rfft(y) #note that rfft return exact values as fft (needs to be doubled after squaring amplitude)
    N = len(y)
    L=span(x,True)  
    f = np.fft.rfftfreq(N,L/N)
    return f,yfft

def plot_transform(f,yfft,label="",euler=False,units=None,includezerofreq=False,**kwargs):
    """units is a 2 element vector."""
    #pdb.set_trace()
    ax3 = plt.gca()
    ##PLOT TRANSFORM AS EULER REPRESENTATION  
    unitstr = psd_units(units) 
    
    if not includezerofreq:
        if f[0]==0:
            f=f[1:]
            yfft=yfft[1:,...]
            
    if euler:
        plt.title('DFT - Module and phase')
        plt.plot(f,np.abs(yfft),label=label,**kwargs)
        #plt.loglog() 
        ax3.legend(loc=2)
        #string for x is set to '' as default:
        plt.xlabel('Freq ('+(unitstr[0] if unitstr[0] else '[X]')+'$^-1$)')
        plt.ylabel('Module'+((' ('+unitstr[1]) if unitstr[1] else '[Y]')+'$^2$)')
            
        ax3b = ax3.twinx()
        ax3.twin=ax3b
        ax3b.plot(f,np.angle(yfft)/np.pi,'--',label=label+' Phase',
        **kwargs)
        ax3b.set_ylabel('Phase /$\pi$')
        ax3b.set_ylim(-1.,1.)
        plt.grid(1)
        ax3b.legend(loc=1)
        
    ##PLOT TRANSFORM AS COMPLEX NUMBER
    else:
        #pdb.set_trace()
        plt.title('DFT - Real and imaginary parts')
        ax3.plot(f,np.real(yfft),label=label+' Re',**kwargs)
        #plt.loglog() 
        s = span (np.real(yfft))
        plt.ylim([s[0],s[1]]) # needed to expland scale if very small
        #ax3.set_xlabel(unitstr[0]+'$^-1$')
        #string for x is set to '' as default:
        plt.xlabel('Freq. ('+(unitstr[0] if unitstr[0] else '[X]')+'$^-1$)')
        ax3.set_ylabel('Re')
        #for tl in ax2.get_yticklabels():
            #tl.set_color('b')
        ax3.legend(loc=2)
            
        ax2b = ax3.twinx()
        ax2b.plot(f,np.imag(yfft),'--',label=label+' Im',**kwargs)
        s = span (np.imag(yfft))
        plt.ylim([s[0],s[1]]) # needed to expland scale if very small
        ax2b.set_ylabel('Im')
        #for tl in ax2b.get_yticklabels():
            #tl.set_color('r')
        plt.grid(True)
        ax2b.legend(loc=1)
        ax3.twin=ax2b
    plt.tight_layout

def xplot_sig_psd(x,y,label="",**kwargs):
    """plot signal, fft and psd on three panels.
    dft sets how the fft is plotted 
    'realim': real and im parts,
    'euler' argument and phase
    'both' uses two panels (four in total).
    """ 
    N = len(y)
    L=span(x,True)    
        
    ax1=plt.subplot(211)
    plt.title('Signal')
    plt.plot(x,y,**kwargs)
    ax1.set_xlabel('mm')
    plt.grid(1)
    
    ax4=plt.subplot(212) 
    plt.title('PSD')
    f,PSD=psd(x,y)
    plt.plot(f,PSD,label=label,**kwargs)
    plt.loglog()     
    ax4.set_xlabel('mm^-1')
    plt.grid(1)
    plt.show()
        
    return ax1,ax4
    
def plot_sig_psd(x,y,label="",realim=False,
    euler=False,power=True,includezerofreq=False,
    yrange=None,prange=None,fignum=None,
    aspect='auto', units=None,outname=None,norm=1,rmsnorm=True,**kwargs):
    """plot signal, fft and psd on n panels according to flags, return axis.
    for axis with a twin axis, the twin axis is appended to the first in a property 'twin'
    parameter for plot function (styles etc., common to all plots) 
        can be passed as kwargs
    plots signal and  additional plots according to respective flags:
    'realim': real and im parts,
    'euler' argument and phase
    'power' plot PSD calling psd with default parameters
    """ 
    #can be done in more advanced way with ax.change_geometry(numrows, numcols, num)
    nsp=1+np.count_nonzero([realim,euler,psd]) #number of subplots
    i=1 #index of subplot
    axes=[]
    
    #ax1 SIGNAL
    axes.append(plt.subplot(nsp,1,i))
    i=i+1
    plt.title('Signal') 
    plt.plot(x,y,**kwargs)
    plt.xlabel('mm')
    plt.grid(1)
    
    f,yfft =  FFTtransform(x,y)
    
    '''
    ##calculate
    N = len(y)
    L=span(x,True)    
    #yfft  = np.fft.fft(y)
    #yfft  = yfft[:int(np.ceil(N/2))] #select positive part of spectrum
    yfft  = np.fft.rfft(y) #note that rfft return exact values as fft (needs to be doubled after squaring amplitude)
    #f = np.fft.fftfreq(N,L/N)
    #f = f[:int(np.ceil(N/2))]
    f = np.fft.rfftfreq(N,L/N)
    '''
    
    '''
    The default normalization has the direct transforms unscaled and the inverse transforms are scaled by 1/n. It is possible to obtain unitary transforms by setting the keyword argument norm to "ortho" (default is None) so that both direct and inverse transforms will be scaled by 1/\sqrt{n}.
    
    N = len(y)
    L=span(x,True)
    #form = 1 --> normfactor = 1./N**2*L
    
    yfft  = np.fft.rfft(y*win) #note that rfft return exact values as fft (needs to be doubled after squaring amplitude)
    
    normfactor=normPSD(N,L,form=norm)
    
    if rmsnorm:  #normalize to rms of non windowed function. This is independent on 0 constant term, 
    # but also alter its meaning 
        normfactor=normfactor*np.nanstd(y)**2/np.nanstd(y*win)**2
    
    psd  = 2*normfactor*np.abs(yfft)**2
    psd[0]=psd[0]/2  #the first component is not doubled
    
    freqs=np.fft.rfftfreq(N,np.float(L)/(N-1))
    '''
        
    if realim:
        ax2=plt.subplot(nsp,1,i)
        axes.append(ax2)
        i=i+1
        plot_transform(f,yfft,label=label,includezerofreq=includezerofreq,**kwargs)
        #for tl in ax2.get_yticklabels():
            #tl.set_color('b')
        
    if euler:
        ax3=plt.subplot(nsp,1,i)  
        axes.append(ax3)
        i=i+1        
        plot_transform(f,yfft,label=label,includezerofreq=includezerofreq,euler=True,**kwargs)
    
    ##PLOT PSD
    if power:
        ax = plt.subplot(nsp,1,i)
        axes.append(ax)
        i=i+1
        plt.title('PSD')
        f,PSD=psd(x,y,**kwargs)
        plt.plot(f,PSD,label=label,**kwargs)
        plt.loglog()     
        plt.xlabel('mm^-1')
        plt.grid(1)
        plt.ylim(prange)
    
    plt.tight_layout()
    #plt.show()
    return axes
    
def plot_sig_psd4(x,y,scale=1.0,label="",**kwargs):
    """ wrapper around`plot_sig_psd4`
    plot signal, fft and psd on three panels.
    like `plot_sig_psd`, with fixed number of variables.
    
    ?? 
    dft sets how the fft is plotted 
    'realim': real and im parts,
    'euler' argument and phase
    'both' uses two panels (four in total).
    """ 
    
    print ("WARNING: `plot_sig_psd4` was replaced with `plot_sig_psd`, update code!")
    return plot_sig_psd(x,y,label=label,realim=True,
    power=True,**kwargs)
'''        
def plot_sig_psd4(x,y,scale=1.0,label="",**kwargs):
    #replaced by the new plot_sig_psd
    """plot signal, fft and psd on three panels.
    like `plot_sig_psd`, with fixed number of variables.
    
    ?? 
    dft sets how the fft is plotted 
    'realim': real and im parts,
    'euler' argument and phase
    'both' uses two panels (four in total).
    """ 
    N = len(y)
    L=span(x,True)    
        
    ax1=plt.subplot(411)
    plt.title('Signal')
    plt.plot(x,y,**kwargs)
    ax1.set_xlabel('mm')
    plt.grid(1)

    ax2=plt.subplot(412) 
    yfft  = np.fft.fft(y)
    yfft  = yfft[:np.ceil(N/2)]
    f = np.fft.fftfreq(N,L/N)
    f = f[:np.ceil(N/2)]
    
    plt.title('DFT - Real and imaginary parts')
    ax2.set_ylabel('time (s)')
    ax2.plot(f,np.real(yfft),label=label+' Re',**kwargs)
    #plt.loglog() 
    ax2.set_xlabel('mm^-1')
    ax2.set_ylabel('Re')
    #for tl in ax2.get_yticklabels():
        #tl.set_color('b')
    plt.legend(loc=0)
		
    ax2b = ax2.twinx()
    ax2b.plot(f,np.imag(yfft),'--',label=label+' Im',**kwargs)
    ax2b.set_ylabel('Im')
	#for tl in ax2b.get_yticklabels():
		#tl.set_color('r')
    plt.grid(True)
    plt.legend(loc=0)
    
    ax3=plt.subplot(413)
    plt.title('DFT - Module and phase')
    ax3.set_ylabel('time (s)')
    ax3.plot(f,np.abs(yfft),label=label+' Norm',**kwargs)
    #plt.loglog() 
    plt.legend(loc=0)
    ax3.set_xlabel('mm^-1')
    ax3.set_ylabel('Norm')
    	
    ax3b = ax3.twinx()
    ax3b.plot(f,np.angle(yfft)/np.pi,'--',label=label+' Phase',
    **kwargs)
    ax3b.set_ylabel('Phase /$\pi$')
    ax3b.set_ylim(-1.,1.)
    plt.grid(1)
    plt.legend(loc=0)
    plt.show()
    
    ax4=plt.subplot(414) 
    plt.title('PSD')
    f,PSD=psd(x,y,scale=scale)
    plt.plot(f,PSD,label=label,**kwargs)
    plt.loglog()     
    ax4.set_xlabel('mm^-1')
    plt.grid(1)
    plt.show()
        
    return ax1,ax2,ax3,ax4
'''
  
def testPSDest(x,y): 
    """plot 3 panels with: test profile, PSD calculated with different functions 
    and PSD calculated by piecewise matplotlib function, with different settings.
    return 3 axis."""
    # 2016/04/04 corrected for updated normalization
    
    
    plt.clf()
    plt.suptitle('Comparison of PSD estimations')
    ax1=plt.subplot(311)
    plt.title('Signal=wave+line+noise')
    plt.plot(x,y)

    ax2=plt.subplot(312)
    plt.title('VC and RA routines')
    f,PSD=psd(x,y-line(x,y))
	#plt.plot(f,PSD/N/L,label='my PSD/N/L (line subtraction)')
    plt.plot(f,PSD,label='my PSD (line subtraction)')
    fw,PSDw=psd(x,y*np.hanning(len(y)))
	#plt.plot(fw,PSDw/N/L,label='my PSD windowed/N/L')
    plt.plot(fw,PSDw,label='my PSD windowed')
    fw,PSDw=psd(x,(y-line(x,y))*np.hanning(len(y)))
	#plt.plot(fw,PSDw/N/L,label='my PSD windowed/N/L (line subtraction)')
    plt.plot(fw,PSDw,label='my PSD windowed (line subtraction)')

    plt.plot(*realPSD(y*np.sqrt(N*L),win=1,dx=L/N),label='realPSD/df (RA)',linestyle='-.',color='r')    
    plt.plot(*realPSD(y*np.sqrt(N*L),dx=L/N),label='realPSD/df window (RA)',linestyle='-.',color='k')

    #plt.plot(*realPSD(y,dx=L/(N-1)),label='realPSD',linestyle='-.',color='r')    
    #plt.plot(*realPSD(y*np.hanning(len(y)),dx=L/(N-1)),label='realPSD - w',linestyle='-.',color='k')
    plt.loglog()
    plt.legend(loc=0)
    
    ax3=plt.subplot(313)
    from matplotlib.pyplot import psd as psdplot
    from matplotlib.mlab import window_none
    plt.title('Matplotlib piecewise estimation')
    detrending=['default' ,'constant' , 'mean' , 'linear' , 'none']
    for det in detrending:
        pp,ff=psdplot(y,Fs=N/float(L),detrend=det,window=window_none,scale_by_freq=True)
    plt.legend(detrending,loc=0)
    plt.loglog()
    plt.show()
    return ax1,ax2,ax3


def test_psd_normalization(x,y,wfun=None,norm=1,**kwargs):
    """Calculate PSD with a given normalization and
    compare its integral over frequency and sum to
    rms square and standard deviation (they differ for rms including also offset, while stddev being referred to mean."""
    
    f,p=psd(x,y,wfun=wfun,norm=norm,**kwargs)
    
    print ("== Quantities calculated from profile ==")
    print ("Profile Height PV %6.3g (min: %6.3g, max: %6.3g)"%
           (np.nanmax(y)-np.nanmin(y),np.nanmin(y),np.nanmax(y)))
    print ("Profile Height avg.: ",np.nanmean(y))
    print ("devstd**2=",np.std(y)**2, '(devstd=%f5.3)'%np.std(y))
    print ("rms**2=",(y**2/len(x)).sum())  #, '(rms=%f5.3)'%np.std(y)
    print ("\n== Quantities calculated from PSD ==")
    print ("sum of PSD is ",p[1:].sum())
    print ("integral of PSD (as sum*deltaf) is ",p[1:].sum()/span(x,1)) #span(x,size=1)=1/L
    print ("integral trapz: ",np.trapz(p[1:],f[1:]))
    print ("psd[0]*deltaf=%f (integral including:%f)"%(p[0]*f[1],p.sum()/span(x,1)))
    print ("#--------\n")
    
    return f,p          
    
        
if __name__=="__main__":
    N=500
    noise=2.
    amp=10.
    L=30.
    nwaves=5.8 #32
    ystartend=(3,20)
    #ystartend=(0,0)
    
    x,y=make_signal(amp,L=L,N=N,nwaves=nwaves,ystartend=ystarted,noise=noise)
    ax1,ax2,ax3=testPSDest(x,y)   
    
    plt.sca(ax2)
    plt.axvline(nwaves/L)    
    plt.sca(ax3)
    plt.axvline(nwaves/L)  
    plt.draw()
    for nn in range(1,3):
        print("norm type= %s"%nn)
        test_psd_normalization(x,y,norm=nn)
 
