import numpy as np
import pdb
from scipy.interpolate import griddata
from imaging.man import stripnans,nearestNaN
from scipy.integrate import simps

#This module contains Fourier analysis routine

def components(d,win=np.hanning):
    """Want to return Fourier components with optional window
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

def continuousComponents(d,dx,win=np.hanning):
    """Want to return Fourier components with optional window
    Divide by frequency interval to convert to continuous FFT
    These components can be safely interpolated onto other frequency
    grids. Multiply by new frequency interval to get to numpy format
    FFT. Frequency units *must* be the same in each case.
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
    return np.fft.fftn(d*win)*dx

def newFreq(f,p,nf):
    """
    Interpolate a power spectrum onto a new frequency grid.
    """
    return griddata(f,p,nf,method=method)

def freqgrid(d,dx=1.):
    """Return a frequency grid to match FFT components
    """
    freqx = np.fft.fftfreq(np.shape(d)[1],d=dx)
    freqy = np.fft.fftfreq(np.shape(d)[0],d=dx)
    freqx,freqy = np.meshgrid(freqx,freqy)
    return freqx,freqy

def ellipsoidalHighFrequencyCutoff(d,fxmax,fymax,dx=1.,win=np.hanning):
    """A simple low-pass filter with a high frequency cutoff.
    The cutoff boundary is an ellipsoid in frequency space.
    All frequency components with (fx/fxmax)**2+(fy/fymax)**2 > 1.
    are eliminated.
    fxmax refers to the second index, fymax refers to the first index
    This is consistent with indices in imshow
    """
    #FFT components in numpy format
    fftcomp = components(d,win=win)*np.size(d)

    #Get frequencies
    freqx,freqy = freqgrid(d,dx=dx)

    #Get indices of frequencies violating cutoff
    ind = (freqx/fxmax)**2+(freqy/fymax)**2 > 1.
    fftcomp[ind] = 0.

    #Invert the FFT and return the filtered image
    return fft.ifftn(fftcomp)

def meanPSD(d0,win=np.hanning,dx=1.,axis=0,irregular=False,returnInd=False,minpx=10):
    """Return the 1D PSD averaged over a surface.
    Axis indicates the axis over which to FFT
    If irregular is True, each slice will be stripped
    and then the power spectra
    interpolated to common frequency grid
    Presume image has already been interpolated internally
    If returnInd is true, return array of power spectra
    Ignores slices with less than minpx non-nans
    """
    #Handle which axis is transformed
    if axis==0:
        d0 = np.transpose(d0)
    #Create list of slices
    if irregular is True:
        d0 = [stripnans(di) for di in d0]
    else:
        d0 = [di for di in d0]
    #Create power spectra from each slice
    pows = [realPSD(s,win=win,dx=dx,minpx=minpx) for s in d0 \
            if np.sum(~np.isnan(s)) >= minpx]
    #Interpolate onto common frequency grid of shortest slice
    if irregular is True:
        #Determine smallest frequency grid
        ln = [len(s[0]) for s in pows]
        freq = pows[np.argmin(ln)][0]
        #Interpolate
        pp = [griddata(p[0],p[1],freq) for p in pows]
    else:
        pp = [p[1] for p in pows]
        freq = pows[0][0]
    #Average
    pa = np.mean(pp,axis=0)
    if returnInd is True:
        return freq,pp
    return freq,pa

def medianPSD(d0,win=np.hanning,dx=1.,axis=0,nans=False):
    """Return the 1D PSD "medianed" over a surface.
    Axis indicates the axis over which to FFT
    If nans is True, each slice will be stripped,
    internally interpolated, and then the power spectra
    interpolated to common frequency grid"""
    d = stripnans(d0)
    if win is not 1:
        win = win(np.shape(d)[axis])/\
              np.sqrt(np.mean(win(np.shape(d)[axis])**2))
        win = np.repeat(win,np.shape(d)[axis-1])
        win = np.reshape(win,(np.shape(d)[axis],np.shape(d)[axis-1]))
        if axis is 1:
            win = np.transpose(win)
    c = np.abs(np.fft.fft(d*win,axis=axis)/np.shape(d)[axis])**2
    c = np.median(c,axis=axis-1)
    f = np.fft.fftfreq(np.size(c),d=dx)
    f = f[:np.size(c)/2]
    c = c[:np.size(c)/2]
    c[1:] = 2*c[1:]
    return f,c

def realPSD(d0,win=np.hanning,dx=1.,axis=None,nans=False,minpx=10):
    """This function returns the PSD of a real function
    Gets rid of zero frequency and puts all power in positive frequencies
    Returns only positive frequencies
    """
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

def computeFreqBand(f,p,f1,f2,df,method='linear'):
    """
    Compute the power in the PSD between f1 and f2.
    f and p should be as returned by realPSD or meanPSD
    Interpolate between f1 and f2 with size df
    Then use numerical integration
    """
    newf = np.linspace(f1,f2,(f2-f1)/df+1)
    try:
        newp = griddata(f,p/f[0],newf,method=method)
    except:
        pdb.set_trace()
    return np.sqrt(simps(newp,x=newf))

def fftComputeFreqBand(d,f1,f2,df,dx=1.,win=np.hanning,nans=False,minpx=10,\
                       method='linear'):
    """
    Wrapper to take the FFT and immediately return the
    power between f1 and f2 of a slice
    If slice length is < 10, return nan
    """
    if np.sum(~np.isnan(d)) < minpx:
        return np.nan
    f,p = realPSD(d,dx=dx,win=win,nans=nans)
    return computeFreqBand(f,p,f1,f2,df,method=method)

def psdScan(d,f1,f2,df,N,axis=0,dx=1.,win=np.hanning,nans=False,minpx=10):
    """
    Take a running slice of length N and compute band limited
    power over the entire image. Resulting power array will be
    of shape (S1-N,S2) if axis is 0
    axis is which axis to FFT over
    """
    if axis is 0:
        d = np.transpose(d)
    sh = np.shape(d)
    m = np.array([[fftComputeFreqBand(di[i:i+N],f1,f2,df,dx=dx,win=win,nans=nans,minpx=minpx) \
      for i in range(sh[1]-N)] for di in d])
    if axis is 0:
        m = np.transpose(m)
    return m
    

def lowpass(d,dx,fcut):
    """Apply a low pass filter to a 1 or 2 dimensional array.
    Supply the bin size and the cutoff frequency in the same units.
    """
    #Get shape of array
    sh = np.shape(d)
    #Take FFT and form frequency arrays
    f = np.fft.fftn(d)
    if np.size(np.shape(d)) > 1:
        fx = np.fft.fftfreq(sh[0],d=dx)
        fy = np.fft.fftfreq(sh[1],d=dx)
        fa = np.meshgrid(fy,fx)
        fr = np.sqrt(fa[0]**2+fa[1]**2)
    else:
        fr = np.fft.fftfreq(sh[0],d=dx)
    #Apply cutoff
    f[fr>fcut] = 0.
    #Inverse FFT
    filtered = np.fft.ifftn(f)
    return filtered

def randomizePh(d):
    """Create a randomized phase array that maintains a real
    inverse Fourier transform. This requires that F(-w1,-w2)=F*(w1,w2)
    """
    #Initialize random phase array
    sh = np.shape(d)
    ph = np.zeros(sh,dtype='complex')+1.
    
    #Handle 1D case first
    if np.size(sh) == 1:
        if np.size(d) % 2 == 0:        
            ph[1:sh[0]/2] = np.exp(1j*np.random.rand(sh[0]/2-1)*2*np.pi)
            ph[sh[0]/2+1:] = np.conjugate(np.flipud(ph[1:sh[0]/2]))
        else:
            ph[1:sh[0]/2+1] = np.exp(1j*np.random.rand(sh[0]/2)*2*np.pi)
            ph[sh[0]/2+1:] = np.conjugate(np.flipud(ph[1:sh[0]/2+1]))
    else:
        #Handle zero frequency column/rows
        ph[:,0] = randomizePh(ph[:,0])
        ph[0,:] = randomizePh(ph[0,:])
        #Create quadrant
        if sh[0] % 2 == 0 and sh[1] % 2 == 0:
            #Handle intermediate Nyquist
            ph[sh[0]/2,:] = randomizePh(ph[sh[0]/2,:])
            ph[:,sh[1]/2] = randomizePh(ph[:,sh[1]/2])
            #Form quadrant
            ph[1:sh[0]/2,1:sh[1]/2] = \
                np.exp(1j*np.random.rand(sh[0]/2-1,sh[1]/2-1)*2*np.pi)
            ph[sh[0]/2+1:,sh[1]/2+1:] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2,1:sh[1]/2])))
            ph[1:sh[0]/2,sh[1]/2+1:] = \
                np.exp(1j*np.random.rand(sh[0]/2-1,sh[1]/2-1)*2*np.pi)
            ph[sh[0]/2+1:,1:sh[1]/2] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2,sh[1]/2+1:])))
        elif sh[0] % 2 == 0 and sh[1] % 2 == 1:
            #Handle intermediate Nyquist
            ph[sh[0]/2,:] = randomizePh(ph[sh[0]/2,:])
            #Form quadrant
            ph[1:sh[0]/2,1:sh[1]/2+1] = \
                np.exp(1j*np.random.rand(sh[0]/2-1,sh[1]/2)*2*np.pi)
            ph[sh[0]/2+1:,sh[1]/2+1:] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2,1:sh[1]/2+1])))
            ph[1:sh[0]/2,sh[1]/2+1:] = \
                np.exp(1j*np.random.rand(sh[0]/2-1,sh[1]/2)*2*np.pi)
            ph[sh[0]/2+1:,1:sh[1]/2+1] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2,sh[1]/2+1:])))
        elif sh[0] % 2 == 1 and sh[1] % 2 == 0:
            #Handle intermediate Nyquist
            ph[:,sh[1]/2] = randomizePh(ph[:,sh[1]/2])
            #Form quadrant
            ph[1:sh[0]/2+1,1:sh[1]/2] = \
                np.exp(1j*np.random.rand(sh[0]/2,sh[1]/2-1)*2*np.pi)
            ph[sh[0]/2+1:,sh[1]/2+1:] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2+1,1:sh[1]/2])))
            ph[1:sh[0]/2+1,sh[1]/2+1:] = \
                np.exp(1j*np.random.rand(sh[0]/2,sh[1]/2-1)*2*np.pi)
            ph[sh[0]/2+1:,1:sh[1]/2+1] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2+1,sh[1]/2:])))
        else:
            #Form quadrant
            ph[1:sh[0]/2+1,1:sh[1]/2+1] = \
                np.exp(1j*np.random.rand(sh[0]/2,sh[1]/2)*2*np.pi)
            ph[sh[0]/2+1:,sh[1]/2+1:] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2+1,1:sh[1]/2+1])))
            ph[1:sh[0]/2+1,sh[1]/2+1:] = \
                np.exp(1j*np.random.rand(sh[0]/2,sh[1]/2)*2*np.pi)
            ph[sh[0]/2+1:,1:sh[1]/2+1] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2+1,sh[1]/2+1:])))
            
        
##        if np.size(d) % 2 == 1:
##            ph[1:sh[0]/2] = np.random.rand(sh[0]/2-1)*2*np.pi
##            pdb.set_trace()
##            ph[sh[0]/2+1:] = np.conjugate(np.flipud(ph[1:sh[0]/2]))
##        else:
##            ph[1:(sh[0]-1)/2] = np.random.rand((sh[0]-1)/2-1)*2*np.pi
##            pdb.set_trace()
##            ph[(sh[0]+1)/2:] = np.conjugate(np.flipud(ph[1:(sh[0]-1)/2]))

            
##    #Fill in positive x frequencies with random phases
##    ind = freqx >= 0.
##    ph[ind] = np.exp(1j*np.random.rand(np.sum(ind))*2*np.pi)
##    #Fill in negative x frequencies with complex conjugates
##    ph[np.ceil(sh[0]/2.):,0] = np.conjugate(\
##        np.flipud(ph[:np.floor(sh[0]/2.),0]))
##    ph[0,np.ceil(sh[1]/2.):] = np.conjugate(\
##        np.flipud(ph[0,:np.floor(sh[1]/2.)]))

    return ph
    
def randomProfile(freq,psd):
    """
    Generate a random profile from an input PSD.
    freq should be in standard fft.fftfreq format
    psd should be symmetric as with a real signal
    sqrt(sum(psd)) will equal RMS of profile
    """
    amp = np.sqrt(psd)*len(freq)
    ph = randomizePh(amp)
    f = amp*ph
    sig = np.fft.ifft(f)
    return np.real(sig)
