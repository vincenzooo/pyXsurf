import numpy as np
import utilities.imaging.man as man
import utilities.imaging.fitting as fit
import scipy.ndimage as nd
from linecache import getline
import astropy.io.fits as pyfits
import pdb

def readCylScript(fn,rotate=np.linspace(.75,1.5,50),interp=None):
    """
    Load in data from 4D measurement of cylindrical mirror.
    File is assumed to have been saved with Ryan's scripting function.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Read in values from header
    f = open(fn+'.hdr','r')
    l = f.readlines()
    f.close()
    #Wavelength should never change
    wave = float(l[0].split()[0])*.001 #in microns
    #Ensure wedge factor is 0.5
    wedge = float(l[1])
    if wedge!=0.5:
        print ('Wedge factor is ' + str(wedge))
        pdb.set_trace()
    #Get pixel scale size
    dx = float(l[-1])

    #Remove NaNs and rescale
    d = np.fromfile(fn+'.bin',dtype=np.float32)
    try:
        d = d.reshape((1002,981))
    except:
        d = d.reshape((1003,982))
    d[d>1e10] = np.nan
    d = man.stripnans(d)
    d = d *wave
    d = d - np.nanmean(d)

    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        d = man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def readCyl4D(fn,rotate=np.linspace(.75,1.5,50),interp=None):
    """
    Load in data from 4D measurement of cylindrical mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Get xpix value in mm
    l = getline(fn,9)
    dx = float(l.split()[1])*1000.
    
    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)

    #d = np.rot90(d,k = 2)
    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        d = man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def readConic4D(fn,rotate=None,interp=None):
    """
    Load in data from 4D measurement of cylindrical mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Get xpix value in mm
    l = getline(fn,9)
    dx = float(l.split()[1])*1000.
    
    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)
    
    #d = np.rot90(d,k = 2)
    #Remove cylindrical misalignment terms
    conic_fit = fit.fitConic(d)
    d = d - conic_fit[0]

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        d = man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx,conic_fit[1]

def readFlatScript(fn,interp=None):
    """
    Load in data from 4D measurement of flat mirror.
    File is assumed to have been saved with Ryan's scripting function.
    Scale to microns, remove misalignments,
    strip NaNs.
    Distortion is bump positive looking at surface from 4D.
    Imshow will present distortion in proper orientation as if
    viewing the surface.
    """
    #Read in values from header
    f = open(fn+'.hdr','r')
    l = f.readlines()
    f.close()
    #Wavelength should never change
    wave = float(l[0].split()[0])*.001 #in microns
    #Ensure wedge factor is 0.5
    wedge = float(l[1])
    if wedge!=0.5:
        print ('Wedge factor is ' + str(wedge))
        pdb.set_trace()
    #Get pixel scale size
    dx = float(l[-1])

    #Remove NaNs and rescale
    d = np.fromfile(fn+'.bin',dtype=np.float32)
    try:
        d = d.reshape((1002,981))
    except:
        d = d.reshape((1003,982))
    d[d>1e10] = np.nan
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)
    d = np.fliplr(d)

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def readFlat4D(fn,interp=None):
    """
    Load in data from 4D measurement of flat mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    Distortion is bump positive looking at surface from 4D.
    Imshow will present distortion in proper orientation as if
    viewing the surface.
    """
    #Get xpix value in mm
    l = getline(fn,9)
    dx = float(l.split()[1])*1000.
    
    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)
    d = np.fliplr(d)

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def write4DFits(filename,img,dx,dx2=None):
    """
    Write processed 4D data into a FITS file.
    Axial pixel size is given by dx.
    Azimuthal pixel size is given by dx2 - default to none
    """
    hdr = pyfits.Header()
    hdr['DX'] = dx
    hdu = pyfits.PrimaryHDU(data=img,header=hdr)
    hdu.writeto(filename,clobber=True)
    return

def read4DFits(filename):
    """
    Write FITS file of processed 4D data.
    Returns img,dx in list
    """
    dx = pyfits.getval(filename,'DX')
    img = pyfits.getdata(filename)
    return [img,dx]

def readCylWFS(fn,rotate=np.linspace(.75,1.5,50),interp=None):
    """
    Load in data from WFS measurement of cylindrical mirror.
    Assumes that data was processed using processHAS, and loaded into
    a .fits file.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Remove NaNs and rescale
    d = pyfits.getdata(fn)
    d = man.stripnans(d)
    d = d - np.nanmean(d)

    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]
    
    # Negate to make bump positive and rotate to be consistent with looking at the part beamside.
    #d = -d
    d = -np.fliplr(d) #np.rot90(d,k = 2)
    
    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d

def readConicWFS(fn,interp=None):
    """
    Load in data from WFS measurement of cylindrical mirror.
    Assumes that data was processed using processHAS, and loaded into
    a .fits file.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    Returns the data with best fit conic removed, as well as the
    coefficients in the conic fit.
    """
    #Remove NaNs and rescale
    d = pyfits.getdata(fn)
    d = man.stripnans(d)
    d = d - np.nanmean(d)
    
    # Negate to make bump positive and rotate to be consistent with looking at the part beamside.
    #d = -d
    d = -np.fliplr(d) #np.rot90(d,k = 2)
    
    #Remove cylindrical misalignment terms
    conic_fit = fit.fitConic(d)
    d = d - conic_fit[0]
    
    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,conic_fit[1]

def readFlatWFS(fn,interp=None):
    """
    Load in data from WFS measurement of flat mirror.
    Assumes that data was processed using processHAS, and loaded into
    a .fits file.
    Scale to microns, strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Remove NaNs and rescale
    d = pyfits.getdata(fn)
    d = man.stripnans(d)
    d = -d
    
    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d

#Read in Zygo ASCII file
def readzygo(filename):
    #Open file
    f = open(filename,'r')

    #Read third line to get intensity shape
    for i in range(3):
        l = f.readline()
    l = l.split(' ')
    iwidth = int(l[2])
    iheight = int(l[3])

    #Read fourth line to get phase shape
    l = f.readline()
    l = l.split(' ')
    pwidth = int(l[2])
    pheight = int(l[3])

    #Read eighth line to get scale factors
    for i in range(4):
        l = f.readline()
    l = l.split(' ')
    scale = float(l[1])
    wave = float(l[2])
    o = float(l[4])
    latscale = float(l[6])

    #Read eleventh line to get phase resolution
    f.readline()
    f.readline()
    l = f.readline()
    l = l.split(' ')
    phaseres = l[0]
    if phaseres is 0:
        phaseres = 4096
    else:
        phaseres = 32768

    #Read through to first '#' to signify intensity
    while (l[0]!='#'):
        l = f.readline()

    #Read intensity array
    #If no intensity, l will be '#' below
    l = f.readline()
    while (l[0]!='#'):
        #Convert to array of floats
        l = np.array(l.split(' '))
        l = l[:-1].astype('float')
        #Merge into intensity array
        try:
            intensity = np.concatenate((intensity,l))
        except:
            intensity = l
        #Read next line
        l = f.readline()

    #Reshape into proper array
    try:
        intensity = np.reshape(intensity,(iheight,iwidth))
    except:
        intensity = np.nan

    #Read phase array
    l = f.readline()
    while (l!=''):
        #Convert to array of floats
        l = np.array(l.split(' '))
        l = l[:-1].astype('float')
        #Merge into intensity array
        try:
            phase = np.concatenate((phase,l))
        except:
            phase = l
        #Read next line
        l = f.readline()

    phase = np.reshape(phase,(pheight,pwidth))
    phase[np.where(phase==phase.max())] = np.nan
    phase = phase*scale*o*wave/phaseres
    f.close()
    print (wave, scale, o, phaseres)

    return intensity, phase, latscale

#Convert Zygo ASCII to easily readable ASCII format
def convertzygo(filename):
    #read in zygo data
    intensity,phase,latscale = readzygo(filename)

    np.savetxt(filename.split('.')[0]+'.txt',phase,header='Lat scale: '+\
            str(latscale)+'\n'+'Units: meters')

def make_extent(data,dx):
    return [-float(np.shape(data)[1])/2*dx,float(np.shape(data)[1])/2*dx,-float(np.shape(data)[0])/2*dx,float(np.shape(data)[0])/2*dx]
