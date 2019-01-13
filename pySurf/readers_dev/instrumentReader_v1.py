"""v1 2018/05/2017 working. WFS options added today before switching
to version with common interface."""


import numpy as np
from pySurf.data2D import crop_data, remove_nan_frame
from astropy.io import fits
from pySurf.points import get_points
from pySurf.points import crop_points
from pySurf.points import points_find_grid
from pySurf.points import resample_grid
from pySurf.read_sur_files import readsur
from utilities.imaging.man import stripnans
from dataIO.read_pars_from_namelist import read_pars_from_namelist
from _instrument_reader import *

def getdata(fitsfile):
    """return x,y (vectors) and data (matrix) from fits file."""
    a=fits.open(fitsfile)
    header=a[0].header
    data=a[0].data
    a.close()
    
    from pySurf.points import level_points
    data=level_points(data)

    dx=header['DX']
    dy=header['DY']
    x=dx*np.arange(data.shape[0])/1000. #convert in mm
    y=dy*np.arange(data.shape[0])/1000.
  
    return x,y,data
    
def FEAreader(FEAfile):
    """read V. Marquez files, keeping original orientation and variables.
    """
    xx,yy,zz,dx,dy,dz=np.genfromtxt(FEAfile,delimiter=',',skip_header=1,
        unpack=1,usecols=(2,3,4,5,6,7))    
    #a=np.genfromtxt(FEAfile,delimiter=',',skip_header=1)
    #xx,yy,zz,dx,dy,dz=np.hsplit(a[:,2:8],6)
    x=-xx-dx
    y=yy+dy  #Correct for gravity
    z=dz  #here I keep only the deformation, and ignore nominal cylinder shape
    p=np.vstack([x,y,dz]).T*1000. #convert to mm, all coordinates
    return p

def points_reader(wfile,crop=None):
    """Read a processed points file as csv output of analysis routines."""
    w0=get_points(wfile,delimiter=' ')
    w=crop_points(w0.copy(),*crop)
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return pdata,x,y



def matrixZygo_reader(wfile,ypix=None,ytox=1.,zscale=None,crop=None,center=None,strip=False,scale=(1,1,1.),intensity=False):
    """Read a data matrix from csv as saved by Zygo, return as data(2D),x,y.
    ypix is the size of pixel in vertical direction.
    ytox the conversion factor (typically given by radius of test optics/radius of CGH).
    Can crop data and translate on three coordinates. crop is calculated on scaled data 
    before centering.
    center is the position of the central point after crop and scaling.
    If strip is set, nan are stripped and center is calculated on valid data only.
    Scale applies to data after applying ypix and ytox, but before centering. 
        It can be used e.g. to invert x axis as in measuring 220 mm mandrel with 1 m CGH (scale=(-1,1,1)).
    Height map is returned unless intensity is set to True to return Intensity map in same format.
        """
    
    #2018/04/04 modified to return  a single set data,x,y with default height data 
    #   and flag to return intensity, instead of two data sets for intensity and height.
    
    #2017/12/15 modeled on the base of matrix4D_reader, Zygo separates lines with # 
    # An invalid point is indicated by a value >= 65535 (typically 2147483640), 
    #header is 15 lines including last line with \#
    # data size is 320 x 240 there are a bunch of other numbers in the header, but who knows what they mean,
    # they are explained in reference guide, however calibrating scale will be hard.
    
    #2017/05/21 modify process in order: scale, center, crop. 
    #crop coordinates are defined on centered and scaled coordinates. 
    # This has advantage of not changing crop.
    #Note also that center=None (coordinates ranging from lower bottom corner) is different from
    #  center=(0,0,0)
    
    with open(wfile) as myfile:
       raw = myfile.readlines()
       
    header,d=raw[:15],raw[15:]

    nx,ny = list(map( int , header[8].split()[:2] )) 
    # coordinates of the origin of the
    #connected phase data matrix. They refer to positions in the camera coordinate system.
    #The origin of the camera coordinate system (0,0) is located in the upper left corner of the
    #video monitor.     
    origin = list(map(int,header[3].split()[:2]))  
    
    #These integers are the width (columns) and height (rows)
    #of the connected phase data matrix. If no phase data is present, these values are zero.
    connected_size=list(map(int,header[3].split()[2:4]))
    
    #This real number is the interferometric scale factor. It is the number of
    # waves per fringe as specified by the user
    IntfScaleFactor=float(header[7].split()[1])
    
    #This real number is the wavelength, in meters, at which the interferogram was measured.
    WavelengthIn=float(header[7].split()[2])
    
    #This integer indicates the sign of the data. The data sign may be normal (0) or
    #inverted (1).
    DataSign = int(header[10].split()[7])
    #This real number is a phase correction factor required when using a
    #Mirau objective on a microscope. A value of 1.0 indicates no correction factor was
    #required. 
    Obliquity = float(header[7].split()[4])
    
    #This integer indicates the resolution of the phase data points. A value of 0
    #indicates normal resolution, with each fringe represented by 4096 counts. A value of 1
    #indicates high resolution, with each fringe represented by 32768 counts.
    phaseres = int(header[10].split()[0])
    if phaseres==0:
        R = 4096
    elif phaseres==1:
        R = 32768
    else:
        raise ValueError
    
    #This real number is the lateral resolving power of a camera pixel in
    #meters/pixel. A value of 0 means that the value is unknown
    CameraRes = float(header[7].split()[6])
    if CameraRes==0:
        CameraRes=1.

    if ypix is None:
        ypix=CameraRes
    if zscale is None:
        zscale=WavelengthIn*1000000.  #original unit is m, convert to um
        
    if len(scale)==2:
        scale=np.array([scale[0],scale[1],1.])

    #import pdb
    #pdb.set_trace()
    
    datasets=[np.array(aa.split(),dtype=int)  for aa in ' '.join(map(str.strip,d)).split('#')[:-1]]
    #here rough test to plot things
    d1,d2=datasets  #d1 intensity, d2 phase
    d1,d2=d1.astype(float).reshape(ny,nx),d2.astype(float).reshape(*connected_size[::-1])
    d1[d1>65535]=np.nan
    d2[d2>=2147483640]=np.nan
    d2=d2*IntfScaleFactor*Obliquity/R*zscale #in um
    
    dd2=d1*np.nan #d1 is same size as sensor
    dd2[origin[1]:origin[1]+connected_size[1],origin[0]:origin[0]+connected_size[0]]=d2
    d2=dd2
    
    #adjust crop and scales
    if strip:  #messes up x and y 
        print("WARNING: strip nans is not properly implemented and messes up X and Y.")
        d1 = stripnans(d1)
        d2 = stripnans(d2)

    #this defines the position of row/columns, starting from
    x=np.arange(nx)*ypix*ytox*nx/(nx-1)*scale[0]
    y=np.arange(ny)*ypix*ny/(ny-1)*scale[1]

    d2=d2*scale[2]
    #if any x or y is inverted invert data and orient as cartesian.
    if x[-1]<x[0]:
        x=x[::-1]
        d1=np.fliplr(d1)
        d2=np.fliplr(d2)
    if y[-1]<y[0]:
        y=y[::-1]
        d1=np.flipud(d1)  
        d2=np.flipud(d2)         
    x=x-min(x)   
    y=y-min(y)
    
    #doing cropping here has the effect of cropping on cartesian orientation,
    #coordinates for crop are independent on center and dependent on scale and pixel size.
    #center is doing later, resulting in center of cropped data only is placed in the suitable position.
    #
    if crop is None:
        crop=[None,None,None]
    #workaround to act with two data, since crop_data modifies x and y
    d1,x1,y1=crop_data(d1,x,y,*crop)
    d2=crop_data(d2,x,y,*crop)[0]
    x,y=x1,y1
    
    #center data if center is None, leave unchanged
    if center is not None:
        assert len(center)>=2
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        y=y-(np.max(y)+np.min(y))/2.+center[1]
        if len(center)==3:
            data=data-(np.nanmax(data)+np.nanmin(data))/2.+center[2]    
    
    return  (d1,x,y) if intensity else (d2,x,y)
    
 
    
def matrix4D_reader(wfile,ypix=1.,ytox=1.,zscale=1.,crop=None,center=None,strip=False,scale=(1,1,1.)):
    """Read a data matrix from csv as saved by 4D.
    ypix is the size of pixel in vertical direction.
    ytox the conversion factor (typically given by radius of test optics/radius of CGH).
    Can crop data and translate on three coordinates. crop is calculated on raw data,
    center is the position of the central point after crop and scaling.
    If strip is set, nan are stripped and center is calculated on valid data only.
    Scale applies to data after applying ypix and ytox, but before centering. 
        It can be used e.g. to invert x axis as in measuring 220 mm mandrel with 1 m CGH (scale=(-1,1,1))."""
    
    #2018/04/19 moved shifting of x axis inside condition for axis inversion
    #2017/05/21 modify process in order: scale, center, crop. 
    #crop coordinates are defined on centered and scaled coordinates. 
    # This has advantage of not changing crop.
    #Note also that center=None (coordinates ranging from lower bottom corner) is different from
    #  center=(0,0,0)
    import pdb
    #pdb.set_trace()
    header=read_pars_from_namelist(wfile,':')
    if ypix is None:
        ypix=np.float(header['xpix'])
    if zscale is None:
        zscale=np.float(header['wavelength'])
    data=np.genfromtxt(wfile,delimiter=',',skip_header=12)
    
    #adjust crop and scales
    if strip:  #messes up x and y 
        print("WARNING: strip nans is not properly implemented and messes up X and Y.")
        data,x,y = stripnans(data,x,y)
    
    #this defines the position of row/columns, starting from 
    ny,nx=data.shape
    x=np.arange(nx)*ypix*ytox*nx/(nx-1)*scale[0]
    y=np.arange(ny)*ypix*ny/(ny-1)*scale[1]
    data=data*zscale*scale[2]
    #if any x or y is inverted invert data and orient as cartesian.
    if x[-1]<x[0]:
        x=x[::-1]
        data=np.fliplr(data)
        x=x-min(x)
    if y[-1]<y[0]:
        y=y[::-1]
        data=np.flipud(data)  
        y=y-min(y)        
    #x=x-min(x) this should have effect only if axis had been inverted, so I moved them in the conditional  2018/04/19
    #y=y-min(y) 
    
    #doing cropping here has the effect of cropping on cartesian orientation,
    #coordinates for crop are independent on center and dependent on scale and pixel size.
    #center is doing later, resulting in center of cropped data only is placed in the suitable position.
    if crop is None:
        crop=[None,None,None]
    data,x,y=crop_data(data,x,y,*crop)
    
    #center data if center is None, leave unchanged
    if center is not None:
        assert len(center)>=2
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        y=y-(np.max(y)+np.min(y))/2.+center[1]
        if len(center)==3:
            data=data-(np.nanmax(data)+np.nanmin(data))/2.+center[2]    
    
    return data,x,y
"""
def make_reader(reader_function,ypix=1.,ytox=1.,zscale=1.,crop=[None,None,None],center=None,strip=False,scale=(1,1,1.)):
    #makes a full reader with standard axis adjust options and mechanism.
    def adjust_axes(ypix,ytox,zscale,crop,center,strip,scale):
        function_to_decorate(arg1, arg2)
    return a_wrapper_accepting_arguments

# Since when you are calling the function returned by the decorator, you are
# calling the wrapper, passing arguments to the wrapper will let it pass them to 
# the decorated function

@make_reader
def print_full_name(first_name, last_name):
    print("My name is {0} {1}".format(first_name, last_name))

print_full_name("Peter", "Venkman")    
"""

def matrixsur_reader(wfile,crop=None,center=None,strip=False,scale=(1,1,1.)):
    """Read a data matrix from a .sur file (e.g. Nanovea profilometer, CCI).

    Can crop data and translate on three coordinates. crop is calculated on raw data,
    center is the position of the central point after crop and scaling.
    x and y are read from data and can be adjusted with scale and center
        (scale applies to data before centering). It can be used e.g. to invert x axis 
        as in measuring 220 mm mandrel with 1 m CGH (scale=(-1,1,1)).        
    If strip is set, nan are stripped and center is calculated on valid data only.
    """
    
    #2018/04/19 From matrix4D_reader, inherits order of operations: scale, center, crop. 
    #crop coordinates are defined on centered and scaled coordinates. 
    # This has advantage of not changing crop.
    #Note also that center=None (coordinates ranging from lower bottom corner) is different from
    #  center=(0,0,0)
    
    header=readsur(wfile)
    data,x,y=header.points,header.xAxis,header.yAxis
    del(header.points,header.xAxis,header.yAxis)
    
    #adjust crop and scales
    if strip:  #messes up x and y 
        print("WARNING: strip nans is not properly implemented and messes up X and Y.")
        data,x,y = stripnans(data,x,y)
        
    if crop is None:
        crop=[None,None,None]
    
    #this defines the position of row/columns, starting from 
    ny,nx=data.shape
    x=x*scale[0]
    y=y*scale[1]
    data=data*scale[2]
    #if any x or y is inverted invert data and orient as cartesian.
    if x[-1]<x[0]:
        x=x[::-1]
        x=x-min(x)
        data=np.fliplr(data)    
    if y[-1]<y[0]:
        y=y[::-1]
        y=y-min(y)
        data=np.flipud(data)   
    
    #doing cropping here has the effect of cropping on cartesian orientation,
    #coordinates for crop are independent on center and dependent on scale and pixel size.
    #center is doing later, resulting in center of cropped data only is placed in the suitable position.
    data,x,y=crop_data(data,x,y,*crop)
    
    #center data if center is None, leave unchanged
    if center is not None:
        assert len(center)>=2
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        y=y-(np.max(y)+np.min(y))/2.+center[1]
        if len(center)==3:
            data=data-(np.nanmax(data)+np.nanmin(data))/2.+center[2]    
    
    return data,x,y


def fitsWFS_reader(wfile,ypix=1.,ytox=1.,zscale=1.,crop=[None,None,None],
    center=None,strip=False,scale=(1,1,1.),center=None):
    """return data (matrix) and x,y (vectors) from fits WFS file.
    I don't  understand very well order of data, inverting data
    works for S22 for bump positive, but sample seems rotated and
    horizontally flipped.
    If strip is set, nan are removed, it doesn't really work with ranges."""
    
    aa=fits.open(wfile)
    header=aa[0].header
    data=-aa[0].data
    aa.close()
    
    #adjust crop and scales
    if strip:  #messes up x and y 
        print("WARNING: strip nans is not properly implemented and messes up X and Y.")
        data,x,y = stripnans(data,x,y)
    
    #this defines the position of row/columns, starting from 
    ny,nx=data.shape
    #x=x*scale[0]
    #y=y*scale[1]
    x=np.arange(nx)*ypix*ytox*nx/(nx-1)*scale[0]
    y=np.arange(ny)*ypix*ny/(ny-1)*scale[1]
    data=data*zscale*scale[2]
    #if any x or y is inverted invert data and orient as cartesian.
    if x[-1]<x[0]:
        x=x[::-1]
        data=np.fliplr(data)
        x=x-min(x)
    if y[-1]<y[0]:
        y=y[::-1]
        data=np.flipud(data)  
        y=y-min(y)   
    
    #doing cropping here has the effect of cropping on cartesian orientation,
    #coordinates for crop are independent on center and dependent on scale and pixel size.
    #center is doing later, resulting in center of cropped data only is placed in the suitable position.
    if crop is None:
        crop=[None,None,None]
    data,x,y=crop_data(data,x,y,*crop)
    
    #center data if center is None, leave unchanged
    if center is not None:
        assert len(center)>=2
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        y=y-(np.max(y)+np.min(y))/2.+center[1]
        if len(center)==3:
            data=data-(np.nanmax(data)+np.nanmin(data))/2.+center[2]    

    return data,x,y

def fitsWFS_reader2(fitsfile,xrange=None,yrange=None,zrange=None,strip=False):
    """return data (matrix) and x,y (vectors) from fits WFS file.
    I don't  understand very well order of data, inverting data
    works for S22 for bump positive, but sample seems rotated and
    horizontally flipped.
    If strip is set, nan are removed, it doesn't really work with ranges."""
    
    aa=fits.open(fitsfile)
    header=aa[0].header
    data=-aa[0].data
    aa.close()
    
    if strip:
        data,x,y=stripnans(data,x,y)

    x=np.arange(data.shape[1]) #convert in mm
    y=np.arange(data.shape[0])
    
    #w0=matrix_to_points(data,x,y) 
    #w=crop_points(w0.copy(),*crop)
    data,x,y=crop_data(data,x,y,xrange=xrange,yrange=yrange,zrange=zrange)

    return data,x,y
    
    
    
def bin_reader(wfile,index=0,ytox=1.,zscale=1.,crop=[None,None,None],center=None):
    """Read a binary .npy file containing a list of lists in form (imdata, dx).
    Can crop data and translate on three coordinates. crop is calculated on raw data,
    center is the position of the central point after crop and scaling."""
    
    d=np.load(wfile)
    data=d[index][0]
    ypix=d[index][1]
    ny,nx=data.shape
    x=np.arange(nx)*ypix*ytox*nx/(nx-1)
    y=np.arange(ny)*ypix*ny/(ny-1)
    y=max(y)-y    
    data=data*zscale
    data,x,y=crop_data(data,x,y,*crop)
    if center is not None:
        assert len(center)>=2
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        y=y-(np.max(y)+np.min(y))/2.+center[1]
        if len(center)==3:
            data=data-(np.nanmax(data)+np.nanmin(data))/2.+center[2]

    return data,x,y

def fitsAFM_reader(fitsfile,sizeum=None):
    """return x,y (vectors) and data (matrix) from fits file. AFM file"""
    a=fits.open(fitsfile)
    header=a[0].header
    data=a[0].data
    a.close()
    #z in m
    from pySurf.points import level_points
    data=level_points(data)

    assert data.shape[0]==data.shape[1]  #if not, it's first time it happens, check axis.
    x=np.linspace(1,sizeum[0],data.shape[0])/1000. #convert in mm
    y=np.linspace(1,sizeum[1],data.shape[1])/1000.
    data=data*1e6  #convert from m to um
    return matrix_to_points(data,x,y)    
    
def fitsCCI_reader(fitsfile,center=None,head=False):
    """return x,y (vectors) and data (matrix) from fits CCI file.
    center can be none, in which case data are not centered.
    To extract center position from header, use center=(None,None)."""
    aa=fits.open(fitsfile)
    header=aa[0].header
    data=aa[0].data
    aa.close()

    dx=header['DX']
    dy=header['DY']
    x=dx*np.arange(data.shape[0])/header['XUNITRAT'] #convert in mm
    y=dy*np.arange(data.shape[1])/header['YUNITRAT'] 
    
    if center is not None:
        assert len(center)>=2
        if (center is None).any():
            center=(header['XOFFSET'],header['YOFFSET'])
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        y=y-(np.max(y)+np.min(y))/2.+center[1]
        if len(center)==3:
            data=data-(np.nanmax(data)+np.nanmin(data))/2.+center[2]
    
    if head:
        return (data,x,y),header
    else:
        return data,x,y

    
def fits_reader(fitsfile,scale=None):
    """Generic fits file reader. Return x,y (vectors) and data (matrix)."""
    
    if scale is None:
        scale=np.array((1.,1.,1.))
    else:
        if np.array(scale).size == 2:
            scale=(scale[0],scale[1],1.)
        if np.array(scale).size == 1:
            scale=(scale,scale,1.)
            
    aa=fits.open(fitsfile)
    header=aa[0].header
    data=aa[0].data*scale[2]
    aa.close()

    x=np.arange(data.shape[0])*scale[0] #convert in mm
    y=np.arange(data.shape[0])*scale[1]
    
    #w0=matrix_to_points(data,x,y) 
    #w=crop_points(w0.copy(),*crop)
    #data,x,y=crop_data(data,x,y,xrange=xrange,yrange=yrange,zrange=zrange)

    return data,x,y
    
def test_zygo():
    import os
    import matplotlib.pyplot as plt
    from  pySurf.data2D import plot_data
    relpath=r'test\input_data\zygo_data\171212_PCO2_Zygo_data.asc'
    wfile= os.path.join(os.path.dirname(__file__),relpath)
    (d1,x1,y1)=matrixZygo_reader(wfile,ytox=220/1000.,center=(0,0))
    (d2,x2,y2)=matrixZygo_reader(wfile,ytox=220/1000.,center=(0,0),intensity=True)
    plt.figure()
    plt.suptitle(relpath)
    plt.subplot(121)
    plt.title('height map')
    plot_data(d1,x1,y1,aspect='equal')
    plt.subplot(122)
    plt.title('continuity map ')
    plot_data(d2,x2,y2,aspect='equal')
    

if __name__=="__main__":  

    def nonenizer(scale=None):

        if scale is None:
            scale=np.array((1.,1.,1.))
        else:
            if len(scale) == 2:
                scale=(scale[0],scale[1],1.)
            if len(scale) == 1:
                scale=(scale,scale,1.)
        return scale
    
    
    """test nonenizer"""
    a=5
    b=[5]
    c=[5,]
    d=np.array(a)
    e=np.array(b)
    f=np.array(c)

    for i in [a,b,c,d,e,f]:
        print(np.array(i).size)
    