"""
2018/06/05 v3 the most used reading functions are replaced by a wrapper around read_data that provides a temporary interface. 
read_data distributes passed parameters to the reader and to the data registration (data2D.register_data) providing a temporary interface. 
Removed original code for functions.

A reader function in this module is a function, specific to a file format, that accepts a filename, a set of format-specific parameters, a set of registering parameters (register_data), and return properly registered data,x,y.

It is implemented here as a wrapper around read_data with the argument reader set to the proper raw reader (from _instrument_reader). 
read_data is needed to divide parameters to be pased to the raw_reader, while the fact that here we are defining each reader is just for interface consistency.
e.g.:

    data,x,y = matrixZygo_reader(file,*args,**kwargs)
    
is equivalent to:

    data,x,y = read_data(file,reader=csvZygo_reader,*args,**kwargs) 

Or could be defined as partial functions or closures.

A call to Omitting register_data parameters has the effect of skipping registration.
A dictionary with supported extension and associated readers is 

2018/05/08 v1.2 with common interface for data preprocessing
(scaling, aligning, centering, etc.. ). 

Start migration by writing wrapper functions with old names, calling core functions in _instrument_reader.
Once interface is designed"""


import numpy as np
from pySurf.data2D import crop_data, remove_nan_frame, register_data
from astropy.io import fits
from pySurf.points import get_points
from pySurf.points import crop_points
from pySurf.points import points_find_grid
from pySurf.points import resample_grid
from pySurf.read_sur_files import readsur
from utilities.imaging.man import stripnans
from dataIO.read_pars_from_namelist import read_pars_from_namelist
import pdb

#from pySurf._instrument_reader import *
#from pySurf._instrument_reader import read_data
#from pySurf._instrument_reader import reader_dic 
#from pySurf._instrument_reader import points_reader

def read_data(file,reader,register=True,*args,**kwargs):
    """non essendo sicuro dell'interfaccia per ora faccio cosi'.
    The function first calls the (raw) data reader, then applies the register_data function to address changes of scale etc,
    arguments are filtered and passed each one to the proper routine.
    18/06/18 add all optional parameters, if reader is not passed,
    only registering is done. note that already if no register_data
    arguments are passed, registration is skipped.
    18/06/18 add action argument. Can be  'read', 'register' or 'all' (default, =read and register). This is useful to give fine control, for example to modify x and y after reading and still make it possible to register data (e.g. this is done in 
    Data2D.__init__).
    
    implementation:
    This works well, however read_data must filter the keywords for the reader and for the register and
    this is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is 
    possible to call the read_data procedure with specific parameters, for example in example below, the reader for 
    Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers, 
    while this can be done using read_data. """
    
    """this is an ugly way to deal with the fact that return
    arguments are different if header is set, so when assigned to a variable as in patch routines in pySurf instrumentReader it fails.
    Workaround has been calling directly read_data, not optimal."""
    if kwargs.pop('header',False):
        try:
            return reader(file,header=True)
        except TypeError:  #unexpected keyword if header is not implemented
            return None
    
    
    #filters register_data parameters cleaning args
    # done manually, there is a function in dataIO.
    scale=kwargs.pop('scale',(1,1,1))
    crop=kwargs.pop('crop',None)
    center=kwargs.pop('center',None)
    strip=kwargs.pop('strip',False)
    regdic={'scale':scale,'crop':crop,'center':center,'strip':strip}
    
    data,x,y=reader(file,*args,**kwargs)
    
    #try to modify kwargs 
    for k in list(kwargs.keys()): kwargs.pop(k)  #make it empty
    #kwargs.update(regdic)
    for k in regdic.keys():kwargs[k]=regdic[k] 
    if register:
        data,x,y=register_data(data,x,y,scale=scale,crop=crop,
        center=center,strip=strip)
    return data,x,y

def xread_data(file,reader,register=True,*args,**kwargs):
    """non essendo sicuro dell'interfaccia per ora faccio cosi'.
    The function first calls the (raw) data reader, then applies the register_data function to address changes of scale etc,
    arguments are filtered and passed each one to the proper routine.
    18/06/18 add all optional parameters, if reader is not passed,
    only registering is done. note that already if no register_data
    arguments are passed, registration is skipped.
    18/06/18 add action argument. Can be  'read', 'register' or 'all' (default, =read and register). This is useful to give fine control, for example to modify x and y after reading and still make it possible to register data (e.g. this is done in 
    Data2D.__init__).
    
    implementation:
    This works well, however read_data must filter the keywords for the reader and for the register and
    this is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is 
    possible to call the read_data procedure with specific parameters, for example in example below, the reader for 
    Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers, 
    while this can be done using read_data. """
    
    """this is an ugly way to deal with the fact that return
    arguments are different if header is set, so when assigned to a variable as in patch routines in pySurf instrumentReader it fails.
    Workaround has been calling directly read_data, not optimal."""
    if kwargs.pop('header',False):
        try:
            return reader(file,header=True)
        except TypeError:  #unexpected keyword if header is not implemented
            return None
    
    
    #filters register_data parameters cleaning args
    # done manually, there is a function in dataIO.
    scale=kwargs.pop('scale',(1,1,1))
    crop=kwargs.pop('crop',None)
    center=kwargs.pop('center',None)
    strip=kwargs.pop('strip',False)
    regdic={'scale':scale,'crop':crop,'center':center,'strip':strip}
    
    data,x,y=reader(file,*args,**kwargs)
    
    #try to modify kwargs 
    for k in list(kwargs.keys()): kwargs.pop(k)  #make it empty
    #kwargs.update(regdic)
    for k in regdic.keys():kwargs[k]=regdic[k] 
    if register:
        data,x,y=register_data(data,x,y,scale=scale,crop=crop,
        center=center,strip=strip)
    return data,x,y
    
def matrixZygo_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrixZygo_reader
    with calls to read_data(wfile,reader=csvZygo_reader)"""
    from pySurf._instrument_reader import csvZygo_reader
    return read_data(wfile,csvZygo_reader,*args,**kwargs)

def matrix_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrix4D_reader
    with calls to read_data(wfile,reader=matrix4D_reader)"""
    from pySurf._instrument_reader import data_from_txt
    return read_data(wfile,data_from_txt,*args,**kwargs)
   
def matrix4D_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrix4D_reader
    with calls to read_data(wfile,reader=matrix4D_reader)"""
    from pySurf._instrument_reader import csv4D_reader
    
    #import pdb
    #pdb.set_trace()
    return read_data(wfile,csv4D_reader,*args,**kwargs)
    
    
def matrixsur_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrixsur_reader
    with calls to read_data(wfile,reader=sur_reader)"""
    from pySurf._instrument_reader import sur_reader
    return read_data(wfile,sur_reader,*args,**kwargs)

    
def fits_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to fits_reader
    with calls to read_data(wfile,reader=fits_reader)"""
    from pySurf._instrument_reader import fits_reader
    return read_data(wfile,fits_reader,*args,**kwargs)

def points_reader(wfile,*args,**kwargs):
    """temporary wrapper for points readers, read and register
    points and convert to 2D data"""
    from pySurf._instrument_reader import points_reader
    return read_data(wfile,points_reader,*args,**kwargs)

### UNWRAPPED FUNCTIONS:    
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
    
def getdata(fitsfile):
    """return x,y (vectors) and data (matrix) from fits file."""
    #works with fits from CCI sur files 
    a=fits.open(fitsfile)
    header=a[0].header
    data=a[0].data
    a.close()
    
    #from pySurf.points import level_points
    #data=level_points(data)

    dx=header['DX']
    dy=header['DY']
    x=dx*np.arange(data.shape[0])/1000. #convert in mm
    y=dy*np.arange(data.shape[0])/1000.
  
    return x,y,data
    
def fitsWFS_reader(wfile,ypix=1.,ytox=1.,zscale=1.,crop=[None,None,None],
    center=None,strip=False,scale=(1,1,1.)):
    """read data from a WFS fits file and register them with scale, crop, center, strip.
    
    return data (matrix) and x,y (vectors) from fits WFS file.
    If strip is set, nan are removed. strip is done after centering and scaling, so center is calculated on full data.
    
    
    This is very similar to reading fits and then calling register data (copy and paste).
      order is however different:
          scale, crop, center,strip. Note that in both z is inverted at reading.
    
    This makes it different to call read_data with fitsWFSreader as reader from directly 
    calling the reader.

    Was changed since then: I don't  understand very well order of data, inverting data
    works for S22 for bump positive, but sample seems rotated and
    horizontally flipped.
    """
    
    import pdb
    #pdb.set_trace()
    aa=fits.open(wfile)
    header=aa[0].header
    data=-aa[0].data
    aa.close()
    
    #adjust crop and scales
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

    if strip:  #messes up x and y 
        data,x,y = remove_nan_frame(data,x,y) 
        
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
    