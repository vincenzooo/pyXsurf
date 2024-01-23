"""
2021/06/15 header is now implemented in data2D.read_data, it calls the function in format_reader with header flag, however there is not exception handling, so I am not sure what happens when the flag is not supported.

2020/07/14 see notes on 02/29. raw readers in format_reader should
   return data with the minimal amount of manipulation and the most general interface. All format/instrument specific aspects
   should be handled there.
   TODO: At the moment `header` flag (set to return header rather than data), is handled by `data2D.read_data` routine called by readers in this file, that wraps the raw reader,
   and registers data. `Data2D.__init__` method when called with a file also fill the property `.header`, this should be handled by calling read_data with     
   , needs to be handled here. Note that some raw readers retrieve header information independently (e.g. as property `header` of an object. In this case, all the handling is done here.
   
   Note also that the entire module might be unnecessary, many of the reader functions say: '''temporary wrapper for new readers, replace call to matrixZygo_reader
    with calls to read_data(wfile,reader=csvZygo_reader)'''

2020/05/13 change import calls to use pySurf.readers.format_reader instead of _instrument_reader

2020/03/17 

2020/02/29 sto definendo meglio la struttura.
_instrument_reader contiene i raw reader, che restituiscono data,x,y come
letti da file con le informazioni necessarie. Questi hanno argomento header, che permette di restituire l'header.

Le funzioni contenute in questo modo devono permettere di restituire funzioni Reader derivate da raw_reader applicando read_data(che esegue registrazioni e allineamento posteriori ai raw e come preprocessing).
Una funzione reader Reader fissa alcuni argomenti per un raw reader e/o per read_data (che devono poi essere overridable alla chiamata).

---------------


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
from pySurf.data2D import read_data

#from utilities.imaging.man import stripnans

from pySurf.points import matrix_to_points2

#from pySurf.readers._instrument_reader import *
#from pySurf.readers._instrument_reader import read_data
#from pySurf.readers._instrument_reader import reader_dic
#from pySurf.readers._instrument_reader import points_reader
'''
2020/07/13 removed, use alt version in data2D
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
            h = reader(file,header=True)
            return h if h else ""
        except TypeError:  #unexpected keyword if header is not implemented
            return None

    #filters register_data parameters cleaning args
    # done manually, there is a function in dataIO.
    scale=kwargs.pop('scale',(1,1,1))
    crop=kwargs.pop('crop',None)
    center=kwargs.pop('center',None)
    strip=kwargs.pop('strip',False)
    regdic={'scale':scale,'crop':crop,'center':center,'strip':strip}

    #pdb.set_trace()
    data,x,y=reader(file,*args,**kwargs)

    #try to modify kwargs
    for k in list(kwargs.keys()): kwargs.pop(k)  #make it empty
    #kwargs.update(regdic)
    for k in regdic.keys():kwargs[k]=regdic[k]
    if register:
        data,x,y=register_data(data,x,y,scale=scale,crop=crop,
        center=center,strip=strip)
    return data,x,y
    '''

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
    from pySurf.readers.format_reader import csvZygo_reader
    #pdb.set_trace()
    return read_data(wfile,csvZygo_reader,*args,**kwargs)

def matrix_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrix4D_reader
    with calls to read_data(wfile,reader=matrix4D_reader)"""
    from pySurf.data2D import data_from_txt
    return read_data(wfile,data_from_txt,*args,**kwargs)

def matrix4D_reader(wfile,header=False,delimiter=',',*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrix4D_reader
    with calls to read_data(wfile,reader=matrix4D_reader).
    
    read csv data with parameters extracted from header as saved by 4D.
    
    """
    from pySurf.readers.format_reader import csv4D_reader

    #import pdb
    #pdb.set_trace()
    return read_data(wfile,csv4D_reader,delimiter=delimiter,*args,**kwargs)


def matrixsur_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrixsur_reader
    with calls to read_data(wfile,reader=sur_reader)"""
    from pySurf.readers.format_reader import sur_reader
    return read_data(wfile,sur_reader,*args,**kwargs)

def matrixdat_reader(wfile,*args,**kwargs):
    """temporary wrapper for zygo metropro .dat binary files"""
    from pySurf.readers.format_reader import datzygo_reader
    return read_data(wfile,datzygo_reader,*args,**kwargs)

def nid_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers,
    with calls to read_data(wfile,reader=fits_reader)"""
    from pySurf.readers.format_reader import read_nid
    return read_data(wfile,read_nid,*args,**kwargs)
    
def points_reader(wfile,*args,**kwargs):
    """temporary wrapper for points readers, read and register
    points and convert to 2D data"""
    from pySurf.readers.format_reader import points_reader
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
    data,x,y=fits_reader(fitsfile)
    header=fits_reader(fitsfile,header=True)

    #from pySurf.points import level_points
    #data=level_points(data)

    dx=header['DX']
    dy=header['DY']
    x=dx*np.arange(data.shape[0])/1000. #convert in mm
    y=dy*np.arange(data.shape[0])/1000.

    return x,y,data


def fits_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to fits_reader
    with calls to read_data(wfile,reader=fits_reader)"""
    from pySurf.readers.format_reader import fits_reader
    return read_data(wfile,fits_reader,*args,**kwargs)
    
def fitsWFS_reader(wfile,ypix=1.,ytox=1.,zscale=1.,crop=[None,None,None],
    center=None,strip=False,scale=(1,1,1.)):
    """return data (matrix) and x,y (vectors) from fits WFS file. Here all lateral sizes must be speficied or default (no header info).
    I don't  understand very well order of data, inverting data
    works for S22 for bump positive, but sample seems rotated and
    horizontally flipped.
    If strip is set, nan are removed. strip is done after centering and scaling, so center is calculated on full data."""
    
    from pySurf.readers.format_reader import fits_reader
    
    data,x,y=fits_reader(wfile)

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
    """return data, x,y from fits file. accept size um to calculate x and y.AFM file"""
    
    from pySurf.readers.format_reader import fits_reader
    data,x,y=fits_reader(fitsfile)
    #z in m
    from pySurf.points import level_points
    data=level_points(data)

    assert data.shape[0]==data.shape[1]  #if not, it's first time it happens, check axis.
    x=np.linspace(1,sizeum[0],data.shape[0])/1000. #convert in mm
    y=np.linspace(1,sizeum[1],data.shape[1])/1000.
    data=data*1e6  #convert from m to um
    return matrix_to_points2(data,x,y)

def fitsCCI_reader(fitsfile,center=None,header=False):
    """return x,y (vectors) and data (matrix) from fits CCI file. `x` and `y` are obtained from 'DX' and 'DY' keys in header. 
    center can be none, in which case data are not centered.
    To extract center position from header, use center=(None,None)."""
    
    from pySurf.readers.format_reader import fits_reader
    
    data,x,y=fits_reader(fitsfile)
    HH=fits_reader(fitsfile,header=True)

    dx=HH['DX']
    dy=HH['DY']
    x=dx*np.arange(data.shape[0])/HH['XUNITRAT'] #convert in mm
    y=dy*np.arange(data.shape[1])/HH['YUNITRAT']

    if center is not None:
        assert len(center)>=2
        if (center is None).any():
            center=(HH['XOFFSET'],HH['YOFFSET'])
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        y=y-(np.max(y)+np.min(y))/2.+center[1]
        if len(center)==3:
            data=data-(np.nanmax(data)+np.nanmin(data))/2.+center[2]

    if header:
        return HH
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

def test_datzygo_reader (wfile=None):
    
    import os
    import matplotlib.pyplot as plt
    from pySurf.data2D_class import Data2D
    from pySurf.readers.format_reader import datzygo_reader

    if wfile is  None:
        #relpath
        #wfile=r'/Volumes/GoogleDrive/Il mio Drive/progetti/sandwich/dati/09_02_27/botta di culo 1a.dat'
        wfile=r'C:\Users\kovor\Documents\python\pyXTel\source\pySurf\test\input_data\readers\botta di culo 1a.dat'
        
        #wfile= os.path.join(os.path.dirname(__file__),relpath)
    (d1,x1,y1)=datzygo_reader(wfile)
    dd1=Data2D(d1,x1,y1)
    
    dd2=Data2D(wfile,reader=datzygo_reader)
    
    plt.figure()
    plt.suptitle(os.path.basename(wfile))
    plt.subplot(121)
    dd1.plot(aspect='equal')
    plt.title('object from data')
    plt.subplot(122)
    dd2.plot(aspect='equal')
    plt.title('object from reader ')
    return dd2




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
