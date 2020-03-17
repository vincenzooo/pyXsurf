"""  This module contains functions able to read raw data from different formats.
     Returns data and header as a couple, with minimal data processing, functions are about formats, 
     not the specific instrument or specific header, that are handled in calling functions
     (e.g. from instrument_reader). 
     
     functions in this module shouldn't have extra args or kwargs.
     

2018/09/26 New refactoring of data reader. Vincenzo Cotroneo vcotroneo@cfa.harvard.edu"""

import numpy as np
from dataIO.fn_add_subfix import fn_add_subfix

def read_fits(fitsfile):
    """ Generic fits reader, returns data,x,y.
    
    header is ignored. If `header` is set to True is returned as dictionary."""
    from astropy.io import fits
    
    a=fits.open(fitsfile)
    meta=a[0]   
    data=meta.data
    a.close()
 
    x= meta.x if hasattr(meta,'x') else np.arange(data.shape[1])
    y= meta.y if hasattr(meta,'y') else np.arange(data.shape[0])
       
    raw = meta.header if hasattr(meta,'header') else meta
    
    return (data,x,y),raw
    
def read_sur(file):
    from read_sur_files import readsur
    """read a sur file using read_sur_files, that is expected to return a structure
     res.points, .xAxis, .yAxis"""

    res = readsur(file) #in readsur the default is False.
    data,x,y = res.points,res.xAxis,res.yAxis
    del res.points
    del res.xAxis
    del res.yAxis

    return (data,x,y),res #stripped of all data information
    
from pySurf.points import get_points,points_find_grid,resample_grid

def read_csv_points(wfile,*args,**kwargs):
    """Read a processed points file in format x,y,z as csv output of analysis routines."""
    print("read_csv_points is obsolete, use read_csv_data")
    w0=get_points(wfile,*args,**kwargs)
    w=w0.copy()
    raw=w0
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return (pdata,x,y),raw


from pySurf.points import matrix_to_points2, points_find_grid, resample_grid
def read_csv_data(filename,x=None,y=None,matrix=True,addaxis=False,skip_header=None,delimiter=' '):

    """return `data, x, y` from generic csv files in xyz or matrix format. 
    
    for example for nanovea saved txt (matrix=False) or gwyddion saved matrix (matrix=True, xrange, yrange must be defined).
    
    x, y: added 2018/02/17 make sense only for data in matrix format and allow reading from file
    ovevrriding or setting of values for the x and y axis. allowing modification that would not be 
    straightforward on the returned value in `points` format.

      input x and y can be None or empty object(calculate from data size), M-element array (must fit data size)
    or range (2-el). If the last two cases (when appropriate axis are passed), they are always used and get priority on what is read from file, be careful to omit them unless you want to alter data.  
      
    This way I can e.g. open a matrix file and return points with a modified x y grid. Setting x or y to an empty object rather than to None discards axis from file and use grid indices.
    
    A complete description of the possible options is:
     read and use from file: addaxis=True, x= None
     read and discard from file, use passed: addaxis=True, x=np.array
     read and discard from file, use calculated: addaxis=True, x=[]
     don't read from file, use passed: addaxis=False, x=np.array
     don't read from file, use calculated: addaxis=False, x=None|[]
    
    matrix: if set, assume data are a matrix rather than x,y,z points.
    addaxis: (if matrix is set) can be set to read values for axis in first row and column
        (e.g. if points were saved with default addaxis=True in save_points. 
    2018/02/17 reintroduced xrange even if discorauged. implemented x and y (unused as well) to axis, range 
    or indices.
    
    skip_header: passed to `np.genfromtxt`, numbers of lines to consider as header
    delimiter:  default=whitespace character to divides columns
    
    2019/04/09 removed scale and extra keywords, copied from points.get_points to format_reader after standardization. It is a more general version of both read_csv_points and data2D.data_from_txt.
    """
    #import pdb
               
    #2014/04/29 added x and y as preferred arguments to xrange and yrange (to be removed).
    if skip_header is None: 
        skip=0 
    else: 
        skip=skip_header
    
    with open(filename) as myfile:   #get first `skip` lines as header
        head = [next(myfile) for x in range(skip)]
    
    mdata=np.genfromtxt(filename,skip_header=skip,delimiter=delimiter)    
    
    if (matrix):     
        if addaxis:
            yy,mdata=np.hsplit(mdata,[1])
            yy=yy[1:] #discard first corner value
            xx,mdata=np.vsplit(mdata,[1])
    
        # x and y can be None or empty object(calculate from data size), M-element array (must fit data size)
        #  or range (2-el). Transform them in M-el vectors    
        if x is None:
            try:
                x=xx #it was called with addaxis
            except NameError:
                x=[] #this has size 0
        else:                #use provided

            if np.size(x) != mdata.shape[1]:
                if np.size(x) == 2:
                    x=np.linspace(x[0],x[1],mdata.shape[1])       
                else:
                    raise ValueError("X is not correct size (or 2-el array)")
        if np.size(x) == 0:  #use calculated or read
            x = np.arange(mdata.shape[1])     
            
        if y is None:
            try:
                y=yy #it was called with addaxis
            except NameError:
                y=[]
        else:                #use provided
            if np.size(y) != mdata.shape[0]:
                if np.size(y) == 2:
                    y=np.linspace(y[0],y[1],mdata.shape[0])       
                else:
                    raise ValueError("Y is not correct size (or 2-el array)")

        if np.size(y) == 0:
            y = np.arange(mdata.shape[0])
          
    else:
        xx,yy=points_find_grid(wmdata,'grid')[1]
        if np.size(x) == 0:  x=xx
        if np.size(y) == 0:  y=yy    
        mdata=resample_grid(mdata,xgrid=x,ygrid=y,matrix=True)
        
    
    return (mdata,x,y),head      
    
    


## location of test input and output data,
##  for economy during development, hard coded path.
#testfolder=r'G:\My Drive\libraries\python\userKov3\pySurf\test' 
    
def test_reader(file,reader,outfolder=None,infolder=None,**kwargs):
    """called without `raw` flag, return data,x,y. Infolder is taken
      from file or can be passed (e.g. to point to local test data)."""
    
    import os
    import matplotlib.pyplot as plt
    from  pySurf.data2D import plot_data
    
    if infolder is None:
        infolder=os.path.dirname(file) 
    
    df=os.path.join(infolder,file)
    res,header=reader(df,**kwargs)
    print("returned values",[r.shape for r in res],header)
    
    plot_data(res[0],res[1],res[2])
    if outfolder is not None:
        if outfolder == "" : 
            display(plt.gcf()) 
        else: 
            outname=os.path.join(infolder,outfolder,os.path.basename(df))
            os.makedirs(os.path.dirname(outname),exist_ok=True)
            plt.savefig(fn_add_subfix(outname,'','.png'))
    return res,header
	
def csv_points_reader(wfile,*args,**kwargs):
    """Read a processed points file in format x,y,z as csv output of analysis routines."""
    w0=get_points(wfile,*args,**kwargs)
    w=w0.copy()
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return pdata,x,y

def csv_zygo_reader(wfile,*args,**kwargs):
    """Read a processed points file in format x,y,z as csv output of analysis routines."""
    w0=get_points(wfile,*args,**kwargs)
    w=w0.copy()
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return pdata,x,y


eegKeys = ["FP3", "FP4"]
gyroKeys = ["X", "Y"]

# 'Foo' is ignored
data = {"FP3": 1, "FP4": 2, "X": 3, "Y": 4, "Foo": 5}

filterByKey = lambda keys: {x: data[x] for x in keys}
eegData = filterByKey(eegKeys)
gyroData = filterByKey(gyroKeys)

print(eegData, gyroData) # ({'FP4': 2, 'FP3': 1}, {'Y': 4, 'X': 3})

filterByKey2 = lambda data,keys : {key: data[key] for key in keys if key in data}

print (filterByKey(eegKeys)) # {'FP4': 2, 'FP3': 1}
