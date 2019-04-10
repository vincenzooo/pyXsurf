"""  This module contains functions able to read raw data from different formats.
     Returns data and header as a couple, with minimal data processing, functions are about formats, 
     not the specific instrument or specific header, that are handled in calling functions
     (e.g. from instrument_reader). 
	 

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
    w0=get_points(wfile,*args,**kwargs)
    w=w0.copy()
    raw=w0
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return (pdata,x,y),raw
	
def csv_zygo_reader(wfile,*args,**kwargs):
    """Read a processed points file in format x,y,z as csv output of analysis routines."""
    pass
	


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
