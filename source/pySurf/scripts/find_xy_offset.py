import numpy as np
import os
from pySurf.data2D_class import Data2D
from matplotlib import pyplot as plt
from dataIO.fn_add_subfix import fn_add_subfix
from itertools import product
# All replaceable with Dlist methods

def read_datalist(filelist, *args, **kwargs):
    """from run32 in faraday cup. """
    
    #p1=Data2D(file=wf,reader=data_from_txt,delimiter=' ',units=['mm','mm','$\mu$A'],center=(0,0))
    datalist=[]

    for ff in filelist:  
        mdata=np.genfromtxt(ff,skip_header=4)
        d=Data2D(mdata,name=os.path.basename(ff))
        datalist.append(d)
        
    return datalist

def plot_datalist(datalist):
    for d in datalist:
        plt.figure(figsize=(8,6))
        d.plot(nsigma=5)
        plt.grid()
        plt.show()
        #plt.savefig(fn_add_subfix(d.name,"",".jpg"))
        
        
def plot_diff(d1,d2):
    netdata=d2-d1
    plt.figure(figsize=(8,6))
    netdata.plot(title='difference full - beam',nsigma=5)
    return netdata

def transl (d,offset):
    d2=d.copy()
    d2.x = d2.x+offset[0]
    d2.y = d2.y+offset[1]
    return d2

def transl_rms(d1,d2,tt):
    r=[]
    for t in tt:
        plt.figure()
        diff = (transl(d1,t)-d2)
        diff.plot()
        plt.title ('Offset: %s, %s'%t)
        print ('Offset: %4.3f, %4.3f rms - %5.4g'%(*t,np.nanstd(diff.data)))
        r.append(np.nanstd(diff.data))
        
    return r 


def plot_transl_comparison(d1,d2,t):
    plt.close('all')
    diff = (transl(d1,t)-d2)

    #one panel
    plt.figure()
    diff.plot()
    plt.title ('Offset: %s, %s'%t)

    #two panels
    plt.figure(figsize=(16,12))

    plt.subplot(121)
    (d2-d1).plot(nsigma=None)
    plt.title ('No Offset')

    plt.subplot(122)
    diff.plot(nsigma=None)
    plt.title ('Offset: %.3g, %.3g'%t)

    print ('Offset: %.4f, %.4f rms - %5.3g'%(*t,np.nanstd(diff.data)))
    
    
def match_xy_offset(d1,d2,xraster,yraster,outfolder=None):
    """Given two Data2D objects, `d1` and `d2`, try to fit a mismatch 
    in x and y by shifting the first on a raster grid, and minimizing rms difference. 
    
    `xoffset` and `yoffset` are vectors used to build the output of the raster grid. 
    output a rms matrix with xoffset and yoffset on axis.
    """

    plt.close('all')   
    plot_datalist((d1,d2))
    
    netdata = plot_diff(d1,d2) #initial difference
    if outfolder:
        os.makedirs(outfolder,exist_ok=True)
        plt.savefig(os.path.join(outfolder,'initial_diff.png'))

    # test raster scan creating rms vector
    tt = list(product(xraster, yraster))
    r = transl_rms(d1,d2,tt)
    
    # find best element 
    ibest = np.nanargmin(r)
    print('\n\n Best: ',r[ibest],tt[ibest])
    t = tt[ibest]
    
    plot_transl_comparison(d1,d2,t)
    if outfolder:
        plt.savefig(os.path.join(outfolder,'best_offset_faraday.png'))
        
    return r