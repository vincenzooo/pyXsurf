#2018/11/02 moved to scripts.repeatabilty from WFS_repeatability in POOL\PSD\WFS_repeatability
#v4 2018/09/18 include functions from reproducibility plot. Includes transformations
#  and differences for any number of plots (it was 3).
#v3 added functions all functions from repeatability.
#v1 works, this v2 adds finer control of styles in make_styles with list value:symbol.

import collections

import numpy as np
import matplotlib.pyplot as plt
import os
from cycler import cycler
from itertools import cycle
import pdb
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import json
import pandas as pd
from plotting.backends import maximize
from pySurf.data2D_class import Data2D
from pySurf.readers.instrumentReader import fitsWFS_reader
from dataIO.fn_add_subfix import fn_add_subfix
from utilities.imaging import fitting as fit
from dateutil.parser import parse

from dataIO.outliers import remove_outliers
from plotting.add_clickable_markers import add_clickable_markers2
from pySurf.affine2D import find_affine, find_rototrans
from plotting.multiplots import find_grid_size, subplot_grid
import itertools

#from pySurf.data2D_class import align_interactive
from plotting.multiplots import commonscale
#from config.interface import conf_from_json

"""2019/04/08 from messy script/repeatability, here sorted function that have been used. When a function is needed by e.g. a notebook, move it here from repeatability_dev.

Each script here should have parameters:
    dis :       call display after plots (and/or prints)
    outfile/outfolder :   produce output with this root name/folder 
    """

def removemis(D2D,func):
    """ convenience function.
    return a copy of data2D object D2D subtracting the results of a function 2D->2D applied to data.
    Original object is not modified."""
    res=D2D.copy() #avoid modification inplace
    res.data=res.data-func(res.data)[0]
    return res

def plot_data_repeat(dlist,name="",num=None,*args,**kwargs):
    """"given a list of Data2D objects dlist, plots them as subplots in maximized window. 
    returns stats.
    """

    res=[]
    plt.clf()
    #fig,axes=plt.subplots(1,len(dlist),num=1)
    xs,ys=find_grid_size(len(dlist),3,square=False)
    fig,axes=plt.subplots(xs,ys,num=num)
    axes=axes.flatten()
    maximize()
    
    for i,(ll,ax) in enumerate(zip(dlist,axes)):
        plt.subplot(xs,ys,i+1,sharex=axes[0],sharey=axes[0])
        ll.plot(stats=True,*args,**kwargs)
        res.append(ll.std())
        #plt.clim([-3,3])
        plt.clim(*(np.nanmean(ll.data)+np.nanstd(ll.data)*np.array([-1,1])))
    plt.suptitle(name+' RAW (plane level)')
    for ax in axes[:len(dlist)-1:-1]:
        fig.delaxes(ax)
        #plt.pause(0.1)
        
    commonscale(plt.gcf())
        
    return res           
    
    
def plot_data_repeat_leveling(dlist,outfile=None,dis=True,name = ""):
    """Create a multiplot on a grid with plots of all data with
    same leveling: raw, conical, cyliindrical, sag removed by line
    
    abused function in old notebooks when type of leveling was uncertain or of interest."""

    xs,ys=find_grid_size(len(dlist),3,square=False)
    res=[]
    r=plot_data_repeat(dlist,name=name,num=1)
    res.append(r)
    if outfile:
        os.makedirs(os.path.dirname(outfile)+'\\raw\\',exist_ok=True)
        plt.savefig(fn_add_subfix(outfile,'_raw','.png',pre='raw\\'))
    if dis: display(plt.gcf())
    
    fig,axes=plt.subplots(xs,ys,num=2)
    axes=axes.flatten()
    maximize()
    res.append([])
    for i,(ll,ax) in enumerate(zip(dlist,axes)):
        plt.subplot(xs,ys,i+1,sharex=axes[0],sharey=axes[0])
        tmp=ll.copy()
        tmp.data=ll.data-fit.fitCylMisalign(ll.data)[0]
        tmp.plot(stats=True)
        res[-1].append(tmp.std())
        #plt.clim([-3,3])
        plt.clim(*(tmp.std()*np.array([-1,1])))
    for ax in axes[:len(dlist)-1:-1]:
        fig.delaxes(ax)
        #plt.pause(0.1)
    plt.suptitle(name+' CYL corrected')
    if outfile:
        os.makedirs(os.path.dirname(outfile)+'\\cyl\\',exist_ok=True)
        plt.savefig(fn_add_subfix(outfile,'_cyl','.png',pre='cyl\\'))
    if dis: display(plt.gcf())
    
    fig,axes=plt.subplots(xs,ys,num=3)
    axes=axes.flatten()
    maximize()
    res.append([])
    for i,(ll,ax) in enumerate(zip(dlist,axes)):
        plt.subplot(xs,ys,i+1,sharex=axes[0],sharey=axes[0])
        tmp=ll.copy()
        tmp.data=ll.data-fit.fitConeMisalign(ll.data)[0]
        tmp.plot(stats=True)
        res[-1].append(tmp.std())
        #plt.clim([-3,3])
        plt.clim(*(tmp.std()*np.array([-1,1])))
    for ax in axes[:len(dlist)-1:-1]:
        fig.delaxes(ax)
        #plt.pause(0.1)
    plt.suptitle(name+' CONE corrected')
    if outfile:
        os.makedirs(os.path.dirname(outfile)+'\\cone\\',exist_ok=True)
        plt.savefig(fn_add_subfix(outfile,'_cone','.png',pre='cone\\'))
    if dis: display(plt.gcf())    
    
    fig,axes=plt.subplots(xs,ys,num=4)
    axes=axes.flatten()
    maximize()
    res.append([])
    for i,(ll,ax) in enumerate(zip(dlist,axes)):
        plt.subplot(xs,ys,i+1,sharex=axes[0],sharey=axes[0])
        tmp=ll.copy()
        #tmp.data=level_data(*ll(),2)[0]
        tmp=tmp.level(2,byline=True)
        tmp.plot(stats=True)
        res[-1].append(tmp.std())
        #plt.clim([-1,1])
        plt.clim(*(tmp.std()*np.array([-1,1])))
    for ax in axes[:len(dlist)-1:-1]:
        fig.delaxes(ax)
        #plt.pause(0.1)
    plt.suptitle(name+' SAG removed by line')
    plt.tight_layout(rect=[0,0,1,0.95])
    
    if outfile:
        os.makedirs(os.path.dirname(outfile)+'\\leg\\',exist_ok=True)
        plt.savefig(fn_add_subfix(outfile,'_leg','.png',pre='leg\\'))
    if dis: display(plt.gcf())  
        
    return res


def plot_repeat(rfiles,outfile=None,dis=True,name = "",plot_func=plot_data_repeat_leveling,ro=None):
    """Functions to plot a list of files side to side (e.g. repeatability or reproducibility)
    with different levelings. Return list of Data2D objects.
    plot_func is a function that accepts a dlist and possibly accepts arguments
       outfile, dis and name (it can have them ignored in **kwargs, or not passed at all to plot_repeat).
    This way, becomes a wrapper around plot_func (that doesn't necessarily have to plot,
      just have an interface outfile, dis, name.
    
    2019/04/08 made function general.
    2018/11/02 moved to scripts. Modified to make it format independent acting on
    data rather than on file extracting the data part in outer routine plot_data_repeat_leveling
    in ."""
    
    plt.close('all')

    if ro is None:
        ro={'reader':fitsWFS_reader,
                'scale':(-1,-1,1),
                'units':['mm','mm','um'],
                'ytox':220/200,
                'ypix':101.6/120,
                'center':(0,0)}
    
    #if name is None:
    #    name = os.path.basename(outfile) if outfile is not None else ""
    
    dlist=[Data2D(file=wf1,**ro).level() for i,wf1 in enumerate(rfiles)]
    res = plot_func(dlist,outfile=outfile,dis=dis,name = name)
    
    
    return dlist


    
from pySurf.readers._instrument_reader import auto_reader

    
def dcouples_plot(dlist,level=True,dis=False):
    """calculate rotating differences, data are supposed to be already aligned.
    Note, differences are not leveled.
    if dis is set to True, call display after plots (not found a better way of doing this)."""
    
    dcouples=[c[1]-c[0] for c in list(itertools.combinations(dlist, 2))]
    if level:
        dcouples=[d.level() for d in dcouples]

    plt.clf()
    maximize()
    
    xs,ys=find_grid_size(len(dcouples),square=True)[::-1]
    fig,axes=plt.subplots(xs,ys)
    if len(np.shape(axes))>1:
        axes=axes.flatten()
    elif len(np.shape(axes))==0:
        axes=[axes]
    maximize()
    for i,(ll,ax) in enumerate(zip(dcouples,axes)):
        plt.subplot(xs,ys,i+1,sharex=axes[0],sharey=axes[0])
        ll.plot()
        plt.clim(*(ll.std()*np.array([-1,1])))
    for ax in axes[:len(dcouples)-1:-1]:
        fig.delaxes(ax)
        #plt.pause(0.1)
    #plt.tight_layout()
    

    #return [d.std() for d in [diff21,diff31,diff32]]
    return dcouples
    
    
def plot_rep_diff(dlist,outfile=None,dis=True):
    """Get three 2d arrays in a list. Calculate rotating differences for different component
    removal: plane, cylinder, cone, legendre.
    returns an 4 element list with the 4 possible removal for the 3 combinations of files to diff  """

    res=[]    
    
    plt.close('all')
    plt.figure(1)
    res.append(dcouples_plot(dlist))
    plt.suptitle('Differences RAW')
    if outfile is not None:
        plt.savefig(fn_add_subfix(outfile,'_raw','.png',pre='raw\\diff_'))
    if dis: display(plt.gcf())    
    
    plt.figure(2)
    res.append(dcouples_plot([removemis(dd,fit.fitCylMisalign) for dd in dlist]))
    plt.suptitle('Differences CYL removed')
    if outfile is not None:
        plt.savefig(fn_add_subfix(outfile,'_cyl','.png',pre='cyl\\diff_'))
    if dis: display(plt.gcf())    

    plt.figure(3)
    res.append(dcouples_plot([removemis(dd,fit.fitConeMisalign) for dd in dlist]))
    plt.suptitle('Differences CONE removed')
    if outfile is not None:
        plt.savefig(fn_add_subfix(outfile,'_cone','.png',pre='cone\\diff_'))
    if dis: display(plt.gcf())    

    plt.figure(4)
    res.append(dcouples_plot([dd.level((2,2)) for dd in dlist]))
    plt.suptitle('Differences 2,2 Legendre removed')
    if outfile is not None:
        plt.savefig(fn_add_subfix(outfile,'_leg','.png',pre='leg\\diff_'))
    if dis: display(plt.gcf())    
    
    return res


