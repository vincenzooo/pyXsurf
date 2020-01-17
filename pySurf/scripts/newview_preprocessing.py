# coding: utf-8

"""
Functions for preprocessing of newview data, moved here from POOL\coatings\newview 2018/10/31.

2018/10/29 add rotation of data (optional)

2018/05/11 this script has the only purpose of preprocessing data by removing
    instrument signature and leveling data. All the remaining part must be moved to
    different modules.

    To launch it, create a list of filenames and a single calibration file f2, launch as: calibrate_samples(files,f2,cr=None )
    Return a dictionary with base filename as key and the calibrated data as value.
    Leveling options have been added as parameters.

    Color scale for difference (calibrated data) can be set with cr."""

"""display_diff and calibrate_samples were contained in newview_plotter, they respectively plot differences
and calibrate files.

plot_diffs and get_datasets were in  and roughly a variation of the above. Moved here to try to integrate,
while newview_plotter is removed as the only other thing done is plotting comparison of PSDs that belongs to PSD2D. """

import os
import copy as cp

from pySurf.data2D_class import Data2D,PSD2D
from pySurf.instrumentReader import matrixZygo_reader
from plotting.backends import maximize
from plotting.fignumber import fignumber
#from pySurf.psd2d import rms_power,plot_rms_power

import matplotlib.pyplot as plt
import numpy as np
from pyProfile.profile import polyfit_profile
from IPython.display import display
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import span, span_from_pixels
from dataIO.outliers import remove_outliers
import logging
import os
import pdb
from scipy.ndimage import map_coordinates
from scipy import interpolate as ip
from plotting.captions import legendbox
from astropy.io import fits
from plotting.backends import maximize

from scripts.plot_surface_analysis import plot_surface_analysis



def display_diff(f1,f2,data=False,dis=True,crd=None):
    """calculates, plots and returns the difference of two data.
      It is just a diff-and-plot with a few added options and settings for color scale and reading files,
      can probably be replaced by more standard routines.

      f1 and f2 are file names, unless data is set, in that case they are interpreted as data.
      Data are plotted on common color scale, while range for difference crd can be imposed,
      otherwise is calculated on outlier removed difference."""

    #set units
    units=['mm','mm','nm']
    zscale=1000

    #load data in d1 and d2 according if arguments are data or filenames
    if not(data):
        d1=Data2D(file=f1,units=units,scale=(1000,-1000,zscale),center=(0,0))
        if f2 is not None:
            d2=Data2D(file=f2,units=units,scale=(1000,-1000,zscale),center=(0,0))
        else:
            d2=cp.deepcopy(d1)*0
            #d2.data=d2.data*0
    else:
        d1=Data2D(f1.data,f1.x,f1.y,units=units)
        if f2 is not None:
            d2=Data2D(f2.data,f2.x,f2.y,units=units)
        else:
            d2=cp.deepcopy(d1)*0
            #d2.data=d2.data*0

    #plane level and diff
    d1=d1.level()
    d2=d2.level()
    diff=(d1-d2).level()

    #calculate common color range
    cr1=remove_outliers(d1.data,span=True)
    cr2=remove_outliers(d2.data,span=True)
    cr=[min([cr1[0],cr2[0]]),max([cr1[0],cr2[1]])]
    if crd is None: #color range for difference.
        crd=remove_outliers(diff.data,span=True)

    #plot on three panels, prints stats
    plt.clf()
    plt.subplot(311)
    maximize()
    d1.plot()
    plt.clim(*cr)
    d1.printstats()
    plt.subplot(312)
    d2.plot()
    plt.clim(*cr)
    d2.printstats()
    plt.subplot(313)
    diff.plot()
    plt.clim(*crd)
    print ("diff 2 - 1 = ")
    diff.printstats()
    plt.tight_layout()
    #plt.clim(*cr)#[-0.0001,0.0001])
    if not(data):
        if f2 is not None:
            plt.title ('Difference %s - %s'%tuple(os.path.basename(ff) for ff in [f1,f2]))
        else:
            plt.title ('%s '%f2)
    plt.show()

    #print ("PV: ",span(diff.data,size=True),diff.units[-1])
    #print ("rms: ",np.nanstd(diff.data),diff.units[-1])
    return d1,d2,diff

def plot_diffs(dataset,repeatlist,labels=None,outfolder=None,nsigma_crange=1,fignum=None,figsize=(9.6,6.7)):
    """Generate comparison and difference plots for a list of datasets (data,x,y) and a list of indices
    `repeatlist` as (index1,index2,label). An optional list of `labels` matching data in repeatlist can be provided,
    and then are used for titles of plots, otherwise numeric indices are simply used."""

    stats=[]
    result=[]
    #plot surface map, who knows why on figure 5
    fignumber(fignum,figsize=figsize)
    for i1,i2,t in repeatlist:
        d1,d2=dataset[i1],dataset[i2]
        diff=d1[0]-d2[0]
        result.append((diff,d1[1],d1[2]))
        #mng = plt.get_current_fig_manager()
        #mng.window.showMaximized()
        plt.clf()
        ax1,ax2,ax3=diff_images(d1[0],d2[0],d2[1],d2[2],fignum=0)  #x and y are taken from second data
        if labels is not None:
            ax1.set_title(labels[i1])
            ax2.set_title(labels[i2])
            stats.append([labels[i1],labels[i2],np.nanstd(diff)])
        else:
            stats.append([i1,i2,np.nanstd(diff)])
        ax3.set_title('Diff, rms= %5.3f nm'%(np.nanstd(diff)*1000) )
        plt.sca(ax3)
        plt.clim(*filtered_span(diff,nsigma=nsigma_crange,itmax=1,span=True))
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        plt.suptitle(t)
        plt.tight_layout()
        if outfolder:
            #pdb.set_trace()
            plt.savefig(os.path.join(outfolder,fn_add_subfix("diff","_%i_%i"%(i1,i2),'.jpg')))
            save_data(os.path.join(outfolder,fn_add_subfix("diff","_%i_%i"%(i1,i2),'.dat')),
                diff,d1[1],d1[2])

    if outfolder:
        np.savetxt(os.path.join(outfolder,"diff_report.txt"),result,fmt="%s")

    return result



def calibrate_samples(files,f2,cr=None,rotate=False,level=((1,1),False)):
    """create a dictionary ddic with calibrated and shape leveled Data2D for each file, key is the base filename.
    Uses display_diff.

    Level is a two element tuple containing (leveling_order,byline) see data2D.level_data."""

    ddic={}
    for ff in files:
        #remove signature
        print("calibrating ",os.path.basename(ff))
        d1,d2,diff=display_diff(ff,f2,dis=False,crd=cr)

        #level and rotate
        diff=diff.level(*level)
        if rotate:
            diff=diff.rot90()

        #plot and saves
        plt.clf()
        lrange=remove_outliers(diff.data,nsigma=3,itmax=1,span=True)
        diff.plot()
        plt.title(d1.name+' leveled' + (' rotated' if rotate else ""))
        plt.clim(*lrange)
        display(plt.gcf())
        diff.printstats(os.path.basename(ff))

        #add data to return dic
        ddic[os.path.basename(ff)]=diff
    return ddic

def getdataset(datalist,outfolder=None,fignum=None,figsize=(9.6,6.7),levelingfunc=None,
    caldata=None,psdrange=[1e-15,1e-6]):
    """for each file in a list read data and make a preview plot using `plot_surface_analysis`,
        return a list of (data,x,y). caldata are optional calibration data of same shape as data."""

    """moved here from newview_plotter. This is a similar function to calibrate samples, """

    if levelingfunc==None:
        levelingfunc= lambda x: x
    result=[]
    if outfolder is not None:
        os.makedirs(outfolder,exist_ok=True)
    fig=fignumber(fignum)
    for ff in datalist:
        plt.clf()
        wdata,x,y=matrixZygo_reader(ff,scale=(1000.,1000,1.),center=(0,0))
        if caldata is not None: wdata=wdata-caldata
        wdata,x,y=levelingfunc(wdata),x,y  #simple way to remove plane
        plot_surface_analysis(wdata,x,y,label=os.path.basename(ff),outfolder=outfolder,nsigma_crange=1,
            fignum=(fignum+1 if fignum else None),psdrange=psdrange,levelingfunc=levelingfunc,frange=[4,100])  #[1e-11,1e-6]
        maximize()
        result.append((wdata,x,y))
    return result
