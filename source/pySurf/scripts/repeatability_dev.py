# 2023/04/11 functions for development or legacy use. Updated ones are in `repeatability.py`.
# 2018/11/02 moved to scripts.repeatabilty from WFS_repeatability in POOL\PSD\WFS_repeatability
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
from pySurf.readers.instrumentReader import fitsWFS_reader,matrix4D_reader
from dataIO.fn_add_subfix import fn_add_subfix
"""2019/04/08 from meassy script/repeatability, function that need to be sorted and moved to repeatability. When a function is needed by e.g. a notebook, move it from repeatability_dev to repeatability."""

from utilities.imaging import fitting as fit
from dateutil.parser import parse

from dataIO.outliers import remove_outliers
from plotting.add_clickable_markers import add_clickable_markers2
from pySurf.affine2D import find_affine, find_rototrans
from plotting.multiplots import find_grid_size, subplot_grid
import itertools

from pySurf.scripts.dlist import align_interactive
from plotting.multiplots import commonscale
from dataIO.config.interface import conf_from_json

from pySurf.scripts.dlist import dcouples_plot  #, plot_repeat  #see also plot_data_repeat in pySurf.scripts.dlist

"""
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
    
def plot_cross_diff_rms(stats, units = ''):
    """ 2023/04/11 wanted to move to `repeatability.plot_cross_stats` from `repeatability_dev.plot_cross_diff_rms`
    but the function can probably be completely replaced by seaborn heatmap,
    so I just added the docstring and made a few changes for general use.
    
    Plots a colormap with values of root mean square (rms) for all cross differences. 
    
    It takes in an argument stats which is a dictionary containing the cross-difference statistics. 
    The original plotted the value `'con'` (= cone removed) from stats dictionary and another key with a string descriptor 
    of the two files involved.

    """
    
    # The function first creates an empty matrix of size (nfiles,nfiles) where nfiles is the number of files in the input dictionary. Then it fills this matrix with the cross-difference statistics from the input dictionary. Finally, it plots the matrix as a colormap using plot_data() function and adds text labels to each cell of the matrix showing its value. The text color is chosen based on the luminance of the cell color. The function also adds x and y labels to the plot and displays it using display() function.
    
    
    nfiles=len(stats)
    # create error matrix by extracting values from dictionary.
    m_err=np.zeros((nfiles,nfiles))*np.nan
    for j in range(1,nfiles):
        for i in range(j):
            m_err[j,i]=stats['con']['%02i-%02i'%(i+1,j+1)]

    plt.figure()
    plot_data(m_err,range(1,nfiles+1),range(1,nfiles+1),units=['','',units])   # units was set to microns

    cmap=matplotlib.cm.get_cmap()

    for j in range(1,nfiles):
        for i in range(j):
            """copied from seaborn heatmap"""
            cval = matplotlib.colors.Normalize(vmin=np.nanmin(m_err), 
                                                vmax = np.nanmax(m_err))(m_err[i,j])
            color = cmap(cval)       
            rgb = mpl.colors.colorConverter.to_rgba_array(color)[:, :3]
            rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            lum = rgb.dot([.2126, .7152, .0722])
            try:
                l = lum.item()
            except ValueError:
                l = lum
            text_color = ".15" if l > .408 else "w"    
            plt.text(i+1,j+1,'%.3f'%m_err[j,i], color=text_color,
                     horizontalalignment='center',verticalalignment='center')

    plt.xlabel("index of file #1")
    plt.ylabel("index of file #2")
    plt.title("rms of diff for file #1 - #2")


    display(plt.gcf())
    
# PLOTTING FOR SINGLE SET OF FILES

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
        tmp=tmp.level(2,axis=1)
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

def process_set(flist,m_trans=None,m_arr=None,outfolder=None):
    rmslist=[]       
    
    #plot surfaces, dlist is list of Data2D objects with raw data
    dlist=plot_repeat(flist,
                    None if outfolder is None else os.path.join(outfolder,k+'.png'),dis=dis)
    rms=[d.level(1,1).std() for d in dlist]
    
    if m_trans is None:
        tmp2=dlist
    else:
        tmp2=[d.apply_transform(t) for d,t in zip(dlist,m_trans)]
    
    if m_arr is not None:    
        #plot transformed data for verification
        fig,axes=plt.subplots(xs,ys,sharex='all',sharey='all',squeeze=True)
        axes=axes.flatten()
        maximize()
        for d,ax,m,t in zip(tmp2,axes,m_arr,m_trans):
            plt.sca(ax)
            ll=d.level(4,byline=True)
            ll.plot()
            plt.clim(*remove_outliers(ll.data,nsigma=2,itmax=1,span=True))
            plt.plot(*((t(m).T)),'xr')
        if outfolder is not None:
            plt.savefig(None if outfolder is None else os.path.join(outfolder,k+'_alignment.png'))

    #plot differences with various levelings
    difflist = plot_rep_diff(tmp2,None if outfolder is None else os.path.join(outfolder,k+'.png'),dis=dis)
    rmslist.append([[dd.std() for dd in d] for d in difflist])
    
    if columns is None:
        columns = pd.RangeIndex(0,np.size(rmslist))

    if stats is None: stats = pd.DataFrame(columns=columns)
        
    pstats = pd.DataFrame(columns=stats.columns,data=np.array(rmslist).reshape(1,-1),index=[k])
    
    return pstats
    
def process_set2(flist,names,outname,crop=None,dis=True):
    # from C1S15_4D_DI_calibration_181031.ipynb
    # called with:
    ypix=None #0.127901442999 #mm  
    ytox=220./200.  
    zscale=None #0.6328
    scale=(1000,-1000,0.001)

    outfolder=os.path.dirname(outname)
    outname=os.path.basename(outname)
    
    dlist=[Data2D(*matrix4D_reader(wf2,scale=scale,zscale=zscale,
       ypix=ypix,ytox=ytox,center=(0,0),strip=True),name=n,units=['mm','mm','um']) 
       for wf2,n in zip(flist,names)]
    np.mean(dlist).save(os.path.join(outfolder,outname+'_avg.dat'))
    
    plt.close('all')
    sx,sy=find_grid_size(len(dlist),3)
    f,axes=plt.subplots(sx,sy,sharex=True,sharey=True)
    for ax,d in zip(axes.flatten(),dlist):
        plt.sca(ax)
        d.plot()
        plt.clim(*remove_outliers(d.data,nsigma=2,itmax=3,span=1))
    for ax in axes.flatten()[:len(dlist)-1:-1]:
        f.delaxes(ax)

    maximize()
    plt.pause(0.1)
    plt.tight_layout(rect=[0,0,1,0.9])
    plt.suptitle ('Surfaces')
    if dis:
        display(plt.gcf())
    
    plt.savefig(os.path.join(outfolder,outname+'.png'))
    
    m_repr=[dcouples_plot([dd for dd in dlist])]
    if dis:
        display(plt.gcf())
    plt.savefig(os.path.join(outfolder,'diff_'+outname+'.png'))
    
    if crop is not None:
        dc=[d.crop(*crop).level() for d in dlist]
        m_repr=[dcouples_plot([dd for dd in dc])]
        if dis:
            display(plt.gcf())
        plt.savefig(os.path.join(outfolder,'diff_'+outname+'_crop.png'))
       
    return dlist
    
## FUNCTIONS TO HANDLE MULTIPLE FILES    
    
from plotting.linestyles import make_styles


def build_database(confdic,outfolder=None,columns=None,dis=False,
                   find_transform=find_rototrans):
    """Process all files and return stats database in form of pd.DataFrame from confdic. 
    plot_repeat and plot_rep_diff are used to read fits files and calculate rmss.
    If outfolder is provided, output plots are generated in functions.
    Expect an index for columns (can be multiindex.)
    If dis is set, display (set to True for single dataset, False for multiple config processing."""
    
    def make_dates(stats):
        """returns a dataframe dates build with indices from stats and value """
        dates=[]
        for k in stats.index:
            try:
                date=confdic[k]['date']
            except KeyError:
                date=k.split('_')[0]
            dates.append(parse(date,yearfirst=True).date()) 

        return pd.Series(data=dates,index=stats.index)
    
    
   #check existence of files before starting processing
    for k,conf in confdic.items():
        flist=[os.path.join(conf['infolder'],f)for f in conf['files']]
        if not np.all([os.path.exists(f) for f in flist]):
            print (k," has some files that cannot be found:")
            print (conf['infolder'])
            print (conf['files'])
            for f in flist:
                open(f).close()  #raise exception if file doesn't exist
    
    stats = None
    #real processing
    for i,(k,conf) in enumerate(confdic.items()):
        print("Process data %i/%i"%(i+1,len(confdic.items())))
        
        #######
        rmslist=[]
        
        #outfile=os.path.join(outfolder,k)
        flist=[os.path.join(conf['infolder'],f)for f in conf['files']]
        
        #plot surfaces, dlist is list of Data2D objects with raw data
        dlist=plot_repeat(flist,
                        None if outfolder is None else os.path.join(outfolder,k+'.png'),dis=dis)
        rms=[d.level(1,1).std() for d in dlist]
        
        #align if necessary:
        transmode=conf.get('trans',None)
        if transmode=='none' or transmode == "":  #do nothing
            tmp2=dlist
        else:
            plt.close('all')
            if transmode == 'interactive':
                print ("""interactively set markers on each axis with CTRL+LeftMouse. 
                Remove markers with CTRL+right_mounse, ENTER when ready.""")
                #run interactive alignment 
                m_arr,m_trans=align_interactive(dlist,find_transform=find_transform)

                #incorporate markers in config and save backup copy 
                conf['trans']='config'   # alignment mode, read from config
                conf['mref']=m_arr[0].tolist()    #reference markers to aligh to
                conf['markers']=[m.tolist() for m in m_arr]    #markers saved as lists
                json.dump({k:conf},open(os.path.join(outfolder,k+'_out.json'),'w'),indent=0)            

            else:
                if transmode == 'config':    #gets markers from configfile
                    pass  #conf is already set
                elif transmode == 'file':
                    d=json.load(open(conf['markers'],'r'))
                    print(list(d.keys()),'\n\n',d[list(d.keys())[0]])
                    print(len(list(d.keys()))," datasets")  
                    k,conf=list(d.items())[0]                    
                else:
                    raise ValueError
                m_arr=[np.array(m) for m in conf['markers']]
                mref=conf['mref']
                m_trans=[find_transform(m,mref) for m in m_arr]  
                        
            #make and plots transformed (aligned) data to verify feature alignment
            tmp2=[d.apply_transform(t) for d,t in zip(dlist,m_trans)]

            #plot transformed data for verification
            fig,axes=plt.subplots(xs,ys,sharex='all',sharey='all',squeeze=True)
            axes=axes.flatten()
            maximize()
            for d,ax,m,t in zip(tmp2,axes,m_arr,m_trans):
                plt.sca(ax)
                ll=d.level(4,byline=True)
                ll.plot()
                plt.clim(*remove_outliers(ll.data,nsigma=2,itmax=1,span=True))
                plt.plot(*((t(m).T)),'xr')
            plt.savefig(None if outfolder is None else os.path.join(outfolder,k+'_alignment.png'))
        
        #yield dlist
        #plot differences with various levelings
        difflist = plot_rep_diff(tmp2,None if outfolder is None else os.path.join(outfolder,k+'.png'),dis=dis)
        rmslist.append([[dd.std() for dd in d] for d in difflist])
        
        if columns is None:
            columns = pd.RangeIndex(0,np.size(rmslist))

        if stats is None: stats = pd.DataFrame(columns=columns)
            
        pstats = pd.DataFrame(columns=stats.columns,data=np.array(rmslist).reshape(1,-1),index=[k])
            
        #### fuori routine
        
        stats=stats.append(pstats)
        dates=make_dates(stats)

    return stats, dates
    
def plot_historical(dates,stats,col=None,yrange=[0,0.25],styarr=None,*args,**kwargs):
    """plot historical values as scatter plot indicating outliers with arrow and value.  
    dates and stats are data frames with same indices. 
    Plot all data matching column col.
    styarr is a list of graphic styles with same length as dates to apply to each element
    plot. Can be created on the base of data content with make_styles."""
    
    plt.ylim(*yrange)
    al=(plt.ylim()[1]-plt.ylim()[0])*0.05  #arrow length
    tyoff=3*al  #vertical offset for text in outliers plots 
    
    m_cycle = cycler(marker=['s', 'x', 'v', '+', '^', 'o' ])()
    #print(kwargs)
    #
    if styarr is None:
        styarr=[m for i,m in zip(range(len(dates)),cycle(m_cycle))]
        #styarr=[{} for i in range(len(dates))]
    #pdb.set_trace()
    styarr=[a if a is None else (a if 'marker' in a.keys() else {**a,**m}) for a,m in zip(styarr,cycle(m_cycle))]
    
    #sarebbe meglio fare passare gia' ordinati
    isort=np.argsort(dates.values)
    dates=dates[isort]
    stats=stats.iloc[isort]
    styarr=[styarr[i] for i in isort]
    
    oldate=None
    #plt.figure(figsize=([11.,6.35]))
    for i,sty in zip(range(len(dates)),styarr):
        if sty is not None:
            n=stats.index[i]
            #print(n)
            #pdb.set_trace()
            points=(stats[col] if col is not None else stats).loc[n]
            out=[p for p in points if p >= plt.gca().get_ylim()[1]]
            inr=[p for p in points if p < plt.gca().get_ylim()[1]]
            if len(out) > 0:
                #if it is out of y range, plot arrow and text 
                if dates[i] == oldate: #if more than one data same date
                    tyoff=tyoff+al     #set offset for text
                oldate=dates[i]
                plt.plot(dates[i],plt.gca().get_ylim()[1]-al, 'k', marker=r'$\uparrow$', markersize=10)
                
                t=plt.text(mdates.date2num(dates[i]),plt.gca().get_ylim()[1]-tyoff,'%3.2g'%np.nanmax(out))
                plt.plot(dates[i],plt.gca().get_ylim()[1]- 2*al, 'k',**sty)
                #print(a['con'][n])
            if len(inr)>0:
                plt.plot(np.repeat(dates[i],len(inr)),inr,label=n,**sty)

    plt.xticks(rotation=60)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    
    plt.ylabel('rms (um)')
    plt.gcf().subplots_adjust(bottom=0.15,right=0.7)
    plt.grid(axis='y')
    #display(plt.gcf())

##rep_stats and conf_stats from tackbonding notebooks.

def rep_stats(flist,columns=None,dis=True):
    """Analyzes a list of files with possible transformations.
        returns repeatability statistics in form of pdDataFrame. """
    rmslist=[]
    
    #plot surfaces, dlist is list of Data2D objects with raw data
    dlist=plot_repeat(flist,
                    None if outfolder is None else os.path.join(outfolder,k+'.png'),dis=dis)
    rms=[d.level(1,1).std() for d in dlist]

    tmp2=[d if t is None else t(d) for d,t in zip(dlist,m_trans)]
    
    #align if necessary:    
    if transmode=='none' or transmode == "":  #do nothing
        tmp2=dlist
    else:
        plt.close('all')
        if transmode == 'interactive':
            print ("""interactively set markers on each axis with CTRL+LeftMouse. 
            Remove markers with CTRL+right_mounse, ENTER when ready.""")
            #run interactive alignment 
            m_arr,m_trans=align_interactive(dlist,find_transform=find_affine)          

        else:
            if transmode == 'file':
                #read markers from file that some 
                raise NotImplementedError
            else:
                raise ValueError
            m_arr=[np.array(m) for m in conf['markers']]
            mref=conf['mref']
            m_trans=[find_transform(m,mref) for m in m_arr]  

        #make and plots transformed (aligned) data to verify feature alignment
        tmp2=[d.apply_transform(t) for d,t in zip(dlist,m_trans)]

        #plot transformed data for verification
        fig,axes=plt.subplots(xs,ys,sharex='all',sharey='all',squeeze=True)
        axes=axes.flatten()
        maximize()
        for d,ax,m,t in zip(tmp2,axes,m_arr,m_trans):
            plt.sca(ax)
            ll=d.level(4,byline=True)
            ll.plot()
            plt.clim(*remove_outliers(ll.data,nsigma=2,itmax=1,span=True))
            plt.plot(*((t(m).T)),'xr')
        plt.savefig(None if outfolder is None else os.path.join(outfolder,k+'_alignment.png'))

    #yield dlist
    #plot differences with various levelings
    difflist = plot_rep_diff(tmp2,None if outfolder is None else os.path.join(outfolder,k+'.png'),dis=dis)
    rmslist.append([[dd.std() for dd in d] for d in difflist])

    if columns is None:
        tags=['%02i-%02i'%(i,j) for i,j in itertools.combinations(range(1,nfiles+1),2)]
        columns=pd.MultiIndex.from_product([['raw','cyl','con','leg'],tags],names=['terms removed','files#'])
        #columns = pd.RangeIndex(0,np.size(rmslist))
    
    #pdb.set_trace()
    cstats = pd.DataFrame(columns=columns,data=np.array(rmslist).reshape(1,-1),index=[k])

    return cstats




def conf_stats(k,conf,columns=None,dis=True):
    """Analyzes a single configuration (as list of files + processing settings).
        returns repeatability statistics in form of pdDataFrame. """
    rmslist=[]

    #outfile=os.path.join(outfolder,k)
    flist=[os.path.join(conf['infolder'],f)for f in conf['files']]
    transmode=conf.get('trans',None)
    
    
    
    #plot surfaces, dlist is list of Data2D objects with raw data
    dlist=plot_repeat(flist,
                    None if outfolder is None else os.path.join(outfolder,k+'.png'),dis=dis)
    rms=[d.level(1,1).std() for d in dlist]

    tmp2=[d if t is None else t(d) for d,t in zip(dlist,m_trans)]
    
    #align if necessary:    
    if transmode=='none' or transmode == "":  #do nothing
        tmp2=dlist
    else:
        plt.close('all')
        if transmode == 'interactive':
            print ("""interactively set markers on each axis with CTRL+LeftMouse. 
            Remove markers with CTRL+right_mounse, ENTER when ready.""")
            #run interactive alignment 
            m_arr,m_trans=align_interactive(dlist,find_transform=find_affine)

            #incorporate markers in config and save backup copy for the configuration. 
            # This is just temporary ass saver in case of crash, final conf is anyway
            # saved outside for all configurations.
            conf['trans']='config'   # alignment mode, read from config
            conf['mref']=m_arr[0].tolist()    #reference markers to aligh to
            conf['markers']=[m.tolist() for m in m_arr]    #markers saved as lists
            json.dump({k:conf},open(os.path.join(outfolder,k+'_out.json'),'w'),indent=0)            

        else:
            if transmode == 'config':    #gets markers from configfile
                pass  #conf is already set
            elif transmode == 'file':
                d=json.load(open(conf['markers'],'r'))
                print(list(d.keys()),'\n\n',d[list(d.keys())[0]])
                print(len(list(d.keys()))," datasets")  
                k,conf=list(d.items())[0]                    
            else:
                raise ValueError
            m_arr=[np.array(m) for m in conf['markers']]
            mref=conf['mref']
            m_trans=[find_transform(m,mref) for m in m_arr]  

        #make and plots transformed (aligned) data to verify feature alignment
        tmp2=[d.apply_transform(t) for d,t in zip(dlist,m_trans)]

        #plot transformed data for verification
        fig,axes=plt.subplots(xs,ys,sharex='all',sharey='all',squeeze=True)
        axes=axes.flatten()
        maximize()
        for d,ax,m,t in zip(tmp2,axes,m_arr,m_trans):
            plt.sca(ax)
            ll=d.level(4,byline=True)
            ll.plot()
            plt.clim(*remove_outliers(ll.data,nsigma=2,itmax=1,span=True))
            plt.plot(*((t(m).T)),'xr')
        plt.savefig(None if outfolder is None else os.path.join(outfolder,k+'_alignment.png'))

    #yield dlist
    #plot differences with various levelings
    difflist = plot_rep_diff(tmp2,None if outfolder is None else os.path.join(outfolder,k+'.png'),dis=dis)
    rmslist.append([[dd.std() for dd in d] for d in difflist])

    if columns is None:
        tags=['%02i-%02i'%(i,j) for i,j in itertools.combinations(range(1,nfiles+1),2)]
        columns=pd.MultiIndex.from_product([['raw','cyl','con','leg'],tags],names=['terms removed','files#'])
        #columns = pd.RangeIndex(0,np.size(rmslist))
    
    #pdb.set_trace()
    cstats = pd.DataFrame(columns=columns,data=np.array(rmslist).reshape(1,-1),index=[k])

    return cstats
    
    
#CONFIGURATION HANDLING



def stats_from_json(jfile,include=None,exclude=None,outfolder=None,columns=None):
    
    """ emulate obsolete function using new functions 
     json->conf (conf_from_json) + conf->stats (build_database).""" 

    confdic = conf_from_json(jfile,include,exclude) 
    stats,dates=build_database(confdic,outfolder=outfolder,columns=columns)
    plt.title('rms after conical correction for different samples')
    display(plt.gcf())
    return stats,dates
    
"""
EXAMPLE OF CONFIG.JSON

{
"180719_CylRef_repeat": {
"infolder": "tests\\input_data\\180719_Repeatibility",
"files": [
"180719_01_CylRef_01.fits",
"180719_02_CylRef_01.fits",
"180719_03_CylRef_01.fits"
],
"outname": "testset01",
"sample": "CylRef",
"testtype": "repeat",
"trans": "",
"operator": "VC",
"date": "2018/07/19",
"markers": []
},

"180719_CylRef_reprod": {
"infolder": "tests\\input_data\\180719_Repeatibility",
"files": [
"180719_04_CylRef_01.fits",
"180719_05_CylRef_01.fits",
"180719_06_CylRef_01.fits"
],
"outname": "testset02",
"sample": "CylRef",
"testtype": "reprod",
"trans": "",
"operator": "LS",
"date": "2018/07/19",
"markers": []
}
}
"""   
   
if __name__=="__main__":
    
    plt.ion()
    #01
    outfolder=os.path.join(r'tests','WFS_repeatability') #output 
    jfile=os.path.join(outfolder,r'test_settings.json')
    include=['180206_c1s03_gentexcut_', '180206_c1s04_gentexcut_'] #limits analysis to only two files
    include=None

    #CALCULATE ALL!
    confdic=conf_from_json(jfile,include=include)
    mi=pd.MultiIndex.from_product([['raw','cyl','con','leg'],['2-1','3-1','3-2']],names=['terms removed','files#'])
    stats, dates = build_database(confdic,outfolder,columns=mi)
    
    plt.figure(figsize=((8,4)))
    plt.title('rms after conical correction for different samples')
    display(plt.gcf())

    plot_historical(dates,stats,'con',outfolder=outfolder)
    display(stats*1000)
    #------------------
    #outfolder=r'results\test_reptester\WFS_repeatability'
    
    #plot_historical(dates,stats,'con',yrange=[0,0.25],color={"operator":['r','g','b']})
    #plot_historical(dates,stats,col,yrange=[0,0.25],color={"operator":cycler})
    #plot_historical(dates,stats,col,yrange=[0,0.25],color={"operator":{'KG':'r','VC':'g','LS':'b',None:'k']}) #is this the most general? We said I want to control only one graphical prop per category,
    #however I might want to set common extra properties. Can this be done with cyclers?
    


