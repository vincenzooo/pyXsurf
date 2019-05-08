

"""functions operating on a list of Data2D objects"""
import os
import matplotlib.pyplot as plt
import numpy as np


from pySurf._instrument_reader import auto_reader
from pySurf.data2D import plot_data,get_data, level_data, save_data, rotate_data, remove_nan_frame, resample_data
from pySurf.data2D import read_data,sum_data, subtract_data, projection, crop_data, transpose_data, apply_transform, register_data
from pySurf.psd2d import psd2d,plot_psd2d,psd2d_analysis,plot_rms_power,rms_power

from pySurf.points import matrix_to_points2

from copy import deepcopy
from dataIO.span import span
from pySurf.data2D import projection
from pySurf.data2D_class import Data2D
from pySurf.affine2D import find_rototrans,find_affine

def load_dlist(rfiles,reader=None,*args,**kwargs):
    """Extracted from plot_repeat. Read a set of rfiles to a dlist.
    readers and additional arguments can be passed as scalars or lists.
    
    You can pass additional arguments to the reader in different ways:
     - pass them individually, they will be used for all readers
         load_dlist(.. ,option1='a',option2=1000)
     - to have individual reader parameters pass them as dictionaries (a same number as rfiles), 
         load_dlist(.. ,{option1='a',option2=1000},{option1='b',option3='c'},..)    
         
    Example:
        dlist=load_dlist(rfiles,reader=fitsWFS_reader,scale=(-1,-1,1),
                units=['mm','mm','um'])
                
        dlist2=load_dlist(rfiles,fitsWFS_reader,[{'scale':(-1,-1,1),
                'units':['mm','mm','um']},{'scale':(1,1,-1),
                'units':['mm','mm','um']},{'scale':(-1,-1,1),
                'units':['mm','mm','$\mu$m']}])
    2019/04/08 made function general from plot_repeat, moved to dlist.
    """
    
    import pdb
    #pdb.set_trace()
    
    if reader is None:
        reader=auto_reader
    if np.size(reader) ==1:
        reader=[reader]*len(rfiles)

    if len(kwargs) >0 : #passed explicit parameters for all readers
        kwargs=[kwargs]*len(rfiles)
    else:
        if np.size(args) ==1: 
            kwargs=[args]*len(rfiles)   
        else:
            if np.size(args) != len(rfiles):
                raise ValueError
    
    #kwargs here is a list of dictionaries {option:value}, matching the readers
    dlist=[Data2D(file=wf1,reader=r,**k) for wf1,r,k in zip(rfiles,reader,kwargs)]
    
    return dlist
    
def test_load_dlist():  

    dlist=load_dlist(rfiles,reader=fitsWFS_reader,scale=(-1,-1,1),
            units=['mm','mm','um'])
            
    dlist2=load_dlist(rfiles,fitsWFS_reader,[{'scale':(-1,-1,1),
            'units':['mm','mm','um']},{'scale':(1,1,-1),
            'units':['mm','mm','um']},{'scale':(-1,-1,1),
            'units':['mm','mm','$\mu$m']}])
    return dlist


def add_markers(dlist):
    """interactively set markers, when ENTER is pressed,
    return markers as list of ndarray.
    It was align_active interactive, returning also trans, this returns only markers,
    transforms can be obtained by e.g. :
    m_trans=find_transform(m,mref) for m in m_arr]  
      """
    
    #set_alignment_markers(tmp)
    xs,ys=find_grid_size(len(dlist),5)[::-1]
    
    fig,axes=subplot_grid(len(dlist),(xs,ys),sharex='all',sharey='all')
    
    axes=axes.flatten()
    maximize()
    for i,(d,ax) in enumerate(zip(dlist,axes)):
        plt.sca(ax)
        ll=d.level(4,byline=True)
        ll.plot()
        plt.clim(*remove_outliers(ll.data,nsigma=2,itmax=1,span=True))
        add_clickable_markers2(ax,hold=(i==(len(dlist)-1)))
    
    return [np.array(ax.markers) for ax in axes]
    
def align_interactive(dlist,find_transform=find_affine,mref=None):
    """plot a list of Data2D objects on common axis and allow to set
    markers. When ENTER is pressed, return markers and transformations"""
    
    m_arr = add_markers (dlist)
    
    #populate array of transforms
    mref = mref if mref is not None else m_arr[0] 
    m_trans = [find_transform(m,mref) for m in m_arr] 
    
    return m_arr,m_trans

def psd2an(dlist,wfun=np.hanning,
                dis=False,ymax=0.05,outfolder="",subfix='_psd2d',*args,**kwargs):
    """2d psd analysis with threshold on sag removed data. Return a list of psd2d.
    ymax sets top of scale for rms right axis.
    if outfolder is provided, save psd2d_analysis plot with dlist names+subfix"""
    m_psd=[]
    for dd in dlist:
        m_psd.append(dd.level(2,byline=True).psd(analysis=True,
            title=os.path.basename(outfolder),wfun=wfun,*args,**kwargs))
        #psd2d_analysis(*dd.level(2,byline=True)(),
        #                 title=os.path.basename(outfolder),wfun=wfun,*args,**kwargs)
        #m_psd.append((fs,ps))
        ax=plt.gcf().axes[-1]
        ax.set_ylim([0,ymax]) 
        plt.grid(0)
        ax.grid()
        plt.suptitle(dd.name+' - hanning window - sag removed by line')
        if outfolder is not None:
            plt.savefig(os.path.join(outfolder,dd.name+subfix+'.png'))
        if dis:
            display(plt.gcf())
    return m_psd
    
def extract_psd(dlist,rmsthr=0.07,rmsrange=None,prange=None,ax2f=None,dis=False):
    """use psd2an, from surface files return linear psd."""
    m_psd=psd2an(dlist,rmsthr=rmsthr,rmsrange=rmsrange,prange=prange,ax2f=ax2f,dis=dis)
    
    plt.figure()
    plt.clf()
    labels=[d.name for d in dlist]
    m_tot=[]
    for (f,p),l in zip(m_psd,labels):
        ptot=projection(p,axis=1)
        #plot_psd(f,ptot,label=l,units=d.units)
        np.savetxt(fn_add_subfix(d.name,'_calib_psd','.dat',
            pre=outfolder+os.path.sep),np.vstack([fs,ptot]).T,
            header='SpatialFrequency(%s^-1)\tPSD(%s^2 %s)'%(u[0],u[2],u[0]))
        m_tot.append((f,ptot))
    plt.title(outname)

    plt.legend(loc=0)
    if dis:
        display(plt.gcf())
    return m_tot
