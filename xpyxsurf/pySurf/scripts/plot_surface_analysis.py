"""
Plot_surface_analysis and leveling.

functions to plot and compare data, to be removed.
"""


import numpy as np
from matplotlib import pyplot as plt
from pySurf.readers.instrumentReader import matrixZygo_reader
from pySurf.data2D import plot_data,removelegendre,level_by_line,projection
from pySurf.data2D import levellegendre,save_data,get_data
# from pySurf.psd2d import psd2d_analysis,rms_power
from utilities.imaging.fitting import legendre2d
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import filtered_span
from dataIO.outliers import remove_outliers
from plotting.multiplots import diff_images
from plotting.fignumber import fignumber
from dataIO.span import span
import os
import pdb
from scipy.stats import ttest_ind
# from pySurf.data2D import plot_surface_analysis
# from pySurf.psd2d import psd_variability
from  plotting.backends import maximize
from plotting.multiplots import compare_images

def plot_surface_analysis(wdata,x,y,label="",outfolder=None,nsigma_crange=1,fignum=None,
    figsize=(9.6,6.7),psdrange=None,levelingfunc=None,frange=None):
    """Create two panel preview plots for data. 
    
    This is two panels of raw and filtered data (second order removed) + histogram 
       of height distribution + PSD. 
    Label is typically the filename and is used to generate title and output filenames.
    fignum and figsize are for PSD2D figure (fignum=0 clear and reuse current figure, 
    None create new).
    nsigma_crange and levelingfunc are used to evaluate color range for plot by iteratively 
    leveling and removing outliers at nsigma until a convergence is reached.
    psdrange is used 
    frange is used for plot of PSD2D
    """
    from pySurf.psd2d import psd2d_analysis
    
    units=['mm','mm','$\mu$m']
    order_remove=2 #remove sag, this is the order that is removed in leveled panel
    if np.size(units)==1:   #unneded, put here just in case I decide to take units
        units=np.repeat(units,3)  #as argument. However ideally should be put in plot psd function.
    
    if levelingfunc is None:
        levelingfunc=lambda x: x-legendre2d(x,1,1)[0]
    
    if outfolder is not None:
        os.makedirs(os.path.join(outfolder,'PSD2D'),exist_ok=True)
    else:
        print('OUTFOLDER is none')
    
    wdata=levelingfunc(wdata)  #added 20181101
    
    rms=np.nanstd(wdata)
    #crange=remove_outliers(wdata,nsigma=nsigma_crange,itmax=1,range=True)
    crange=remove_outliers(wdata,nsigma=nsigma_crange,
        flattening_func=levelingfunc,itmax=2,span=1)
    
    plt.clf()
    
    #FULL DATA
    plt.subplot(221)
    #plt.subplot2grid((2,2),(0,0),1,1)
    plot_data(wdata,x,y,units=units,title='leveled data',stats=True)
    plt.clim(*crange)
    
    #SUBTRACT LEGENDRE
    ldata=levellegendre(y,wdata,order_remove)
    plt.subplot(223)
    lrange=remove_outliers(ldata,nsigma=nsigma_crange,itmax=1,span=True)
    #plt.subplot2grid((2,2),(1,0),1,1)
    plot_data(ldata,x,y,units=units,title='%i orders legendre removed'%order_remove,stats=True)
    plt.clim(*lrange) 
    
    #HISTOGRAM
    #plt.subplot2grid((2,2),(0,1),2,1)
    plt.subplot(222)
    plt.hist(wdata.flatten()[np.isfinite(wdata.flatten())],bins=100,label='full data')
    plt.hist(ldata.flatten()[np.isfinite(ldata.flatten())],bins=100,color='r',alpha=0.2,label='%i orders filtered'%order_remove)
    #plt.vlines(crange,*plt.ylim())
    plt.vlines(crange,*plt.ylim(),label='plot range raw',linestyle='-.')
    plt.vlines(lrange,*plt.ylim(),label='plot range lev',linestyle=':')
    plt.legend(loc=0)
    
    #FREQUENCY AND PSD ANALYSIS
    if fignum==0:   #this is the figure created by psd2d_analysis
        plt.clf()
        fig = plt.gcf()
    #pdb.set_trace()
    stax=plt.gca()   #preserve current axis
    f,p=psd2d_analysis(ldata,x,y,title=label,
        wfun=np.hanning,fignum=fignum,vrange=lrange,
        prange=psdrange,units=units,
        rmsrange=frange)#,ax2f=[0,1,0,0])
    plt.ylabel('slice rms ($\mu$m)')
    ps=projection(p,axis=1,span=True) #avgpsd2d(p,span=1)      
    if outfolder:
        maximize()
        plt.savefig(os.path.join(outfolder,'PSD2D',
            fn_add_subfix(label,"_psd2d",'.png')))
        save_data(os.path.join(outfolder,'PSD2D',fn_add_subfix(label,"_psd2d",'.txt')),
            p,x,f)
        np.savetxt(os.path.join(outfolder,fn_add_subfix(label,"_psd",'.txt')),np.vstack([f,ps[0],ps[1],ps[2]]).T,
            header='f(%s^-1)\tPSD(%s^2%s)AVG\tmin\tmax'%(units[1],units[2],units[1]),fmt='%f')
            #pdb.set_trace()
    plt.sca(stax)
    
    plt.subplot(224)
    plt.ylabel('axial PSD ('+units[2]+'$^2$ '+units[1]+')')
    plt.xlabel('Freq. ('+units[1]+'$^{-1}$)')
    plt.plot(f,ps[0],label='AVG')
    plt.plot(f,ps[1],label='point-wise min')
    plt.plot(f,ps[2],label='point-wise max')
    plt.plot(f,p[:,p.shape[1]//2],label='central profile')
    plt.ylim(psdrange)
    plt.loglog()
    plt.grid(1)
    plt.legend(loc=0) 
    #ideally to be replaced by:
    """
    plt.subplot(224)
    plot_psd(((f,ps[0]),(f,ps[1]),(f,ps[2),(f,p[:,p.shape[1]//2])),
            ['AVG','point-wise min','point-wise max','central profile'])
    plt.ylabel('axial '+plt.ylabel)
    plt.ylim(psdrange)
    """

    plt.suptitle('%s: rms=%5.3f 10$^{-3}$'%(label,rms*1000)+units[2])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if outfolder:
        plt.savefig(os.path.join(outfolder,fn_add_subfix(label,"",'.png')))
    
    return (ldata,x,y),(f,p)
    
def leveldata(wdata,xwg,ywg): 
    """plot and return matrices of wdata, lwdata, lwdata2
    from data wdata. use polynomial fit, non legendre, each line is leveled on 
    extremes, not on zero piston/tilt""" 
    #import pdb
    #pdb.set_trace()
    from pySurf.data2D import removesag
    
    #create data
    lwdata=level_by_line(wdata.copy())
    lwdata2=level_by_line(wdata.copy(),removesag)
    lwdata2=level_by_line(lwdata2)  ##why this? to level extremes, no piston. The default action of level_by_line is remove line through extremes.
    
    #make plots
    #generator of axis
    for ax in compare_images([wdata,lwdata,lwdata2],xwg,ywg,titles=['original','line lev.','sag. lev.'],fignum=0)  :
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

    return wdata,lwdata,lwdata2