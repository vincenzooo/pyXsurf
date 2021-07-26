import matplotlib.pyplot as plt
import numpy as np
from dataIO.span import span
from scipy.interpolate import interp1d
import copy
import pdb

def updating_plot(ax=None,title=None):
    """make the passed axis ax active (or the current one if not passed), such that
    it updates the color scale when zoomed to include only visible data.
    Axes is modified in place and returned.
    Usage:
        ax=updating_plot()  #make the current axis an updating_plot, axis is returned
        updating_plot(ax1)  #make the specific axis ax1 an updating_plot
    """
    
    """2017/03/09 BUG: if launched on a plot afteer already zoomed, gives runtime error:
    
    
    from plotting.updating_plot import updating_plot

    updating_plot()
    Out[9]: <matplotlib.axes._subplots.AxesSubplot at 0x41bbe908>
    C:\Anaconda2\lib\site-packages\numpy\lib\nanfunctions.py:675: RuntimeWarning: Mean of empty slice
      warnings.warn("Mean of empty slice", RuntimeWarning)
    C:\Anaconda2\lib\site-packages\numpy\lib\nanfunctions.py:1134: RuntimeWarning: invalid value encountered in less_equal
      isbad = (dof <= 0)
    C:\Anaconda2\lib\site-packages\numpy\lib\nanfunctions.py:1136: RuntimeWarning: Degrees of freedom <= 0 for slice.
      warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning)"""
      
    #note: it gives message is zoomed with both extremes out of range 
    # (zooming and panning near the edges). It fails calculating indices and cdata,
    # but still working
    # TODO: make class so that it is possible to enable/disable scale updates or disconnect the callbacks. Can be combined with different modes for setting scale crange (PV as now, std or frozen).
    # TODO: remove retrieval of figure using plt
    
    
    def on_lims_change(ax):
        nsigma=1.5
        
        ax.set_autoscale_on(False)
        #ax.figure.canvas.draw_idle(self)
        xl=ax.xaxis.get_view_interval()   #range of plot in data coordinates, sorted lr
        yl=ax.yaxis.get_view_interval()      #sorted bottom top
        #print xl,yl
        
        
        im=ax.images[-1]
        imdata=im.get_array()  #.filled(np.nan)  #this helps when a masked array is returned, for some reason it doesn't work
        dims=imdata.shape    #number of pixels
        #print dims
        extent=im.get_extent()      #this is the range for axis of  full data.
        #ixl=np.interp(xl,extent[0:2],[0,dims[1]])  #pixel indices for plotted data
        #iyl=np.interp(yl,extent[2:],[0,dims[0]])
        ixl=interp1d(extent[0:2],[0,dims[1]],assume_sorted=False,bounds_error=False)(xl).astype(int)  #pixel indices for plotted data
        iyl=interp1d(extent[2:],[0,dims[0]],assume_sorted=False,bounds_error=False)(yl).astype(int)
        #prevent empty array
        if np.abs(ixl[1]-ixl[0])<1:
            ixl[1]=ixl[0]+1
        if np.abs(iyl[1]-iyl[0])<1 : 
            iyl[1]=iyl[0]+1
        #print ixl[0],ixl[1],iyl[0],iyl[1]
        cdata=imdata[ixl[0]:ixl[1],iyl[0]:iyl[1]]
        #crange=np.nanmean(cdata)+[-np.nanstd(cdata),np.nanstd(cdata)] #this is a more sensitive view.
        if nsigma:
            crange=np.nanmean(cdata)+np.nanstd(cdata)*np.array([-1.,1.])*nsigma
        else:    
            crange=span(cdata)
        #print ax.__repr__(),cdata
        
        #make a copy of zoom history
        fig=ax.figure
        s = copy.copy( fig.canvas.toolbar._views )
        p = copy.copy( fig.canvas.toolbar._positions )
        
        title="min:%4.1f max:%4.1f rms:%4.3f"%(np.nanmin(cdata),np.nanmax(cdata),np.nanstd(cdata))
        #print crange
        ax.set_title(title)
        if not (im.colorbar is None):
            im.colorbar.remove()
        #if not (title is None):
        #    if title=="":
        #these commands resets colorbar
        plt.colorbar()
        plt.clim(crange)
        
        #restore zoom history
        fig.canvas.toolbar._views = s
        fig.canvas.toolbar._positions = p
        
        ax.figure.canvas.draw_idle()
    
    if ax is None: ax=plt.gca()
    
    """
    #failed attempt to remove existing on_lims_change if already existing. 
    
    if 'xlim_changed' in ax.callbacks.callbacks:
        try:
            ff=ax.callbacks.callbacks['xlim_changed']
            for k,f in ff.items():
                if f.func is on_lims_change:
                    print 'disconnect'
                    ax.callbacks.disconnect(k)
                else:
                    print 'no disco'
                    print f.func
                    print on_lims_change
        except KeyError:
            pass
    """
    
    ax.callbacks.connect('xlim_changed', on_lims_change)
    ax.callbacks.connect('ylim_changed', on_lims_change)
    return ax

def updating_mu(ax=None,title=None):
    """mock up. to solve nsigma and 
    make the passed axis ax active (or the current one if not passed), such that
    it updates the color scale when zoomed to include only visible data.
    Axes is modified in place and returned.
    Usage:
        ax=updating_plot()  #make the current axis an updating_plot, axis is returned
        updating_plot(ax1)  #make the specific axis ax1 an updating_plot
    """    
    
    def on_lims_change(ax,title=""):
        nsigma=1.5
        
        #calculate crange from plot, test values:
        crange=np.array([100.,1000.])*nsigma 
 
        ax.set_title(title+" nsigma: %4.1f"%(nsigma))
        ax.figure.canvas.draw_idle()
        
    if ax is None: ax=plt.gca()
    """
    #failed attempt to remove existing on_lims_change if already existing. 
    
    if 'xlim_changed' in ax.callbacks.callbacks:
        try:
            ff=ax.callbacks.callbacks['xlim_changed']
            for k,f in ff.items():
                if f.func is on_lims_change:
                    print 'disconnect'
                    ax.callbacks.disconnect(k)
                else:
                    print 'no disco'
                    print f.func
                    print on_lims_change
        except KeyError:
            pass
    """
    
    ax.callbacks.connect('xlim_changed', on_lims_change)
    ax.callbacks.connect('ylim_changed', on_lims_change)
    return ax

    
    
if __name__=="__main__":
    plt.ion()
    from pySurf.testSurfaces import test_makeGaussian
    a=test_makeGaussian(100)
    fig=plt.gcf()
    fig.canvas.set_window_title('A plot that updates colorscale on zoom')
    updating_plot()
    plt.show()


    