# From MandlebrotDisplay, I remove the class to adjust it to my example.
#http://matplotlib.org/examples/event_handling/viewlims.html

import numpy as np
import matplotlib.pyplot as plt
from dataIO.span import span
import copy

def mandel(xstart, xend, ystart, yend):
    niter=50
    radius=2.
    power=2
    dims = plt.gca().axesPatch.get_window_extent().bounds
    width = int(dims[2] + 0.5)
    height = int(dims[2] + 0.5)
    x = np.linspace(xstart, xend, width)
    y = np.linspace(ystart, yend, height).reshape(-1, 1)
    c = x + 1.0j * y
    threshold_time = np.zeros((height, width))
    z = np.zeros(threshold_time.shape, dtype=np.complex)
    mask = np.ones(threshold_time.shape, dtype=np.bool)
    for i in range(niter):
        z[mask] = z[mask]**power + c[mask]
        mask = (np.abs(z) < radius)
        threshold_time += mask
    return threshold_time
        
        
def ax_update(ax):
    ax.set_autoscale_on(False)  # Otherwise, infinite loop

    # Get the number of points from the number of pixels in the window
    #dims = ax.axesPatch.get_window_extent().bounds

    # Get the range for the new area
    xstart, ystart, xdelta, ydelta = ax.viewLim.bounds
    xend = xstart + xdelta
    yend = ystart + ydelta

    # Update the image object with our new data and extent
    xl=[xstart,xend]
    yl=[ystart,yend]

    # Update the image object with our new data and extent
    im = ax.images[-1]
    cdata=mandel(xstart, xend, ystart, yend)
    im.set_data(cdata)
    crange=span(cdata)
    im.set_extent((xstart, xend, ystart, yend))
    if not (im.colorbar is None):
        fig=ax.get_figure()
        s = copy.copy( fig.canvas.toolbar._views )  
        #this store and restore is made to preserve the zooming history, 
        p = copy.copy( fig.canvas.toolbar._positions )
        im.colorbar.remove()
        plt.clim(crange)
        plt.colorbar()
        fig.canvas.toolbar._views = s
        fig.canvas.toolbar._positions = p
    #
        
    #ax.figure.canvas.draw_idle()

def ax_update2(ax):
    ax.set_autoscale_on(False)  # Otherwise, infinite loop

    # Get the number of points from the number of pixels in the window
    #dims = ax.axesPatch.get_window_extent().bounds

    # Get the range for the new area
    xstart, ystart, xdelta, ydelta = ax.viewLim.bounds
    xend = xstart + xdelta
    yend = ystart + ydelta

    # Update the image object with our new data and extent
    xl=[xstart,xend]
    yl=[ystart,yend]

    # Update the image object with our new data and extent
    im = ax.images[-1]
    cdata=mandel(xstart, xend, ystart, yend)
    im.set_data(cdata)
    crange=span(cdata)
    im.set_extent((xstart, xend, ystart, yend))
    if not (im.colorbar is None):
        fig=ax.get_figure()
        s = copy.copy( fig.canvas.toolbar._views )  
        #this store and restore is made to preserve the zooming history, 
        p = copy.copy( fig.canvas.toolbar._positions )
        im.colorbar.remove()
        plt.clim(crange)
        plt.colorbar()
        fig.canvas.toolbar._views = s
        fig.canvas.toolbar._positions = p  
    #ax.figure.canvas.draw_idle()

    
if __name__=="__main__":
    md = MandlebrotDisplay()
    Z = md(-2., 0.5, -1.25, 1.25)

    fig1, ax2 = plt.subplots()

    plt.imshow(Z, origin='lower', extent=(md.x.min(), md.x.max(), md.y.min(), md.y.max()))
    plt.colorbar()

    ax2.callbacks.connect('xlim_changed', ax_update2)
    ax2.callbacks.connect('ylim_changed', ax_update2)

    plt.show()