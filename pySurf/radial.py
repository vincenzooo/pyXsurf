"""Radial psf functions.

https://photutils.readthedocs.io/en/stable/psf.html

https://learn.astropy.org/rst-tutorials/synthetic-images.html?highlight=filtertutorials#3.-Prepare-a-Point-Spread-Function-(PSF)
"""


import numpy as np
from matplotlib import pyplot as plt
from dataIO.span import span


def barycenter(data,x,y):
    """return barycenter of 2D data, weighted by data."""
    xc = np.sum(x*data)/np.sum(data)
    yc = np.sum(y*data)/np.sum(data)

def radius (x,y,xc,yc):
    """calculate radius for points xx,yy from center in xc,yc.
    
    Return an array same size of xx/yy with all radii."""
    return np.sqrt((y-yc)**2+(x-xc)**2)

def radius_sorted(data,x,y,xc,yc):
    """return radius and corresponding data sorted by distance from xc,yc."""
    
    xg,yg = np.meshgrid(x,y)
    r = radius (xg.flatten(),yg.flatten(),xc,yc)
    i = np.argsort(r,axis=None)
    return r.flatten()[i],data.flatten()[i]

def radial_slice(x,y,angle,xyc,plot=False,
                    *args,**kwargs):
    """Gives extreme points of a segment passing by xc, yc with slope angle in radians inside the span of points x and y.
    works in all four quadrants."""
    
    if len(np.shape(angle))==0:
        angle = [angle]

    xx = x - xyc[0]
    yy = y - xyc[1]
    xs = span(xx)
    ys = span(yy)
    
    xxs = []
    for a in angle:
        cx = np.cos(a)
        cy = np.sin(a)

        t = xs/cx #parameter that brings to the x limits
        t2 = ys/cy #parameter that brings to the y limits
        # they are one negative and one positive if center in rectangle.
        tmin = max(min(t),min(t2))
        tmax = min(max(t),max(t2))
        p1 = (xyc[0]+tmin*cx,xyc[1]+tmin*cy)
        p2 = (xyc[0]+tmax*cx,xyc[1]+tmax*cy)
        xxs.append((p1,p2))
        
        if plot:
            l = plt.plot(*zip(p1,p2),*args,**kwargs)
            c = l[-1].get_color()
            plt.plot(*p1,'x',color = c)
            plt.plot(*p2,'^', color = c)
            #plt.arrow(*p1,*(np.array(p2)-np.array(p1)),*args,**kwargs)
    plt.show()
        
    return np.array(xxs)