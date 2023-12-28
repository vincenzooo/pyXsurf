# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 23:42:20 2018
https://stackoverflow.com/questions/2478447/find-largest-rectangle-containing-only-zeros-in-an-n%c3%97n-binary-matrix

@author: Vincenzo
"""

from collections import namedtuple
from operator import mul
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from plotting.plot_positions import plot_poly
from pySurf.points import points_find_hull,points_autoresample
from pySurf.points import plot_points,matrix_to_points2
#

from dataIO.span import span
from scipy.ndimage import label, generate_binary_structure 



def area(size):
    return size[0] * size[1]

def max_rectangle_size(histogram):
    Info = namedtuple('Info', 'start height')
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0, 0) # height, width and start position of the largest rectangle
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            elif stack and height < top().height:
                max_size = max(max_size, (top().height, (pos - top().start), top().start), key=area)
                start, _ = stack.pop()
                continue
            break # height == top().height goes here

    pos += 1
    for start, height in stack:
        max_size = max(max_size, (height, (pos - start), start), key=area)

    return max_size

def max_rect(mat, value=0):
    """returns (height, width, left_column, bottom_row) of the largest rectangle 
    containing all `value`'s.

    Example:
    [[0, 0, 0, 0, 0, 0, 0, 0, 3, 2],
     [0, 4, 0, 2, 4, 0, 0, 1, 0, 0],
     [1, 0, 1, 0, 0, 0, 3, 0, 0, 4],
     [0, 0, 0, 0, 4, 2, 0, 0, 0, 0],
     [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
     [4, 3, 0, 0, 1, 2, 0, 0, 0, 0],
     [3, 0, 0, 0, 2, 0, 0, 0, 0, 4],
     [0, 0, 0, 1, 0, 3, 2, 4, 3, 2],
     [0, 3, 0, 0, 0, 2, 0, 1, 0, 0]]
     gives: (3, 4, 6, 5)
    """
    it = iter(mat)
    hist = [(el==value) for el in next(it, [])]
    max_rect = max_rectangle_size(hist) + (0,)
    for irow,row in enumerate(it):
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        max_rect = max(max_rect, max_rectangle_size(hist) + (irow+1,), key=area)
        # irow+1, because we already used one row for initializing max_rect
    return max_rect


def internal_nan_mask(a,x=None,y=None):
    """return a mask with True on internal nans."""
  
    if x is None:
        x=np.arange(np.shape(a)[1])
    if y is None:
        y=np.arange(np.shape(a)[0])    
        
    sx=[np.min(x),np.max(x)]
    sy=[np.min(y),np.max(y)]
     
    # b is 1 where a is nan, 0 othewise
    #used to detect when a region is made of nans
    aisnan=np.where(np.isnan(a),True,False)
    #plt.figure(); plot_data(b,x,y)
    #plt.title('b')

    # s is different integer for each contiguous region, can be  
    s=label(aisnan)[0]
    #plt.figure()
    #plot_data(s)
    
    #m_int mask of internal nan regions of a
    m_int=np.zeros(a.shape).astype(bool)
    for i in range(np.max(s)+1):
        c=s==i 
        if aisnan[c].all():
            #plt.figure()
            #plot_data(c,x,y)
            #for each one find hull and check if it touches borders
            p=matrix_to_points2(c,x,y)
            p=p[p[:,2]!=0,:] #keep only points of current region
            ssx=span(p[:,0]) #ranges of points in region
            ssy=span(p[:,1])
            #print(i,sx,sy,ssx,ssy)
            if (ssx[0] > sx[0] and 
                ssx[1] < sx[1] and 
                ssy[0] > sy[0] and 
                ssy[1] < sy[1]):
                #print(i,'internal')
                import pdb
                #pdb.set_trace()
                m_int=np.logical_or(m_int,c!=0)          
            #print(len(np.where(m_int!=0)[0]))
            #plt.figure()
            #plot_points(p,scatter=1)
            #p=p[~np.isnan(p[:,2]),:]
            #h=points_find_hull(p)
            #plot_poly(h,'bo')
            '''
            if not(((h[0,:] <= min(x)).any() or 
                    (h[0,:] >= max(x)).any() or 
                    (h[1,:] <= min(y)).any() or 
                    (h[1,:] >= max(y))).any()):
                print(i,'internal')
                m_int=np.logical_or(m_int,c==1)    
            '''
    return m_int

def find_internal_rectangle(data,x=None,y=None,continuous=False):
    """ return x and y ranges as ((x0,x1),(y0,y1)) that gives masimum area not containing external nans. 
    if contunuous is True, skip patching the interior nans (bad data surrounded by good data) and crops on the largest area without nans, can be used to speed up the calculation and cut out also holes."""
    
    if continuous:
        #not tested
        ic=max_rect(np.isnan(data))
    else:
        mask=np.zeros(data.shape).astype(bool)
        
        mask = ~np.isnan(data) | internal_nan_mask(data,x,y)
        ic=max_rect(~mask)  #note weird order width-height
        
    
    
    ll=(x[ic[2]],y[ic[3]])
    ur=(x[ic[2]+ic[1]-1],y[ic[3]-ic[0]+1])  #last x and y included
    
    return ((ll[0],ur[0]),(ur[1],ll[1]))
    
    

def test_holes():
    from pySurf.data2D import plot_data
    from pySurf.data2D_class import Data2D
    
    nx,ny=300,200
    a=np.random.random(nx*ny).reshape(ny,nx)
    x=np.arange(nx)*3-300
    y=np.arange(ny)*5-500
    a[:5,:]=np.nan
    a[-15:,:]=np.nan
    a[:,-3:]=np.nan
    a[60:100,:20]=np.nan
    
    a[110:120,160:165]=1
    a[110:120,80:95]=np.nan
    a[20:40,105:125]=np.nan
    
    plt.close('all')
    #plt.subplot(121)
    plt.figure()
    plot_data(a,x,y)
    plt.title('crop all external nans')

    #b=np.where(np.isnan(a),1,a)
    #plt.figure(); plot_data(b,x,y)
    #plt.title('b')
    
    r=find_internal_rectangle(a,x,y)
    
    plot_poly([
    [r[0][0],r[1][0]],
    [r[0][1],r[1][0]],
    [r[0][1],r[1][1]],
    [r[0][0],r[1][1]],
    [r[0][0],r[1][0]]])
        
   # plt.figure()
    
if __name__=="__main__":
    # %%TEST

    
    nx,ny=700,500
    a=np.zeros(nx*ny).reshape(ny,nx)
    x=np.arange(nx)*3-300
    y=np.arange(ny)*5-500
    gd=Data2D(a,x,y,name='test_data')
    
    #gd.data[40:60,150:160]=1
    #gd.data[400:420,300:310]=1
    #gd.data[80:100,420:440]=1

    gd.data[50:60,40:60]=1
    gd.data[300:310,600:620]=1
    gd.data[420:440,80:100]=1
    
    gd.data[50:60,340:360]=1
    gd.data[420:440,330:350]=1    
    gd.data[250:260,40:60]=1
    
    plt.close('all')
    
    plt.figure()
    gd.plot()
    
    ic=max_rect(gd.data)  #note weird order width-height
    
    ll=(gd.x[ic[2]],gd.y[ic[3]])
    ur=(gd.x[ic[2]+ic[1]-1],gd.y[ic[3]-ic[0]+1])  #last x and y included
    
    plot_poly([ll,[ur[0],ll[1]],ur,[ll[0],ur[1]],ll],ls='b')

    plt.show()