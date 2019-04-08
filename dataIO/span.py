import numpy as np
import warnings

def ispan(array:np.array,size:bool=False,axis:int=None)->np.array:
    """2018/02/09 as span, but works on indices rather than elements."""
    
    min=np.nanargmin(array,axis=axis)
    max=np.nanargmax(array,axis=axis)
    if size:
        return max-min
    else:
        return np.dstack([min,max]).squeeze()
        
def span(array:np.array,
        size:bool=False,
        axis:int=None,
        index:bool=False)->np.array:
    """
    Returns min and max of an array together as a array of couples (or their indices or interval size).
    Args:
        size: If set, return the size of the interval instead of the span.
        index: If set, return indices instead of elements.
    
    Example:    
    
        pts.shape  #pts is an array of x,y,z coordinates
        Out[206]: (270051, 3)
        
        s=span(pts,axis=0)   #get coordinate span
        s.shape
        Out[208]: (3, 2)
        
        s[0]  #span of x axis, works easily for 2d arrays, but see notes
              below for ndarrays.
        Out[209]: array([-34.3813775 ,  34.57937714])         
    
        # separate axis spans:
        xs,ys,zs=span(pts,axis=0)
        xs
        Out[229]: array([-34.3813775 ,  34.57937714])
        
        xl,yl=span(pts0,axis=0)[:2] #get only x and y spans
    
    
    Note that couples [min,max] are accessible on last index.
    This has the effect to have a couple where the value returned by
      min and max would be a single value and the rank of the array is not changed.
    
    Note that the intent of the function is not to return a result that can be addressed as a 2-element list to give min and max as entire arrays (equal to array.min and array.max). 
    For this behavior on ndim arrays, use instead directly np.nanmin and np.nanmax. 
    
    In general, indexing span as list will have same effect it has on input array, and iteratively with the last axis being a couple [min,max] instead of a single value.
    
    `fit_cylinder` uses this function as: 
        
        # pts.shape
        # Out[138]: (270051, 3)
        
        s=span(pts,axis=0)
        print ('X: [%6.3f,%6.3f]'%tuple(s[0]))
        print ('Y: [%6.3f,%6.3f]'%tuple(s[1]))
        print ('data range: [%6.3f,%6.3f]'%tuple(s[2]))
        
    """    
        
#    2019/03 [solved 04/01] for some reason span starts to fail where it was not. Note that max and min
#       return an array that has the dimension specified by axis removed. span must return
#       array of same dimension x 2. However its is debatable if returning the two min, max
#       arrays with dimension 2 in place of the removed axis or as first dimension
#       (or return min and max as list of two elements).
#       The question is if I want to return span as an array of intervals (one for each axis) or as a two element list of min and max array.
#       s=span(pts,axis=0)
#
#       Note that fit_cylinder expects the first option. 
#       
#   this is solved and updated in documentation.

#   note that this `solved` 19/04 fixes the issue with print as in docstring, 
#     but had to change also in points_find_grid:
#
#        #xs,ys,zs=np.hsplit(span(points,axis=0),3)
#        xs,ys,zs=span(points,axis=0)
#
#     and:
#        xl,yl=span(pts0,axis=0)[:2] #was np.transpose(span(pts0,axis=0)[:,:2])
#
#   this was recently failing too.
       
#    index introduced 2018/02/09
#    axis introduced 2015/12/17"""
    
    if index:
        return ispan(array,size=size,axis=axis)
    min=np.nanmin(array,axis=axis)
    max=np.nanmax(array,axis=axis)
    if size:
        return max-min
    else:
        return np.stack([min,max],axis=-1) # couples [min,max] accessible with 
        #  instead of scalar elements as in e.g. np.min
        #  see explanation in docstring. 
        # it was before 2019/04/01
        # return np.stack([min,max]).squeeze()  #?

def span_from_pixels(p,n=None):
    """From positions of pixel centers p returns a range from side to side. Useful to adjust plot extent in imshow.
    
    In alternative, p can be provided as range and number of pixels. 
    
    note that np.linspace has flag retsteps to return step size."""
    
    if n is None:
        n=len(p)
        
    dx=span(p,size=True)/(n-1)
    
    return (span(p)[0]-dx/2,span(p)[1]+dx/2)

def test_span_from_pixels():    
    print (span_from_pixels([0,3],4)) #[-0.5,3.5]
    print (span_from_pixels([0,2],3)) #[-0.5,2.5]
    print (span_from_pixels([0,1,2])) #[-0.5,2.5]
    print (span_from_pixels([0,0.5,1,1.5,2])) #[-0.25,2.25]
        
def filtered_span(*args,**kwargs):
    from dataIO.outliers import remove_outliers 
    warnings.warn('This routine was replaced by outliers.remove_outliers please update calling code, will be removed.',RuntimeWarning)
    return remove_outliers(*args,**kwargs)
    
    '''   
def filtered_span(  data:np.array,
                    nsigma:float=3,
                    itmax:int=100,
                    span:bool=False,
                    print_partial:bool=False)->np.array:
                    
    """determine the rms after a number of iterations of outlier removals.
        For each iteration, outliers outside `nsigma` standard deviations from average are removed.
        Loop ends after `itmax` iterations or if convergency is reached (two consecutive iterations with same stdev).
        
    see also dataIO.outliers
        """
    #see also dataIO.outliers
    #data=data.flatten()
    data=data[np.isfinite(data)]
    sigma=np.nanstd(data)
    sigmast=0
    i=1
    while sigma != sigmast:
        sigmast=sigma
        
        data=data[np.abs(data-np.nanmean(data))<nsigma*sigma]
        sigma=np.nanstd(data)
        i=i+1
        if i >= itmax or sigma==0:
            break
        if print_partial:
            print (sigma,sigmast)
    return np.nanmean(data)+np.array((-1,1))*sigma if span else sigma 
    '''
    