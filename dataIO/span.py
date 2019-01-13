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
    Returns min and max of an array together as a tuple.
    Args:
        size: If set, return the size of the interval instead of the span.
        index: If set, return indices instead of elements.
        
    index introduced 2018/02/09
    axis introduced 2015/12/17"""
    
    if index:
        return ispan(array,size=size,axis=axis)
    min=np.nanmin(array,axis=axis)
    max=np.nanmax(array,axis=axis)
    if size:
        return max-min
    else:
        return np.stack([min,max]).squeeze()  #?

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
    