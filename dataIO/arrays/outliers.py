import numpy as np
from .span import span as sp
from typing import Callable

def remove_outliers(data: np.array,
                    nsigma: float = 3,
                    itmax: int = 100,
                    flattening_func: Callable[[np.array],np.array] = None,
                    span: bool = False,
                    print_partial: bool = False) -> np.array:
                    
    """ iteratively remove outliers out of an interval. Returns a mask, True on data to keep.
        
        For each iteration, outliers outside `nsigma` standard deviations from average are removed. 
        A flattening function calculated on accepted values only can be passed to be performed at each iteration.
        Loop ends after `itmax` iterations or if convergency is reached (two consecutive iterations with same stdev).
        """
        #see also dataIO.span.filtered_span
    if flattening_func is not None: data=flattening_func(data)
    mask=np.isfinite(data)  #mask keep track of good data
    sigma=np.nanstd(data[mask])
    sigmast=0
    i=1
    if itmax > 0:
        while sigma != sigmast:
            sigmast=sigma

            mask=mask & (np.abs(data-np.nanmean(data))<(nsigma*sigma))
            sigma=np.nanstd(data[mask])
            i=i+1
            if i >= itmax or np.all(mask is False):
                break
            if print_partial:
                print (sigma,sigmast)
    elif print_partial:
        print ("itmax = 0, just mask valid data.")
            
    return mask if not(span) else sp(data[mask]) 
    
    

    