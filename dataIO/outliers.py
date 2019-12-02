import numpy as np
from dataIO.span import span as sp #to avoid conflict with argument.
from typing import Callable
import warnings

class EmptyRangeWarning(RuntimeWarning):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

def remove_outliers(data: np.array,
                    nsigma: float = 3,
                    itmax: int = 1,
                    flattening_func: Callable[[np.array],np.array] = None,
                    span: bool = False,
                    print_partial: bool = False) -> np.array:
                    
    """ iteratively remove outliers out of an interval. Returns a mask, True on data to keep.
        
        For each iteration, outliers outside `nsigma` standard deviations from average are removed. Loop ends after `itmax` (default 1) iterations or if convergency is reached (two consecutive iterations with same stdev).
        A flattening function (or any function that returns a modified version of data) can be passed to be performed at each iteration. At each interaction function is calculated on accepted values only.
        If at any point data is empty, an empty array is returned.

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
    
    if (~mask).all():
        warnings.warn('Returning empty array after filtering of outliers.',EmptyRangeWarning)
        return []
         
    return mask if not(span) else sp(data[mask]) 
    

if __name__ == "__main__":
    a=np.arange(10)
    print('initial array',a,'\n--------------')
    
    b=remove_outliers(a,span=True,nsigma=1)
    print('span=True,nsigma=1',b,'\n--------------')

    b=remove_outliers(a,span=True,nsigma=3)
    print('span=True,nsigma=3',b,'\n--------------')
    
    b=remove_outliers(a,span=True,nsigma=0)
    print('span=True,nsigma=0',b,'\n--------------')
    