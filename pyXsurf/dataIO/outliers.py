import numpy as np
from dataIO.span import span as span # rename to span to avoid conflict with argument.
from typing import Callable
import warnings
import pdb

class EmptyRangeWarning(RuntimeWarning):
    pass
    #def __init__(self, *args, **kwargs):
    #    super().__init__( *args, **kwargs)

def remove_outliers(data: np.array,
                    nsigma: float = 3.,
                    itmax: int = 1,
                    flattening_func: Callable[[np.array],np.array] = None,
                    span: bool = False, #deprecated
                    print_partial: bool = False) -> np.array:
                    
    """ iteratively remove outliers out of an interval. Returns a mask, True on data to keep.
        
        For each iteration, outliers outside `nsigma` standard deviations from average are removed. Loop ends after `itmax` (default 1) iterations or if convergency is reached (two consecutive iterations with same stdev).
        A flattening function (or any function that returns a modified version of data) can be passed to be performed at each iteration. At each interaction function is calculated on accepted values only.
        If at any point data is empty, an empty array is returned.
        
        A mask is returned by default, A copy of the original array with invalid values as nan. Values array can equivalently be obtained from mask by `np.where(mask,a,np.nan)`.
        
        `span` argument is deprecated and it will be removed. Please update your code to use `from dataIO.span import span; span (remove_outliers(data,...))`.

        """
        #see also dataIO.span.filtered_span
        
    if span: # useless and was giving conflict, solved by workaround below.
        print("`span` argument is deprecated and it will be removed. Please update your code to use `from dataIO.span import span; span (remove_outliers(data,...))`.\n\nentering debugger, `c` to continue, `u` to see caller function, `l` list code.")
        pdb.set_trace()
    get_span = span  #rename variable
    from dataIO.span import span
    
    if flattening_func is not None: data=flattening_func(data)
    
    M = np.isfinite(data)  #mask for good data
    sigma=np.nanstd(data[M])
    sigmast=0
    i=1
    if itmax > 0:
        while sigma != sigmast:
            sigmast=sigma
            # keep in M only points in the range and recalculate sigma
            M = M & (np.abs(data-np.nanmean(data)) < (nsigma*sigma))
            data = np.where(M,data,np.nan)
            if flattening_func is not None: data=flattening_func(data)
            sigma=np.nanstd(data)
            if print_partial:
                print (i,sigma,sigmast)        
            if i >= itmax or np.all(M is False):
                break                
            i=i+1
    elif print_partial:
        print ("itmax = 0, just M valid data.")
    
    if not (M).any():
        warnings.warn('Returning empty array after filtering of outliers.',EmptyRangeWarning)
        return []
    #pdb.set_trace()  
    #if get_span: return span(data)
    #return M if mask else np.where(M,data,np.nan)
    return M


def filter_outliers(data: np.array,
                    nsigma: float = 3.,
                    itmax: int = 1,
                    flattening_func: Callable[[np.array],np.array] = None,
                    span: bool = False, #deprecated
                    print_partial: bool = False) -> np.array:
                    
    """ This is a generator version that uses remove_outliers to provide iterative result if 
    that iteratively remove outliers out of an interval. 
    Each iteration return a mask, True on data to keep.
        
        For each iteration, outliers outside `nsigma` standard deviations from average are removed. Loop ends after `itmax` (default 1) iterations or if convergency is reached (two consecutive iterations with same stdev).
        A flattening function (or any function that returns a modified version of data) can be passed to be performed at each iteration. At each interaction function is calculated on accepted values only.
        If at any point data is empty, an empty array is returned.
        
        A mask is returned by default, A copy of the original array with invalid values as nan. Values array can equivalently be obtained from mask by `np.where(mask,a,np.nan)`.
        
        `span` argument is deprecated and it will be removed. Please update your code to use `from dataIO.span import span; span (remove_outliers(data,...))`.

        """
        #see also dataIO.span.filtered_span
        
    if span: # useless and was giving conflict, solved by workaround below.
        print("`span` argument is deprecated and it will be removed. Please update your code to use `from dataIO.span import span; span (remove_outliers(data,...))`.\n\nentering debugger, `c` to continue, `u` to see caller function, `l` list code.")
        pdb.set_trace()
    get_span = span  #rename variable
    from dataIO.span import span
    
    # --------------
    if flattening_func is not None: data=flattening_func(data)
    
    M = np.isfinite(data)  #mask for good data
    sigma=np.nanstd(data[M])
    sigmast=0
    i=1
    if itmax >= 0:
        while sigma != sigmast:
            if i >= itmax or np.all(M is False):
                break 
            sigmast=sigma
            # keep in M only points in the range and recalculate sigma
            M = M & (np.abs(data-np.nanmean(data)) < (nsigma*sigma))
            data = np.where(M,data,np.nan)
            if flattening_func is not None: data=flattening_func(data)
            sigma=np.nanstd(data)
            if not (M).any():
                warnings.warn('Returning empty array after filtering of outliers.',EmptyRangeWarning)
                yield []  
            yield M      
               
            i=i+1

def test_filter_outliers(a=None):
    if a is None:
        a=np.arange(10)
    
    print('initial array',a,'\n--------------')    
    for i,m in enumerate(filter_outliers(a,nsigma=1,itmax=3)):
        print('iter: ',i)
        print('mask: ',m)
        print('values: ',a[m])
        print('\n')
    
def test_remove_outliers(a=None):
    if a is None:
        a=np.arange(10)
    
    print('initial array',a,'\n--------------')
    
    b=remove_outliers(a,nsigma=1)
    print('span=True,nsigma=1',b,'\n--------------')
    
    b=remove_outliers(a,nsigma=1,itmax =3,print_partial=True)
    print('span=True,nsigma=1,itmax =3,print_partial=True',b,'\n--------------')

    b=remove_outliers(a,nsigma=3)
    print('nsigma=3',b,'\n--------------')
    
    b=remove_outliers(a,nsigma=0)
    print('nsigma=0',b,'\n--------------')    

if __name__ == "__main__":
    a=np.arange(10)
    test_remove_outliers(a)