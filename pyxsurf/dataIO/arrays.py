import numpy as np
from dataIO.span import span
import pdb
import logging
import warnings

warnings.simplefilter("always")

def stats (data=None,units=None,string=False,fmt=None,vars=None):
    """ Return selected statistics on data as numerical array or list of strings (one for each stats).  
    
    vars is a list of indices that select the variables to be included, wrt a list (if called without `data` returns a format representation of the variables in the list):
    0 - mean
    1 - stddev
    2 - rms
    3 - PV
    4 - min
    5 - max
    6 - number of elements
    
    N.B.: (1) is intended as the rms of the deviation from the mean, while (2) is the root mean square of the signal as value (wrt to zero).
    Note that span doesn't exclude nan data, put flag to tune this option.
    
    string if set to True return stats as strings. In this case a string `units` can be used to add a postfix to statistics. A finer control can be obtained by passing in `fmt` a list of format strings for each var.
    e.g.: the default is obtained with:
    
            fmt = ['mean: %.5g'+units,
                   'StdDev: %.5g'+units,
                   'rms: %.5g'+units,
                   'PV: %.5g'+units,
                   'min: %.5g'+units,
                   'max: %.5g'+units,
                   'n:  %i']
    
    2021/06/30 added rms (different from standard dev, which is centered about mean).
    """

    if vars is None: vars = [0,1,2,3,4,5,6]
    
    if data is not None:
        # `st` stats to return are filtered based on vars
        data = np.array(data)
        st = [np.nanmean(data),np.nanstd(data),
              np.sqrt(np.nansum(data**2)/np.size(np.where(~np.isnan(data))[0])),
              span(data,size=True),*span(data),np.size(data)]
        if len(vars) > 0:
            st = [s for i,s in enumerate(st) if i in vars]
        else:
            st = []
    
    # convert to string
    #pdb.set_trace()
    if string or data is None:
        # `units` must be a single string and it is applied according to default fmt.   
        # `units` defaults to empty string, for different units must be incorporated in `fmt`. 
        if units is None or len(units)==0:   # if not empty adjust spacing  
            units = ""
        else:
            units = " " + units 
        if fmt is None or data is None or not(fmt):
            fmt = ['mean: %.5g'+units,
                'StdDev: %.5g'+units,
                'rms: %.5g'+units,
                'PV: %.5g'+units,
                'min: %.5g'+units,
                'max: %.5g'+units,
                'n:  %i']
            
            # filter `fmt` for indices in vars
            fmt = [f for i,f in enumerate(fmt) if i in vars]
            
            if data is None:
                warnings.warn ("`stats` called without data, print and returns default vars.")
                return fmt    
        elif np.size(fmt) == 1:  #??
            #if single string replicates to all variables
            fmt = np.repeat(fmt,len(st))
            if units:
                fmt = [f+units for f in fmt]
            
        #otherwise `fmt` and  is assumend already of same size as `st`
        
        st = [f%s for f,s in zip(fmt,st)]
        # 2021/06/30 here st and fmt are of same length. 
        # before today it was expecting complete format
        # st = [s for i,s in enumerate(st) if i in vars]  # filter st
    
    return st


def test_stats(*args,**kwargs):
    # generate random distribution
    n  = 1000
    
    v = np.random.random(n)
    
    print('len(data): ',len(v))   
    print("no options: all stats as float:\n")
    print(stats(v),"\n----------\n\n")    
    print("with empty vars, return empty:", stats(v,vars=[]),"\n----------\n\n")
    print("select two stats (vars=[0,1])", stats(v,vars=[0,1]),"\n----------\n\n")
    
    print("return as string (string=True), include units:\n", stats(v,string=True,vars = [2,6], units ='km'),"\n----------\n\n")    
    print("return as string (single string), that is appended to format according to default meaning:", stats(v,string=True,vars = [2,6],fmt = ['rms: %.6g A',
    'size: %.3i pts'],units='[A]'),"\n----------\n\n")  
    print("for finer control (e.g. more that one unit), it needs to be included in fmt)", stats(v,string=True,vars = [2,6],fmt = ['rms: %.6g A',
    'N: %.3i points']),"\n----------\n\n")
    
    print("if called without data, return a format string (units can be added, fmt is irrelevant)", stats(string=True,fmt=None,units='um',vars = [2,6]),"\n----------\n\n") 
    
    print("without data and vars return full format string", stats(string=True),"\n----------\n\n") 
 
    return stats(v,*args,**kwargs)
    
def is_nested_list(l):
    """Return true if it is a nested list (at least one element is a list), False otherwise (all elements are scalar).
    Uses isinstance, so arrays are not considered lists. 
    from https://stackoverflow.com/questions/24180879/python-check-if-a-list-is-nested-or-not."""
    
    try:
          next(x for x in l if isinstance(x,list))
    
    except StopIteration:
        return False
    
    return True

    
    
if __name__ == "__main__":
    test_stats()