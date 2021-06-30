import numpy as np
from dataIO.span import span
import pdb
import logging


def stats (data,units=None,string=False,fmt=None,vars=None):
    """ Return selected statistics on data as numerical array or list of strings (one for each stats).  
    
    vars is a list of indices that select the variables to be included, wrt a list:
    0 - mean
    1 - stddev
    2 - PV
    3 - min
    4 - max
    5 - number of elements
    
    Note that span doesn't exclude nan data, put flag to tune this option.
    
    string if set to True return stats as strings. In this case a string `units` can be used to add a postfix to statistics. A finer control can be obtained by passing in `fmt` a list of format strings for each var.
    e.g.: the default is obtained with:
    
            fmt = ['mean: %.3g'+units,
                   'StdDev: %.3g'+units,
                   'PV: %.3g'+units,
                   'min: %.3g'+units,
                   'max: %.3g'+units,
                   'n:  %i']
    
    """
    # pdb.set_trace()
    st = [np.nanmean(data),np.nanstd(data),span(data,size=True),*span(data),np.size(data)]
        
    if vars is None: vars = [0,1,2,3,4,5]
    #pdb.set_trace()

    if len(vars) > 0:
        st = [s for i,s in enumerate(st) if i in vars]
    else:
        st = []
        
    # convert to string
    if string:
        # units defaults to empty string, or adjust spacing
        if units is None: 
            units = ""
        else:
            units = " " + units
        # default format uses the units    
        if fmt is None:
            fmt = ['mean: %.3g'+units,
                'StdDev: %.3g'+units,
                'PV: %.3g'+units,
                'min: %.3g'+units,
                'max: %.3g'+units,
                'n:  %i']
        elif np.size(fmt) == 1:
            #if single string replicates to all variables (no units)
            fmt = np.repeat(fmt,len(st))
        
        # 2021/06/30 here st and fmt are of same length. 
        # before today it was expecting complete format
        # st = [s for i,s in enumerate(st) if i in vars]  # filter st
        st = [f%val for f,val in zip (fmt,st)]
    
    
    return st


def test_stats(*args,**kwargs):
    # generate random distribution
    n  = 1000
    
    v = np.random.random(n)
    
    print("no options: all stats as float:", stats(v),"\n----------\n\n")    
    print("with empty vars, return empty:", stats(v,vars=[]),"\n----------\n\n")
    print("select two stats", stats(v,vars=[0,1]),"\n----------\n\n")
    print("return as string, include units:", stats(v,string=True,vars = [0,1], units ='km'),"\n----------\n\n")    
    print("return string, with format", stats(v,string=True,vars = [0,1],
                                              fmt = ['media: %.6g A',
                                                     'scarto: %.3i mm']
                                              ),"\n----------\n\n")
    
    #
    
    """
    [0.5012723328195388, 0.29169410906179205]

    stats(v)
    [0.5012723328195388,
    0.29169410906179205,
    0.9992940201696846,
    0.0005865607705232145,
    0.9998805809402078,
    1000]

    stats(v,string='True')
    ['mean: 0.501',
    'StdDev: 0.292',
    'PV: 0.999',
    'min: 0.000587',
    'max: 1',
    'n:  1000']

    stats(v,string=True, units ='km')
    ['mean: 0.501 km',
    'StdDev: 0.292 km',
    'PV: 0.999 km',
    'min: 0.000587 km',
    'max: 1 km',
    'n:  1000']
    """
    
    
    return stats(v,*args,**kwargs)
    
    
    
if __name__ == "__main__":
    test_stats()