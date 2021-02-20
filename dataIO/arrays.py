import numpy as np
from dataIO.span import span
import pdb

def stats (data,units=None,string=False,fmt=None,vars=None):
    """ Return selected statistics on data as array or list of strings (one for each stats).  
    
    string if set to True return stats as strings. In this case a string `units` can be used to add a postfix to statistics. A finer control can be obtained by passing in `fmt` a list of format strings for each var.
    
    vars is a list of indices that select the variables to be included, wrt a list:
        1 - mean
        2 - PV
        3 - min
        4 - max
        5 - number of elements
    
    Note that span doesn't exclude nan data, put flag to tune this option.
    """
    # pdb.set_trace()
    st = [np.nanstd(data),span(data,size=True),*span(data),np.size(data)]
    
    if string:
        if units is None: 
            units = ""
        else:
            units = " "+units
        if fmt is None:
            fmt = ['mean: %.3g'+units,
                   'PV: %.3g'+units,
                   'min: %.3g'+units,
                   'max: %.3g'+units,
                   'n:  %i']
        
        st = [f%val for f,val in zip (fmt,st)]
        
    if vars is None: vars = [0,1,2,3,4]
    #pdb.set_trace()
    try:
        _ = len(vars[0]) == 0 #[], TypeError if None
    except TypeError:  #
        vars=[vars]
    
    #try:
    if isinstance(vars,str):
        vars=[vars]        
    #except:
    #    print('cane')
    st = [s for i,s in enumerate(st) if i in vars]
        
    return st