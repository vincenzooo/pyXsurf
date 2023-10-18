import numpy as np
from dataIO.span import span
from itertools import product
import pdb
import logging
import warnings

warnings.simplefilter("once")

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

def split_on_indices(vector,indices,absolute=False):
    """Split an array on a list of lengths (indices represent the number of items in each block). 
    Return a list of splitted sub-arrays.
    N.B.: the preferred implementation is block length and not absolute indices to avoid ambiguity in meaning of
    indices and first/last blocks. If absolute is set, indices are assumed as starting indices, last block is
    automatically included, if first one is wanted, must be explicetely included passing 0 as first index."""
    
    if absolute:
        istartblock = [b for a,b in zip([0]+indices[:-1],indices) if b-a!=1]+[len(vector)]
        indices = [b-a for a,b in zip([0]+istartblock[:-1],istartblock)]
    
    if np.sum(indices) != len(vector):
        warnings.warn ('len of indices for splitting is not matching vector length!')
    ind = np.cumsum(indices).tolist()
    result = [vector[i1:i2] for i1,i2 in zip([0]+ind,ind)]

    return result




def make_raster(*vectors, as_axis = False, extend = True):
    """Create a raster grid on the base of a list of vectors.
    
    Vectors can be passed as axis if as_axis is False, in that case they are used as they are
        or as ranges, in which case they must be 3-vectors with start, stop, range, to be used as
        arguments for np.arange.
        In the second case, extremes can be included by extending the range by one additional step.
    
    TODO: vectorize as_axis and extend.
    """
    
    
    if extend and not as_axis:   vectors = [ (v[0], v[1] + v[2], v[2]) for v in vectors]
    gridaxis = (vectors if as_axis else [np.arange(*v) for v in vectors])    
    return list(product(*gridaxis))


def split_blocks (vector, sep = None, direction = None, split = False):

    """ Given a vector return the lenght of blocks (or blocks if split is True).
    
    A block is a monotonic sequence.
    
    The direction of monotony can be explicitly set with ``direction`` argument 
    (positive if increasing, negative if decreasing, 0 consider blocks as formed by equal elements).
    If direction is not provided, it is inferred from the first two elements.
    
    Return a list: if ``split`` is False (default), a list of the lengths of blocks is returned, otherwise a list of blocks is returned. 
    
    (e.g. [1,2,3,1,2,3] -> return [3,3])
    (e.g. [1,1,1,2,2,2,3,3,3,4,4,4], direction = 0 -> return [3,3,3,3]) 
    (e.g. [5,2,8,7,6,8], direction = 1 -> return [1,2,1,1,1])
    (e.g. [5,2,8,7,6,8], direction = None -> return [2,3,1])
    (e.g. [5,2,8,7,6,8], direction = -1 -> return [2,3,1]) # inferred from first two
    
    """
    
    if direction is None:  # exclude invalid points
        x = [vector[i] for i in np.where(np.isfinite(vector))[0]]
        direction = np.sign(x[1]-x[0])
    diff = np.diff(vector)
    
    if direction == 0:
        diff = np.insert(diff,0,0)
        ind = np.where(diff != 0)[0].tolist()
    else:
        direction = np.sign(direction)
        ind = list(np.where(np.sign(diff) != np.sign(direction))[0]+1)
    ind.append (len(vector))
    
    # correct for nan or invalid data, which artificially create two artificial changes of direction.
    # keep only the first, invalid data will be included in output in second block.
    ind = [i for i in ind if i in np.where(np.isfinite(vector))[0]+1]  # +1 to include nan in following block and keep into account first element if invalid. I suspect it will give problems if last element is invalid,
    #must be kept specifically into account.
    #if not np.isfinite(vector[0]): ind.remove(1)  # 1 is legitimate only if first two elements are finite
    
    #if not cumulative or split: # cumulative can be obtained with np.cumsum()
    ind = [i1-i2 for i1,i2 in zip(ind,[0]+ind[:-1])]
    
    
    
    return split_on_indices(vector, ind) if split else ind


def test_len_blocks():

    print(split_blocks([1,2,3,1,2,3]))   #[3,3]
    print(split_blocks([1,2,3,1,2]))   #[3,2]
    print(split_blocks([1,2,3,1,2],direction = -1))   #[1,1,2,1]
    print(split_blocks([1,2,3,1,2],direction = 0))   #[1,1,1,1,1]
    print(split_blocks([1,1,1,2,2,3,3,3],direction = 0))   #[3,2,3]
    
    print(split_blocks([1,2,3,1,2,3],split = True))   #[[1,2,3],[1,2,3]]
    print(split_blocks([1,2,3,1,2],split = True))   # [[1, 2, 3], [1, 2]]
    print(split_blocks([1,2,3,1,2],direction = -1,split = True))   # [[1], [2], [3, 1], [2]]
    print(split_blocks([1,2,3,1,2],direction = 0,split = True))   # [[1], [2], [3], [1], [2]]
    print(split_blocks([1,1,1,2,2,3,3,3],direction = 0,split = True))   # [[1, 1, 1], [2, 2], [3, 3, 3]]



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
    #breakpoint()
    try:
          next(x for x in l if isinstance(x,list))
    
    except StopIteration:
        return False
    except TypeError: #l is scalar or None
        return False
    
    return True

def test_is_nested_list():
    """
    testvalues=[None,[None],[None,[],[]],[[1,2,3]],[3],[[3]],[[],[2],[3]],[[1],[2],[3]],[[1],[2],[3]],[[1],[2,3],[3]],[[2,3],[],[3]]]
    '''
    None | False
    [None] | False
    [None, [], []] | True
    [[1, 2, 3]] | True
    [3] | False
    [[3]] | True
    [[], [2], [3]] | True
    [[1], [2], [3]] | True
    [[1], [2], [3]] | True
    [[1], [2, 3], [3]] | True
    [[2, 3], [], [3]] | True
    '''
    """
    testvalues=[None,[None],3,[None,[],[]],[[1,2,3]],[3],[[3]],
        [[],[2],[3]],[[1],[2],[3]],[[1],[2],[3]],
        [[1],[2,3],[3]],[[2,3],[],[3]]]
    for tv in testvalues:
        #print('test string output:')
        print (tv,"|",is_nested_list(tv))

def is_iterable(obj):
    """ Check if an object is iterable.
    
    Apparently there is no standard way of diung it in Python, so this function is created to introduce
    a consistent check. Note that this is the suggested way, as checkng an __iter__ method is not enough
    because it can be iterated by __get_items__"""
    
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def test_make_raster(vectors):
    print (make_raster((1.,2,0.5),(3,4.,0.5),(2,4,1),extend = False)) 
    #[(1.0, 3.0, 2), (1.0, 3.0, 3), (1.0, 3.5, 2), (1.0, 3.5, 3), (1.5, 3.0, 2), (1.5, 3.0, 3), (1.5, 3.5, 2), (1.5, 3.5, 3)]
    print (make_raster((1.,2,0.5),(3,4.,0.5)))
    # [(1.0, 3.0), (1.0, 3.5), (1.0, 4.0), (1.5, 3.0), (1.5, 3.5), (1.5, 4.0), (2.0, 3.0), (2.0, 3.5), (2.0, 4.0)]    
    print (make_raster((1.,2,0.5),(3,4.,0.5), as_axis=True))
    # [(1.0, 3), (1.0, 4.0), (1.0, 0.5), (2, 3), (2, 4.0), (2, 0.5), (0.5, 3), (0.5, 4.0), (0.5, 0.5)]
    print (make_raster((1.,2,0.5),(3,4.,0.5), as_axis=True, extend = True))
    # [(1.0, 3), (1.0, 4.0), (1.0, 0.5), (2, 3), (2, 4.0), (2, 0.5), (0.5, 3), (0.5, 4.0), (0.5, 0.5)]


    
if __name__ == "__main__":
    test_stats()