import inspect  # to use with importing docstring

def update_docstring(func,source,delimiter=''):
    """given a current function and a source function, update current function docstring
     appending signature and docstring of source.

     It provides a decorator @update_docstring(source) update the decorated
       function docstring. User should take care to create docstring of decorated function
       only as preamble to source docstrings, e.g.:
       this is func function, derived from source function by modifying the usage of parameter foo.
       parameter foo is changed of sign."""
       
    print("Using `update_docstring` which is supeseded by decorator `append_doc_from`")
    doc0="" if func.__doc__ is None else func.__doc__  # retrieve current docstring
    func.__doc__='\n'.join([doc0,delimiter,source.__name__+str(inspect.signature(source)),source.__doc__])
    return func
  

def append_doc_from(reference_func,delimiter=''):
    """
    Decorator that appends the docstring of `reference_func` to the docstring of the decorated function. supersed update_docstring.
    
    Usage:
    
        import matplotlib.pyplot as plt

        @append_doc_from(plt.plot)
        def plot(x, y, **kwargs):
            """Plot x and y using custom logic."""
            return plt.plot(x, y, **kwargs)

        help(plot)  # Will show your docstring plus the original plt.plot doc
        
    """
    def decorator(func):
        ref_doc = reference_func.__doc__ or ""
        func_doc = func.__doc__ or ""
        func.__doc__='\n'.join([ref_doc,delimiter,reference_func.__name__+str(inspect.signature(reference_func))])
        func.__doc__ = func_doc.rstrip() + "\n\n---\nOriginal docstring from `{}`:\n\n{}".format(
            reference_func.__name__, ref_doc.strip()
        )
        return func
    return decorator

    '''
    # Note also this attempts
    
    from functools import update_wrapper
    #@update_wrapper(rotate_data)  #this doesn't work as I would like
    def rotate(self,angle,*args,**kwargs):
        """call data2D.rotate_data, which rotate array of an arbitrary angle in degrees in direction
        (from first to second axis)."""
        res = self.copy()
        res.data,res.x,res.y=rotate_data(self.data,self.x,self.y,angle,*args,**kwargs)
        return res
    rotate=update_docstring(rotate,rotate_data)
    '''

def strip_kw(kws,funclist,split=False,exclude=None,**kwargs):
    """ Enhanced version of pop_kw, kw can be extracted from inspection of a function (or a list  of functions).
    Defaults can be passed as kwargs.
    `exclude` is a list of keys that are kept in kws unchanged
       and not returned.
    
    2020/05/26 added `exclude`.
    Use case is presented from problems with psd_analysis.
    `psd2d.psd_analysis` calls `strip_kw(kwargs,psd2d_analysis,title="")`. 
    `ps2d_analysis` has keyword `units`, that is not used in this case (used, only if called with `analysis=True`). Here we want to preserve `units` to strip it later.
    
    Old notes:
        was in theory a safer version of the first pop_kw.
           I am not sure in what it was supposed to be "safer",
       but it was accepting functions as input.
	   It might have been used a negligible amount of times.

       sub_kw=strip_kw(kws,[sub],def1=val1,def2=val2)"""

    res=[]
    if callable(funclist): #if scalar, makes funclist a one element list
        funclist=[funclist]    
        
    for func in funclist:
        f=inspect.getfullargspec(func)
        defaults=kwargs
        l = len(f.defaults) if f.defaults is not None else 0
        fdic=dict(zip(f.args[-l:],f.defaults))
        res.append(pop_kw(kws,fdic,exclude=exclude))

    if not split:  #merge dictionaries, don't ask me how
        tmp = {}
        [tmp.update(d) for d in res]
        res=tmp

    return res
    
def prep_kw(function, kwargs, strip = False):
    """
    return a dictionary only of `kwargs` pertaining to `function`.
    Arguments can be stripped from kwargs as they are used, if strip is set to True, otherwise kwargs is left unchanged.
    This can be useful to check against extraneous args, e.g.:    
    intended use (inside a function called with **kwargs):
    
    function (*args, **kwargs):
    
        kwargs = kwargs.copy() # this is copied to avoid modification as side effect, can be avoided if you don't need a final check, in that case, simply remove strip = True from subfunction calls

        a = subfunction1(**prep_kw(function1, kwargs, strip=True))
        b= subfunction2(**prep_kw(function2, kwargs, strip=True))
        
        if len(kwargs):  # check there are not unused keywords.
            raise TypeError("invalid arguments in function.")
    
    see also https://stackoverflow.com/questions/26534134/python-pass-different-kwargs-to-multiple-functions
    
    There might be a problem with multiple-level nested functions, for example if instead of plt.plot there is a function which itself calls other functions which accept **kwargs."""
    
    if not strip:
        kwargs = kwargs.copy()
    kwlist = list(inspect.signature(function).parameters)
    return {k: kwargs.pop(k) for k in dict(kwargs) if k in kwlist}
  
  
  
