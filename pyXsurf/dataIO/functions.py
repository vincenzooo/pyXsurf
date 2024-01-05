import inspect  # to use with importing docstring

def update_docstring(func,source,delimiter=''):
    """given a current function and a source function, update current function docstring
     appending signature and docstring of source.

     It provides a decorator @update_docstring(source) update the decorated
       function docstring. User should take care to create docstring of decorated function
       only as preamble to source docstrings, e.g.:
       this is func function, derived from source function by modifying the usage of parameter foo.
       parameter foo is changed of sign."""
       
       
    doc0="" if func.__doc__ is None else func.__doc__  # retrieve current docstring
    func.__doc__='\n'.join([doc0,delimiter,source.__name__+str(inspect.signature(source)),source.__doc__])
    return func