def read_data(file,reader,*args,**kwargs):

    """read data from a file using a given reader with custom options in args, kwargs.
    
    The function calls reader, but, before this, strips all options that are recognized by register_data,
      all remaining unkown parameters are passed to reader.
    Then register_data is called with the previously stored settings (or defaults if not present).
    
    This was made to hide messy code beyond interface. See old notes below, internal behavior can be better fixed e.g. by using dataIO.dicts.pop_kw and inspect.signature and fixing header interface.
    
    Old notes say:

        Division of parameters is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is 
        possible to call the read_data procedure with specific parameters, for example in example below, the reader for 
        Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers, 
        while this can be done using read_data. 
        
        this is an ugly way to deal with the fact that return
    arguments are different if header is set, so when assigned to a variable as in patch routines in pySurf instrumentReader it fails.
    Workaround has been calling directly read_data, not optimal."""
    if kwargs.pop('header',False):
        try:
            return reader(file,header=True)
        except TypeError:  #unexpected keyword if header is not implemented
            return None
            
    
    #filters register_data parameters cleaning args
    # done manually, there is a function in dataIO.
    scale=kwargs.pop('scale',(1,1,1))
    crop=kwargs.pop('crop',None)
    #zscale=kwargs.pop('zscale',None) this is removed and used only for reader
    # functions where a conversion is needed (e.g. wavelength)
    center=kwargs.pop('center',None)
    strip=kwargs.pop('strip',False)
    ##regdic={'scale':scale,'crop':crop,'center':center,'strip':strip}
    
    data,x,y=reader(file,*args,**kwargs)
    return register_data(data,x,y,scale=scale,crop=crop,
    ## #try to modify kwargs 
    ## for k in list(kwargs.keys()): kwargs.pop(k)  #make it empty
    ## #kwargs.update(regdic)
    ## for k in regdic.keys():kwargs[k]=regdic[k] 
    ## if register:
    ##    data,x,y=register_data(data,x,y,scale=scale,crop=crop,
    ##    center=center,strip=strip)
    ## return data,x,y
