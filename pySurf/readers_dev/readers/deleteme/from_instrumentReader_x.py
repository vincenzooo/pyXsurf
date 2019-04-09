def xread_data(file,reader,register=True,*args,**kwargs):
    """non essendo sicuro dell'interfaccia per ora faccio cosi'.
    The function first calls the (raw) data reader, then applies the register_data function to address changes of scale etc,
    arguments are filtered and passed each one to the proper routine.
    18/06/18 add all optional parameters, if reader is not passed,
    only registering is done. note that already if no register_data
    arguments are passed, registration is skipped.
    18/06/18 add action argument. Can be  'read', 'register' or 'all' (default, =read and register). This is useful to give fine control, for example to modify x and y after reading and still make it possible to register data (e.g. this is done in 
    Data2D.__init__).
    
    implementation:
    This works well, however read_data must filter the keywords for the reader and for the register and
    this is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is 
    possible to call the read_data procedure with specific parameters, for example in example below, the reader for 
    Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers, 
    while this can be done using read_data. """
    
    """this is an ugly way to deal with the fact that return
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
    center=kwargs.pop('center',None)
    strip=kwargs.pop('strip',False)
    regdic={'scale':scale,'crop':crop,'center':center,'strip':strip}
    
    data,x,y=reader(file,*args,**kwargs)
    
    #try to modify kwargs 
    for k in list(kwargs.keys()): kwargs.pop(k)  #make it empty
    #kwargs.update(regdic)
    for k in regdic.keys():kwargs[k]=regdic[k] 
    if register:
        data,x,y=register_data(data,x,y,scale=scale,crop=crop,
        center=center,strip=strip)
    return data,x,y