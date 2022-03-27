

def chk_args_id(func,n=None,*args,**kwargs):
    """check the first n arguments id and print a warning if
    they don't match the first n return arguments."""

    def wrapper(*args,**kwargs):

        ids = [id(a) for a in args]

        res=func(*args,**kwargs)
        
        ids2 = [id(a) for a in res]
        
        if ids != ids2:
            print("ids not matching.")
        else: 
            print("ids matching")
        print ("ids (old,new):\n",ids,ids2)
    return wrapper
    
    
def test_func(x,y,z,replacez=False):
    if replacez: z='replaced' #this should make the test fail
     
    return x,y,z
    
def test_decorator():
    
    print (test_func(1,2,3))
    t=chk_args_id(test_func)
    t(1,2,3)
    
    print (test_func(1,2,3,True))
    t=chk_args_id(test_func)
    t(1,2,3,True)
    
    """
    a,b,c=[1,2],-6,['cane']
    print (test_func(a,b,c))
    t=chk_args_id(test_func)
    t(a,b,c)
    
    print (test_func(a,b,c,True))
    t=chk_args_id(test_func)
    t(a,b,c,True)
    """

def decorate_module(mod,dec):
    """decorate all functions in a given module passed as string. it probably will not work, how export to caller routines?

    modified from:  https://stackoverflow.com/questions/8718885/import-module-from-string-variable?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    """
    
    import importlib

    i = importlib.import_module(mod)

    import types
    for k,v in vars(i).items():
        if isinstance(v, types.FunctionType):
            vars(i)[k] = dec(v)    
    
if __name__=="__main__":
    test_decorator()