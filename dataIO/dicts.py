"""contains operations on dictionaries"""
import warnings
import inspect
import os

def make_file_dict(filelist,delimiter='_',index=0):
    """make dictionary from full path file names
    grouping filenames in lists with same key.
    key is obtained by file base name by string.split
    on delimiter (case insensitive)."""
    
    filedic={}
    for d in filelist:
        k=os.path.basename(d).lower().split(delimiter)[index]
        try:
            filedic[k].append(d)
        except KeyError:
            filedic[k]=[d]

    return filedic

def pop_kw(kws,defaults,keylist=None):
    """A multiple pop function for a dictionary, hopefully useful to filter kws and manage function parameters.
    
    a copy of `defaults` is returned where values are stripped from kws if a key is present. 
    If **kws are passed to a function that calls two other subs, each one with optional arguments, it can be used as (see full example below):
    res1=sub1(arg1,**pop_kw(kws,{'mes1':'blah!'}))
    res2=sub2(arg1,**kws)

    or values can be retrieved in variables (and sorted):
    mes2,mes1=pop_kw(kws,{'mes1':'blah!'},['mes2','mes1'])
    print (mes1,mes2)
    
    Note however that only parameters explicitly listed in defaults are filtered. A safer version for filtering function parameters rather than generic dictionaries is strip_kw.
    
    """
    res={}
    keys=list(defaults.keys())
    for k in keys:
        res[k]=kws.pop(k,defaults[k])
    
    if keylist is not None:
        try:
            res=[res[k] for k in keylist]
        except KeyError as e:
            print ("ignoring key invalid in dict %s:\n'"%(res))
            warnings.warn(str(e),RuntimeWarning)
            
    return res

def strip_kw(kws,funclist,split=False,**kwargs):
    """Given a list of functions and a dictionary of parameters,
       
       sub_kw=strip_kw(kws,[sub],def1=val1,def2=val2)"""
    
    res=[]
    if callable(funclist):
        funclist=[funclist]
    for func in funclist:
        f=inspect.getfullargspec(func)
        defaults=kwargs 
        l = len(f.defaults) if f.defaults is not None else 0
        fdic=dict(zip(f.args[-l:],defaults))
        res.append(pop_kw(kws,fdic))
    
    if not split:  #merge dictionaries, don't ask me how
        tmp = {}
        [tmp.update(d) for d in res]
        res=tmp
        
    return res
    
def example_pop_kw():
    def function(arg1,**kwargs):
        '''a function with 1 arg'''
        def sub1(arg,mes1='--'):
            print (arg,mes1)
        def sub2(arg,mes2='--'):
            print (arg,mes2)
            
        res1=sub1(arg1,**pop_kw(kwargs,{'mes1':'blah!'}))
        
        res2=sub2(arg1,**kwargs)
        return res1,res2
    
    print (function('This is:',mes1='mes1!'))
    print (function('This is:',mes1='mes1!',mes2='mese2!'))
    
def test_pop_kw():
    #without keylist return a dictionary
    kwargs={'cane':'dog','units':'mm'}
    kk={'units':3,'nsigma':1}
    print('start:\n',kwargs,kk,'\n---\n')
    res=pop_kw(kwargs,kk)
    print(kwargs,'\n',kk,'\n',res)
    print("#---------------\n")
    
    #keylist sorts values, a list is returned
    kwargs={'cane':'dog','units':'mm'}
    kk={'units':3,'nsigma':1}
    print('start:\n',kwargs,kk,'---\n')
    res=pop_kw(kwargs,kk,['nsigma','units'])
    print(kwargs,'\n',kk,'\n',res)
    print("#---------------\n")    
    
    #same, but one of the keys is invalid, ignore
    kwargs={'cane':'dog','units':'mm'}
    kk={'units':3,'nsigma':1}
    print('start:\n',kwargs,kk,'---\n')
    res=pop_kw(kwargs,kk,['nsigma','units','gatto'])
    print(kwargs,'\n',kk,'\n',res)
    print("#---------------\n")    
    
if __name__=="__main__":
    test_pop_kw()