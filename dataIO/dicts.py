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


filterByKey2 = lambda data,keys : {key: data[key] for key in keys if key in data}

def pop_kw(kws,defaults=None,keylist=None):
    """A multiple pop function for a dictionary, hopefully useful to filter kws and manage function **kwargs parameters.

    Return a dictionary of values corresponding to kws[k] for k in keylist by popping
      the element from kws.
    If k is present in a dictionary `defaults`, take the values from `defaults` if not present in kws.
      return a dictionary {key:val} with val from `kws` or `defaults` if in any of them.
    Another way of seeing the return value is as a copy of `defaults` where values are replaced
      (after popping them) from same key in kws.


        kwargs={'dog':'bau','cat':'miao'}
        defs={'dog':'barf','sheep':'bee'}

        #pop 'cat' and 'sheep' if they are in kwargs or take them from defaults if there,
        #  put them in res
        >>> res=pop_kw(kwargs,defaults=defs)
        >>> kwargs
        Out[103]: {'cat': 'miao'}
        >>> res
        Out[104]: {'dog': 'barf', 'sheep': 'bee'}

        #pop 'cat' and 'sheep' if they are in kwargs or take them from defaults if there,
        #  put them in res
        >>> res=pop_kw(kwargs, defaults = defs, keylist = ['cat','sheep'])
        >>> kwargs
        Out[103]: {}
        >>> res
        Out[104]: {'cat': 'miao', 'dog': 'bau', 'sheep': 'bee'}


    A typical usage can be dividing arbitrary **kwargs between different subroutines,
      where the subroutines might not accept **kwargs, and be constrained to specific list of parameters.
      To pass kwargs in this circumstance do inside the routine:

    Module `inspect` can be used to retrieve subfunction parameter names and make a safe call, this is done automatically in `strip_kw`, equivalently to:

       import inspect
       pars=list(inspect.signature(f1).parameters) #give list of all accepted parameters in f1
       res1=f1(arg1,**pop_kw(kwargs,keylist = pars)) #note that pars will have keys 'args' and 'kwargs' if
                            # f1 accepts arbitrary arguments. kws and defaults with same name can be
                            #  in conflict

    In general, if **kws are passed to a function that calls two other subs, each one with optional arguments, it can be used as (see full example in `dataIO.dicts.example_pop_kw`):

        def example_pop_kw(**kwargs):

            # here if passed directly with `res0=f1(arg1,**kwargs)` gives error because mes2 is not a defined parameter
            res1=f1(arg1,**pop_kw(kwargs,{'mes1':'blah!'},['mes1'])) #only mes1 is passed to res1 with default if not in kwargs:
            res2=f2(arg1,**kwargs)  #other remaining kwargs can be safely passed to f2

    Completely reviewed 2019/04/08, changed interface and function of keylist.


    """
    res={}
    keys=list(defaults.keys())
    for k in keys:
        res[k]=kws.pop(k,defaults[k])

    if keylist is not None:
        if isinstance(keylist,str): keylist=[keylist] #make list if only one element
        for k in keylist:
            if k in kws:
                res[k]=kws.pop(k)

    return res

    '''
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
	'''

def strip_kw(kws,funclist,split=False,**kwargs):
    """was in theory a safer version of the first pop_kw.
           I am not sure in what it was supposed to be "safer",
       but it was accepting functions as input.
	   It might have been used a negligible amount of times.

       sub_kw=strip_kw(kws,[sub],def1=val1,def2=val2)"""

    res=[]
    if callable(funclist): #if scalar, makes it a one element list
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

def test_filterByKey():
	eegKeys = ["FP3", "FP4"]
	gyroKeys = ["X", "Y"]

	# 'Foo' is ignored
	data = {"FP3": 1, "FP4": 2, "X": 3, "Y": 4, "Foo": 5}
	print(filterByKey(data,eegKeys))

def example_pop_kw():

    def f1(arg,mes1='--'):
        print (arg,mes1)
    def f2(arg,mes2='--'):
        print (arg,mes2)

    def function(arg1,**kwargs):
        '''a function with 1 arg. pop_kw is used to manipulate and smartly filter `**kwargs` between two functions,
          `f1(arg,mes1='--')` and `f2(arg,mes2='--')`, so not to pass unexpected arguments.
          `res0=f1(arg1,**kwargs)` #would give error.'''

        print ("arguments passed to functions: \n arg:",arg1,"\nkwargs:",kwargs)

        # here if passed directly with `res0=f1(arg1,**kwargs)` because mes2 is not a defined parameter
        res1=f1(arg1,**pop_kw(kwargs,'mes1',{'mes1':'blah!'})) #only mes1 is passed to res1 with default if not in kwargs:

        #other remaining kwargs can be passed to f2
        res2=f2(arg1,**kwargs)

        return res1,res2


    function('This is:',mes2='mes2!')
    print('**')
    function('This is:',mes1='mes1!')
    print('**')
    function('This is:',mes1='mes1!',mes2='mese2!')
    print('**')
    function('This is:',mes1='mes1!',mes2='mese2!')

#from dataIO.dicts import pop_kw, strip_kw
def test_pop_kw():

    from inspect import signature
    print (signature(pop_kw))
    print (pop_kw.__doc__)

    print("Without keylist return a dictionary using only keys in defaults, used keys are removed from kwargs:\n")
    kwargs={'dog':'bau','cat':'miao','cow':'muu'}
    defs={'dog':'barf','sheep':'bee'}
    print('start -> kwargs = {},\n defaults = {}'.format(kwargs,defs))
    res=pop_kw(kwargs,defaults=defs)  #remove from kwargs the keys in defs assigning default if they don't exist in kwargs
    print("\nResult: ",res)
    # {'sheep': 'bee', 'dog': 'bau'}
    print('end ->\n kwargs = {},\n defaults = {}'.format(kwargs,defs))
    print("#---------------")

    print("Keys to keep can be specified separately passing keylist:\n")
    kwargs={'dog':'bau','cat':'miao','cow':'muu'}
    defs={'dog':'barf','sheep':'bee'}
    keys=['cat','sheep']
    print('start -> kwargs = {},\n keylist = {},\n defaults = {}\n'.format(kwargs,keys,defs))
    res=pop_kw(kwargs,defaults=defs,keys)
    print("\nResult: ",res)
    # {'sheep': 'bee', 'cat': 'miao', 'dog': 'bau'}
    print('end ->\n kwargs = {},\n keylist = {},\n defaults = {}'.format(kwargs,keys,defs))
    print("#---------------\n")

    print("Same, with added keys, when not present in kwargs, they are ignored:\n")

    kwargs={'dog':'bau','cat':'miao','cow':'muu'}
    defs={'dog':'barf','sheep':'bee'}
    keys = ['nsigma','units','gatto']
    print('start ->\n kwargs = {},\n keylist = {},\n defaults = {}\n'.format(kwargs,keys,defs))
    res=pop_kw(kwargs,defs,keys) #gatto doesn't exist in kwargs
    print("\nResult: ",res)
    # ignoring key invalid in dict {'units': 'mm', 'nsigma': 1}:
    #'
    #{'sheep': 'bee', 'dog': 'bau'}
    print('end ->\n kwargs = {},\n keylist = {},\n defaults = {}'.format(kwargs,keys,defs))
    print("#------------")

if __name__=="__main__":
    test_pop_kw()
