"""contains operations on dictionaries"""
import inspect

import numpy as np
from copy import deepcopy



filterByKey2 = lambda data,keys : {key: data[key] for key in keys if key in data}

def vectorize(kwargs,n):
    if kwargs : #passed explicit parameters for each reader
        # Note, there is ambiguity when rfiles and a kwargs value have same
        # number of elements()
        #pdb.set_trace()
        #vectorize all values
        for k,v in kwargs.items():
            if (np.size(v) == 1): 
                kwargs[k]=[v for i in range(n)]    
            #elif (len(v) != len(rfiles)):
            else:
                try:
                    kwargs[k]=[v.copy() for i in range(n)]
                except AttributeError:  #fails e.g. if tuple which don't have copy method
                    kwargs[k]=[v for i in range(n)]  
            # else:  #non funziona perche' ovviamente anche chiamando esplicitamente, sara'
            # # sempre di lunghezza identica a rfiles.
            #    print ('WARNING: ambiguity detected, it is not possible to determine'+
            #    'if `%s` values are intended as n-element value or n values for each data.\n'%k+
            #    'To solve, call the function explicitly repeating the value, es. `units=[['um','um','nm']]*3.`')
            # in realtà la forma "esplicità" fallisce al plot, mi fa temere che non funzioni anche se le immagini l'altra funziona. 
            
    
        # 2020/07/10 args overwrite kwargs (try to avoid duplicates anyway).
        # args were ignored before.
        
        #if not args:  #assume is correct number of elements
        #    args = [[]]*len(rfiles)   ## Non va fatto cosi'!! senno' duplica "by ref"
        
        #pdb.set_trace()
        
        #transform vectorized kwargs in list of kwargs
        kwargs=[{k:deepcopy(v[i]) for k,v in kwargs.items()} for i in range(n)]
    else:
        kwargs = [{} for i in range(n)] 
        
    return kwargs

def pop_kw(kws,defaults=None,keylist=None,exclude=None):
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

    2020/05/26 added `exclude` keywork, as a consequence of adding to strip_kw.
    Completely reviewed 2019/04/08, changed interface and function of keylist.


    """
    res={}
    keys=list(defaults.keys())
    
    if exclude is None: exclude = [] #makes a list, so I can use `in`
    for k in keys:
        #print("%s %s %s %s"%(k,defaults[k],res,kws))
        if k in exclude:
            pass
            #print ('Excluded')
        else:
            res[k]=kws.pop(k,defaults[k])
    #print("final: %s %s"%(res,kws))
    
    if keylist is not None:
        if isinstance(keylist,str): keylist=[keylist] #make list if only one element
        for k in keylist:
            if k in kws:
                res[k]=kws.pop(k)

    return res

# 2024/02/26 strip_kw and prep_kw moved to dataIO.functions
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

    
def print_tree(tree, depth = 0, maxlen = None):
    """print dictionary as tree, recursively calling itself,
    based on http://www.siafoo.net/snippet/91"""
    
    if tree == None or len(tree) == 0:
        #print () #"--" * depth)
        pass
    else:
        for key, val in tree.items():
            #print ("\t" * depth, key)
            #printTree(lst_view_path(val), depth+1)    
            
            if isinstance(val,dict): #folder:
                print ("\t" * depth+'|', key.upper(),"----\\")  #print ("\t" * depth, key.upper())
                print_tree(val, depth+1)    
            else:  #file:
                v = val if maxlen is None else val[:maxlen]
                print ("\t" * depth+'|', key,':',v)  #print ("\t" * depth, key,':',val[:6])
                # non so perche' qui sopra si limita a 6, forse per array grandi, ma non va bene per altri dati.
               
 
def test_filterByKey():
	eegKeys = ["FP3", "FP4"]
	gyroKeys = ["X", "Y"]

	# 'Foo' is ignored
	data = {"FP3": 1, "FP4": 2, "X": 3, "Y": 4, "Foo": 5}
	print(filterByKey(data,eegKeys))

def example_pop_kw():
    """Test of `pop_kw` coupled with a function. This mechanism is extended and used in `dataIO.functions`
    to implement functions `strip_kw` and `prep_kw` which use `inspect` to automatically handle function keywords.
    `pop_kw` works with generic dictionaries."""
    
    
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


def test_pop_kw(function=pop_kw):#function = None):

    """run a set of dict stripping tests using a function with interface compatible with `pop_kw`.
    uses pop_kw if function is not provided. This is implemented in dataIO.functions.prep_kw` and `.strip_kw`."""

    from inspect import signature
    
    print (signature(function))
    print (function.__doc__)

    print("Without keylist return a dictionary using only keys in defaults, used keys are removed from kwargs:\n")
    kwargs={'dog':'bau','cat':'miao','cow':'muu'}
    defs={'dog':'barf','sheep':'bee'}
    print('start -> kwargs = {},\n defaults = {}'.format(kwargs,defs))
    res=function(kwargs,defaults=defs)  #remove from kwargs the keys in defs assigning default if they don't exist in kwargs
    print("\nResult: ",res)
    # {'sheep': 'bee', 'dog': 'bau'}
    print('end ->\n kwargs = {},\n defaults = {}'.format(kwargs,defs))
    print("#---------------")

    print("Keys to keep can be specified separately passing keylist:\n")
    kwargs={'dog':'bau','cat':'miao','cow':'muu'}
    defs={'dog':'barf','sheep':'bee'}
    keys=['cat','sheep']
    print('start -> kwargs = {},\n keylist = {},\n defaults = {}\n'.format(kwargs,keys,defs))
    res=function(kwargs,defaults=defs,keylist=keys)
    print("\nResult: ",res)
    # {'sheep': 'bee', 'cat': 'miao', 'dog': 'bau'}
    print('end ->\n kwargs = {},\n keylist = {},\n defaults = {}'.format(kwargs,keys,defs))
    print("#---------------\n")

    print("Same, with added keys, when not present in kwargs, they are ignored:\n")

    kwargs={'dog':'bau','cat':'miao','cow':'muu'}
    defs={'dog':'barf','sheep':'bee'}
    keys = ['nsigma','units','gatto']
    print('start ->\n kwargs = {},\n keylist = {},\n defaults = {}\n'.format(kwargs,keys,defs))
    res=function(kwargs,defs,keys) #gatto doesn't exist in kwargs
    print("\nResult: ",res)
    # ignoring key invalid in dict {'units': 'mm', 'nsigma': 1}:
    #'
    #{'sheep': 'bee', 'dog': 'bau'}
    print('end ->\n kwargs = {},\n keylist = {},\n defaults = {}'.format(kwargs,keys,defs))
    print("#------------")

    print("Use `exclude`, keys in this list are left untouched in kwargs.\n")

    kwargs={'dog':'bau','cat':'miao','cow':'muu'}
    defs={'dog':'barf','sheep':'bee'}
    exclude = ['dog']
    print('start ->\n kwargs = {},\n exclude = {},\n defaults = {}\n'.format(kwargs,exclude,defs))
    res=function(kwargs,defs,exclude=exclude) #gatto doesn't exist in kwargs
    print("\nResult: ",res)
    # res =  {'sheep': 'bee'} #dog is not modified or popped
    # kwargs = {'dog': 'bau', 'cat': 'miao', 'cow': 'muu'}
    print('end ->\n kwargs = {},\n exclude = {},\n defaults = {}'.format(kwargs,exclude,defs))
    print("#------------")

if __name__=="__main__":
    test_pop_kw()