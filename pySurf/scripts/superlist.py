
"""functions operating on a list of Data2D objects"""
# turned into a class derived from list 2020/01/16, no changes to interface,
# everything should work with no changes.

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pdb

class superlist1(list):
    """Test class that vectorizes methods. Basic, implements properties,
        but not methods."""
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = [obj.__getattr__(self, name) for obj in self] #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        result = []
        if hasattr(attr, '__call__'):
            def newfunc(*args, **kwargs):
                #pdb.set_trace()
                print('before calling %s' %a.__name__)
                result = a(*args, **kwargs)
                print('done calling %s' %a.__name__)
                return result
            return newfunc
        else:
            return a

        return result  #it works as a property. As method, this returns a list of methods, but since the method is called as sl.method(),
        #gives an error as it tries to call the list.
        # deve ritornare una funzione che ritorna una lista.

class superlist2(list):
    """Test class that vectorizes methods."""
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = [obj.__getattr__(self, name) for obj in self] #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        """
        if hasattr(attr[0], '__call__'):
            attr=attr[0]
            def newfunc(*args, **kwargs):
                #pdb.set_trace()
                print('before calling %s' %attr.__name__)
                result = attr(*args, **kwargs)
                print('done calling %s' %attr.__name__)
                return result
            return newfunc
        else:
            return [a for a in attr]
        """
        
        result = []
        for a in attr:
            #print('loop ',a)
            #pdb.set_trace()
            if hasattr(a, '__call__'):
                result=np.vectorize(a)
                """
                def newfunc(*args, **kwargs):
                    #pdb.set_trace()
                    print('before calling %s' %a.__name__)
                    result = a(*args, **kwargs)
                    print('done calling %s' %a.__name__)
                    return result
                result.append(newfunc)
                """
            else:
                result.append(a)

        return result  #it works as a property. As method, this returns a list of methods, but since the method is called as sl.method(),
        #gives an error as it tries to call the list.
        # deve ritornare una funzione che ritorna una lista.
        
class superlist3(list):
    """Test class that vectorizes methods.  """
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = [obj.__getattr__(self, name) for obj in self] #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        result = []
        for a in attr:
            if hasattr(a, '__call__'):
                result=np.vectorize(a)
            else:
                result.append(a)

        return result  #it works as a property. As method, this returns a list of methods, but since the method is called as sl.method(),
        #gives an error as it tries to call the list.
        # deve ritornare una funzione che ritorna una lista.        

def test_superlist(cls,obj=None):

    print ('test class ',cls)
    if obj is None:
        #s=type(cls,[np.arange(4),np.arange(3)])
        s = cls([np.arange(4),np.arange(3)])
        #s = superlist([np.arange(4),np.arange(3)])

    print('original data:')
    print(s)
    print('\ntest property (np.shape):')
    print(s.shape)
    print('\ntest method (np.flatten):')
    print(s.flatten())
    
    
if __name__ == "__main__":
    test_superlist(superlist1)
        #test_superlist(superlist)
    """
    except: # catch *all* exceptions
        e = sys.exc_info()[0]
        print( "Error: %s" % e )
    """