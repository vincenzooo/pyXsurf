#from pprint import pprint
import numpy as np

def read_blocks(filename):
    """Read a file containing blocks of numerical data separated by a white line.
    A dictionary is returned in which the first line of each block is used as a key and the following lines
    are assumed to be matrices of data with data on the same line separated by comma.
    e.g.:

    <Block name on this line (data in following lines)>
    351.08931,-165.07514,-691.24115
    352.28775,-125.48324,-691.24096
    353.64744,-78.56968,-691.37016
    355.13786,-30.21684,-691.3701
    """
    
    f = open(filename, 'r')
    content = f.read()
    f.close()
    pDic={}
    for block in content.split('\n\n'):
        if (block.strip() != ''):
            """build a dic with a set of points, first line is used as key"""
            lines=block.split('\n')
            pDic[lines[0]]=np.array([l.split(',') for l in lines[1:]],dtype=float)
    f.close()
    return pDic
    
    
def head(fn,N=10):
    """return first n lines of file `fn`, without reading the other lines.
    
    from https://stackoverflow.com/questions/1767513/read-first-n-lines-of-a-file-in-python"""
    with open(fn) as myfile:
        return [next(myfile) for x in range(N)]
    

def program_path0():
    """first version. Both are from 
    https://stackoverflow.com/questions/51487645/get-path-of-script-containing-the-calling-function
    this one is not working on case 2 of test.
    See also:
    https://note.nkmk.me/en/python-script-file-path/#:~:text=In%20Python%2C%20you%20can%20get,python%20(or%20python3%20)%20command.
    """
    import inspect
    stack = inspect.stack()
    calling_context = next(context for context in stack if context.filename != __file__)
    return calling_context.filename
    #return stack


def program_path():
    """return the path
    
    to the calling file. """
    import sys

    namespace = sys._getframe(1).f_globals  # caller's globals
    #pprint(namespace)
    return namespace['__file__']


def test_program_path():
    """cases:
        1. called from input (VScode python file): 
            <ipython-input-11-0fddf82ac4ab>     
            
        if 2. called from ipython shell running this file:
            %run .../pyXTel/dataIO/files.py
            
        or 3. importing from this function from here with:
            from dataIO.files import test_program_path
            test_program_path()
        
        gives correct result:            
            caller's path: ...\pyXTel\dataIO\files.py
            

       if 4. called from a  .py file containing a copy of 
            this function.
            
            %run .../dataIO/test/test_program_path.py
            
            caller's path: ...\dataIO\test\test_program_path.py
            
        N.B. program_path0 gives incorrect results on case 2. only.
    """ 
    p = program_path0()  
    print("caller's path: ", p )
    return  p
        
if __name__ == "__main__":
    a = test_program_path()