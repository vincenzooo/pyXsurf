
from dataIO.files import program_path
from dataIO.files import program_path0

def test_program_path():
    """copied from test function in    
            
       called from script importing from this file:
            # .py file contains a copy of this function.
            %run .../dataIO/test/test_program_path.py
            
            caller's path: ...\pyXTel\dataIO\files.py
            program path: None
            
    """ 
    p = program_path0()  
    print("caller's path: ", p )
    
    p = program_path()  
    print("caller's path: ", p )
    return  p


p = test_program_path()

