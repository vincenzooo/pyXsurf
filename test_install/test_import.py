"""Despite differemt conventions, from 
https://docs.python.org/3/distutils/sourcedist.html
test/test*.py is included in manifest."""

# import pyxsurf
# pyxsurf.pySurf    #fails

print ("TRY: import pySurf")
print("OK\n")

# >>> pySurf.data2D_class    #fails
print ("import Data2D pyxsurf.pySurf.data2D_class ")
from pySurf.data2D_class import Data2D
print("OK\nData2D is:") 
print(Data2D)
print("\n")
## <module 'pyxsurf.pySurf.hello' from 'C:\\Users\\User\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python37\\site-packages\\pyxsurf\\pySurf\\hello.py'>
# hello.hello()    #fails

print("Initialize empty Data2D object in `a`:") 
a = Data2D()
print("---")
print(a)

print("Import points from pySurf as a module:") 
from pySurf import points
print (points)