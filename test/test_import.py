"""Despite differemt conventions, from 
https://docs.python.org/3/distutils/sourcedist.html
test/test*.py is included in manifest."""

import pyxsurf
# pyxsurf.pySurf    #fails
from pyxsurf import pySurf
# >>> pySurf.hello    #fails
from pyxsurf.pySurf import hello
##  hello cane
# hello()    #fails

hello
## <module 'pyxsurf.pySurf.hello' from 'C:\\Users\\User\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python37\\site-packages\\pyxsurf\\pySurf\\hello.py'>
# hello.hello()    #fails

hello.hello_cane()