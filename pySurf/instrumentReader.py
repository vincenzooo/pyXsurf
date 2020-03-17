print ("module moved, modify your import to use readers.instrumentReader")


# from .readers import instrumentReader 

import sys
#del sys.modules['instrumentReader']
#sys.modules['instrumentReader'] = __import__('Mod_2')

sys.modules['instrumentReader'] = __import__('.readers.instrumentReader')
#return ir

