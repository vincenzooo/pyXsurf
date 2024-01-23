print ("module moved, modify your import to use readers.instrumentReader")


# from .readers import instrumentReader 

'''
import sys
#del sys.modules['instrumentReader']
#sys.modules['instrumentReader'] = __import__('Mod_2')

sys.modules['instrumentReader'] = __import__('pySurf.readers.instrumentReader')
#return ir
'''

'''
def span(*args,**kwargs):
    from dataIO.span import span
    print "module span was moved to dataIO, please update inport in code as from dataIO.span import span."
    return span (*args,**kwargs)
'''