import numpy as np

def any_is_none (val):
    """Return True if any of the elements in an iterable is none.
       Needed because builtins.any fails on non iterable and np.any gives
         inconsistent results (signaled to numpy).
       Note, it is not recursive (if nested iterable, only first level is checked)."""
    return val is None if np.ndim(val)==0 else np.any([vv is None for vv in val])
        
        
        
if __name__ == "__main__":
	tv=[None,[None],[None,2],'cane']  #test values
	for v in tv: print (any_is_none(v))
