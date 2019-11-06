import numpy as np
from matplotlib import pyplot as plt



def eucDist(p1,p2):
    '''demenziale distanza euclidea'''
    d=np.sqrt(np.nansum(((p1-p2)**2)))
    return d

defaultDist=eucDist

	
def distanceTable(pList,dist=defaultDist,format=None):
    '''from list of points pList, create table of distances. 
    dist is a function setting a metric, if format is format string, if provided result is converted to string using the format'''
    
    ndim=len(pList)
    t=[dist(u,v) for u in pList for v in pList]
    t=np.resize(t,(ndim,ndim))
    if format is not None:
        t=np.char.mod(format,t)
    return t