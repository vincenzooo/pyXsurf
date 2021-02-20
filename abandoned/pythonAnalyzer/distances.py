import os
import math
import numpy as np

'''versione con array '''
    
def eucDist(p1,p2):
    '''demenziale distanza euclidea'''
    if isinstance(p1,float):
        p1=[p1]
    if isinstance(p2,float):
        p2=[p2]
    if len(p1)!=len(p2): raise "different len p1p2"
    d=0
    for i in range(len(p1)):
        d=d+(p1[i]-p2[i])**2
    d=math.sqrt(d)
    return d

def maxDist(l1,l2):
    '''massimo delle distanze tra elementi corrispondenti delle liste l1 e l2'''
    if not isinstance(l1,list):
        l1=[l1]
    if not isinstance(l2,list):
        l2=[l2]
    if len(l1)!=len(l2): raise "different len l1l2"
    d=0
    for i in range(len(l1)):
        if abs(l1[i]-l2[i])>d:d=abs(l1[i]-l2[i])
    return d

def powerlaw5(p):
    nbil=100  
    a,b,c,g1,gl=p
    gm=(gl-g1)/(nbil-1)
    gq=g1
    #print a,b,c,gm,gq
    gi = lambda i:i*gm+gq
    pl = lambda i:a/(i-b)**c
    spes=[[pl(i)*gi(i),pl(i)*(1-gi(i))] for i in range(1,nbil+1)]
    s=[]
    for i in spes: s.extend(i)
    return s

def spesDist(l1,l2):
    '''massimo delle distanze tra elementi corrispondenti delle liste l1 e l2'''
    p1,p2=powerlaw5(l1),powerlaw5(l2)
    if len(p1)!=len(p2): raise "different len p1p2"
    d=0
    for i in range(len(p1)):
        print (i,d, p1[i],p2[i])
        if abs(p1[i]-p2[i])>d:d=abs(p1[i]-p2[i])
    return d
  