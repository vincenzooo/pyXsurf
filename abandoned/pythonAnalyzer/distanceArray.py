import os
import math
from Numeric import *

'''versione con array '''
    
def eucDist(p1,p2):
    '''demenziale distanza euclidea'''
    if isinstance(p1,float):
        p1=[p1]
    if isinstance(p2,float):
        p2=[p2]
    if len(p1)<>len(p2): raise "different len p1p2"
    d=0
    for i in range(len(p1)):
        d=d+(p1[i]-p2[i])**2
    d=math.sqrt(d)
    return d
    
def startend(nomefile,ndim):
    '''dal file dei risultati restituisce liste con punti iniziali e finali'''
    f=open(nomefile,"r")
    d=f.readlines()
    d=[str(i).split()[0:ndim+1] for i in d]
    d=[i for i in d if i <> []]
    d=[map(float,i) for i in d]
    d=array(d)
    print "*", d
    
    starting=[]
    ending=[]
##    for i,v in enumerate(d):
##        if i%(2*(ndim+1))<= ndim:
##            starting.append(map(float,v))
##        else:
##            ending.append(map(float,v))
##    f.close()
    return starting,ending

def valueDic(resVec,ndim):
    '''dato un vettore di tutti i risultati finali crea un dizionario
    con il migliore risultato di ogni simplex come valore,
    e il numero del simplex come chiave'''
    funkVal={}
    for i,e in enumerate(resVec[0::ndim+1]):
        funkVal[i]=e[ndim]
    return funkVal

##def clusterDiam(cluster,distance):
##    '''restituisce il diametro del cluster basato sulla distanza distance.
##    il cluster e' passato come una lista di punti (eventualmente liste)'''
##    #per ora stupida distanza massima dal primo punto del cluster
##    for i in cluster:
        
    
def QTclust(punti,distance,Dmax):
    ''''''
    cc=[] #candidate clusters
    for i,v in punti.iteritems():
        dis={}
        for i2,v2 in punti.iteritems():
            dis[distance(v,v2)]=i
            #print v,v2,dis
            
            
        

def analizza(nomefile,ndim):
    starting,ending=startend(nomefile,ndim)
    funkVal=valueDic(ending,ndim)
    clusters=QTclust(funkVal,eucDist,toll)
    return funkVal

        
if __name__=="__main__":
    toll=1e-4
    test=analizza("risultati00001.dat",5)

    

             