import os
import math
from numpy import *
from pylab import *
from .distances import *
from functools import reduce

defaultDist=eucDist
    
def startend(d,ndim):
    '''da lista di punti estratta dal file dei risultati
    restituisce liste con punti iniziali e finali
    (ogni punto e' una lista di valori,l'ultimo e' il valore della funzione)'''
    starting=[]
    ending=[]
    for i,v in enumerate(d):
        if i%(2*(ndim+1))<= ndim:
            starting.append(list(map(float,v)))
        else:
            ending.append(list(map(float,v)))
    return starting,ending

def startendFromFile(nomefile,ndim):
    '''legge i dati dal file dei risultati e lancia
    la sub che restituisce liste con punti iniziali e finali
    (ogni punto e' una lista di valori,l'ultimo e' il valore della funzione)'''
    f=open(nomefile,"r")
    d=f.readlines()
    d=[str(i).split()[0:ndim+1] for i in d]
    d=[i for i in d if i != []]
    d=[list(map(float,i)) for i in d]
    return startend(d,ndim)
    f.close()

def valueDic(resVec,ndim):
    '''dato una lista di punti (ognuno e' una lista dei valori delle coordinate,
    in cui l'ultimo e' il valore della funzione nel punto), crea un dizionario
    con le coordinate di un punto ogni ndim punti (il migliore x ogni simplex, nel caso dei programmi isoxm)
    e il numero del punto (simplex) come chiave'''
    funkVal={}
    for i,e in enumerate(resVec[0::ndim+1]):
        funkVal[i]=e[ndim]
    return funkVal

##def clusterDiam(cluster,distance):
##    '''restituisce il diametro del cluster basato sulla distanza distance.
##    il cluster e' passato come una lista di punti (eventualmente liste)'''
##    #per ora stupida distanza massima dal primo punto del cluster
##    for i in cluster:



def distanceTable(pList,dist=defaultDist):
    '''da una lista di punti pList, crea la tabella delle distanze con
    la distanza dist'''
    ndim=len(pList)
    #a=array(pList)
    #tab=dist(a,a[:,NewAxis])
    t=[dist(u,v) for u in pList for v in pList]
    t=resize(t,(ndim,ndim))
    return t
    
    
def QTclust(punti,Dmax,distance=defaultDist):
    '''punti e' un dizionario con indice come chiave e valori su cui calcolare
    la distanza distance. Dmax e' il raggio massimo del cluster.
    restituisce una lista di liste ognuna delle quali e' un gruppo di indici che
    formano un cluster. usa algoritmo "quality threshold clustering (QTclust)"
    in caso di cluster candidati con stesso numero di membri sceglie quello di raggio
    minore'''

    def clustRad(cc,n,disTab):
        '''dato un candidato cluster cc e l'indice n di un punto nella tavola delle
        distanze disTab, restituisce la massima distanza del punto n da ogni punto del
        cluster.'''
        SelCol=choose(cc,disTab[n])
        m=reduce(maximum,SelCol)
        return m
    
    np=len(punti)
    p=[v for v in list(punti.values())] #p e' una lista con i valori
    k=[v for v in list(punti.keys())] #k con le chiavi
    disTab=distanceTable(p,distance)
    clusterList=[]
    max_ccm=0
    minrad=0
    for j in range(np):
        cc=[] #candidate cluster
        ccrad=0
        col=disTab[j]
        interestArg= nonzero(less(col,Dmax))    #array di indici di valori che possono essere presi in considerazione
        if len(interestArg)<max_ccm:          #e' il massimo numero teorico di membri per questo cluster
            continue     #se meno di quelli del miglior cluster candidato..
        #crea un nuovo array solo con i valori da considerare, interestArg tiene conto della corrispondenza
        #con l'array originario
        interestVal= compress(less(col,Dmax),col) #array di valori che possono essere presi in considerazione
        sv=argsort(interestVal) #indici ordinati dal piu' piccolo al + grande

        #crea cluster candidato
        cc.append(j)
        for cp in sv[1:]:
            cpArg=interestArg[cp]   #cpArg indice corrispondente nel vettore punti
            cr=clustRad(cc,cpArg,disTab)
            if cr<=Dmax:
                cc.append(interestArg[cp])
                if cr>ccrad:ccrad=cr
        ccm=len(cc)

        #decide se e' il candidato migliore finora trovato
        if ccm>max_ccm or (ccm==max_ccm and ccrad<minrad):
            #print ccm>max_ccm, ccm==max_ccm , ccrad<minrad, (ccm==max_ccm and ccrad<minrad)
            max_ccm=ccm
            ccc=cc
            minrad=ccrad
            #potrebbe celare ambiguita', cluster con stesso numero di membri,ma membri diversi e
            #stesso raggio, non garantisce unicita' del risultato
    clusterList.append([k[i] for i in ccc])
    #print punti,[k[i] for i in ccc],"\n"
    for i in ccc: del punti[k[i]]
    if len(punti)!=0: clusterList.extend(QTclust(punti,Dmax,distance))
    return clusterList
            
      
def plotC(clusterati, ii, jj=-1):
    '''plotta le colonne ii e jj dei dati clusterati, con un colore diverso per ogni
    cluster'''
    
    col="bgrcmyk"
    for i,c in enumerate(clusterati):
        nc=col[i%len(col)]
        if (jj==-1):
            plot(c[:,ii],nc+'o')
        else:
            plot(c[:,ii],c[:,jj],nc+'o')
    show()

class resDir :
    def __init__(self,ar,ndim=0):
        self.data=array(ar)
        self.ndim=ndim
        if ndim==0:
            self.ndim=len (ar[0])-1
            print("numero di dimensioni non dato!")
            print("fissato in" , self.ndim)
            #print dir(self)
    def __getitem__ (self,index):
        return self.data[index]
    def __repr__(self):
        s=""
        for res in self.data:
            for el in res:
                s=s+(str(el)+"\t")
            s=s+("\n")
        return s
    def sel(self,nd=-1):
        if nd==-1:nd=self.ndim+1
        bd=self.data[0::nd]
        n=resDir(bd,nd)
        return n
    def put(self,file):
        f=open(file,"w")
        f.write(str(self))
        f.close()
    
        

          

    

             
