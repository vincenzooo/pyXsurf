import Cdata
from CQTclust import *
import CimpFile
     
if __name__=="__main__":
    #toll=0.5
    #nomefile="risultati00001.dat"
    #ndim=5

    anris=CimpFile.anris("2_anris.txt")
    results=anris.arrayFromGroup(1,1,5)

    '''    
    starting,ending=startendFromFile(nomefile,ndim)
    endBest=resDir(ending).sel()
    funkVal={}
    for i,e in enumerate(endBest):funkVal[i]=endBest[:,ndim]
    
    clusters=QTclust(funkVal,toll)
    endBest.put("endBest.txt")
 
    clusterati=[]   
    f=open("clusterati.txt","w")
    for cluster in clusters:
        clusterati.append([])
        for el in cluster:
            clusterati[-1].append(endBest[el]) 
            for e in endBest[el]:
                f.write(str(e)+"\t")
            f.write("\n")
        f.write("\n\n")
    f.close()

    clusterati=map(array,clusterati)


    plotC(clusterati,1,5)
    '''