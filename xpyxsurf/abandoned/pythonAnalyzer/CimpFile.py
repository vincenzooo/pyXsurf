import namelist_class
import os
from numpy import *

class anris(namelist_class.Namelist):
    '''funzione di analisi dei risultati basata sui valori, passati
    come file o dizionario, da cui leggere la namelist delle impostazioni
    '''
    def __init__(self,nl):
        def dirfom(maindir):
            d=[df.strip(" \n") for df in open(os.path.join(maindir,"dirlist.dat")).readlines()]
            return d
        namelist_class.Namelist.__init__(self,nl)
        maindir=os.path.join(self["WORKDIR"].strip("\"' "),self["DIRORIGIN"].strip("\"' "))
        self.maindir=maindir
        self.dirfom=dirfom(maindir)

    def arrayFromGroup(self,dfn,group,ndim=-1,res="risultati00000.dat"):
        '''legge i risultati relativi alla dirfom[dfn] con dfn intero, e al
        gruppo group. manca sistema di correzione errori per righe incomplete.'''
        if dfn>len(self.dirfom):
            print "non posso leggere la dirfom [",dfn,"], "
            print "dirfom=",self.dirfom
        ndigit=len(res.split("0"))-1
        basedir=os.path.join(self.maindir,self.dirfom[dfn],str(group).zfill(3))
        i=1
        filename=[f for f in res.split("0") if f!=""]      
        n=str(i).zfill(ndigit).join(filename)
        if ndim ==-1:
            self.parametri=namelist_class.Namelist(os.path.join(self.maindir,self.dirfom[dfn],"parametri.txt"))
            self.ndim=self.parametri["NDIM"]
            self.lastExplored=(dfn,group)
        else:
            self.ndim=ndim
        data=[]
        while os.path.exists(os.path.join(basedir,n)):
            f=open(os.path.join(basedir,n),"r")
            print "reading data from: ",os.path.join(basedir,n)
            selData=f.read()
            selData=selData.split("\n")
            selData=[str(j).split()[0:self.ndim+1] for j in selData]
            selData=[j for j in selData if j <> []]
            selData=[map(float,j) for j in selData]
            data.extend(selData)
            f.close()
            i=i+1
            n=str(i).zfill(ndigit).join(filename)
        def clean(data):
            return data
        return array(clean(data))

        
#nl2=anris("2_anris.txt")
#dd = nl2.dataRead(1,2)
##
##class anris(namelist_class.Namelist):
##    '''funzione di analisi dei risultati basata sui valori, passati
##    come file o dizionario, da cui leggere la namelist delle impostazioni
##    '''
##    def __init__(self,nl):
##        def dirfom(maindir):
##            d=[df.strip(" \n") for df in open(os.path.join(maindir,"dirlist.dat")).readlines()]
##            return d
##        namelist_class.Namelist.__init__(self,nl)
##        maindir=os.path.join(self["WORKDIR"].strip("\"' "),self["DIRORIGIN"].strip("\"' "))
##        self.maindir=maindir
##        self.dirfom=dirfom(maindir)
##
##    def dataRead(self,dfn,group,res="risultati00000.dat"):
##        '''legge i risultati relativi alla dirfom[dfn] con dfn intero, e al
##        gruppo group'''
##        if dfn>len(self.dirfom):
##            print "non posso leggere la dirfom [",dfn,"], "
##            print "dirfom=",self.dirfom
##        ndigit=len(res.split("0"))-1
##        basedir=os.path.join(self.maindir,self.dirfom[dfn],str(group).zfill(3))
##        data=""
##        i=1
##        filename=[f for f in res.split("0") if f!=""]      
##        n=str(i).zfill(ndigit).join(filename)
##        while os.path.exists(os.path.join(basedir,n)):
##            f=open(os.path.join(basedir,n),"r")
##            print "reading data from: ",os.path.join(basedir,n)
##            data=data+f.read()
##            f.close()
##            i=i+1
##            n=str(i).zfill(ndigit).join(filename)
##        def clean(data):
##            return data
##        return clean(data)
        