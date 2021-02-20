class kovint(int):
    def __init__(self,v):
        self.val=v
        int.__init__(self)

class P(object):
    pass

class Persona(object):
    def __init__(self,n):
        self.nome=n
    def __eq__(self,other):
        return self.nome == other.nome

class resDir :
    def __init__(self,ar,ndim=0):
        self.data=array(ar)
        self.ndim=ndim
        if ndim==0:
            self.ndim=len (ar[0])-1
            print "numero di dimensioni non dato!"
            print "fissato in" , self.ndim
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

from numpy import *
class resList(list):
    #questa funziona (credo), subclassa la lista e ne eredita i metodi
    def __init__(self,data=[]):
        print data
        self.__data=data
        list.__init__(self,self.__data)

class resList2(list):
    # anche questa funziona in modo ancora piu' semplice
    def __init__(self,data=[]):
        list.__init__(self,data)        

class resList3:
    #voglio provare a ottenere un effetto simile senza ereditarieta'.
    #per gli array infatti non si puo' ereditare
    def __init__(self,data=[]):
        self.data=list(data)
####
####    questo non funziona:
####    def __getattr__(self,attr):
####        print "in getattr"
####        print attr
####        #cosi' funziona, ma applicando qui dentro un metodo
####        #a self.data da' errore, per es.:
####        print self.data.sort()
####        print self.data[2]

##>>> a=resList3([1,-2,3])
##>>> a.kov()
##in getattr
##kov
##None
##3
##Traceback (most recent call last):
##  File "<interactive input>", line 1, in ?
##TypeError: 'NoneType' object is not callable

class desc(object):
    def __init__ (self,initval=None,name='var'):
        self.val=initval
        self.name=name
    def __get__(self,obj,objtype):
        print "get: ",self.name
        return self.val
        print "obj: ",obj
        print "self: ",self
    def __set__(self,obj,val):
        print "set: ",self.name
        print self.val
        print "obj: ",obj
        print "self: ",self

class my(object):
    x=desc(10,'var "x"')
    y=5


class resArr:
    def __init__(self,data=[]):
        print data
        self.__data=array(data,copy=copy)
        #arraytype.__init__(self,data)

        

class C(object):
    classattr="attr on class"
    def f(self):
        return "funzione f"
    def g():
        return "funzione g"
        


if __name__=="__main__":
    a=resList3([1,-2,3])
    #print a.kov()
    #print a[2]
    #print a[1]
    b=[4,4,2]
    a.data=b
    pass