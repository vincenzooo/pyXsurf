'''ora voglio provare con un vero array'''
    
class subArray:
    ''''voglio provare a ottenere un effetto simile senza ereditarieta'.
    per gli array infatti non si puo' ereditare'''
    def __init__(self,data=[]):
        self.data=numpy.array(data)
        
    def __getattr__(self,attr):
        #print  "\nin __getattr__: attr= ",attr
        #print "print self.data:",self.data
        b= getattr(self.data,attr)
        #print "b= getattr(self.data,attr)"
        #print "print b,self.data:",b,self.data
        #print "return b:"
        return b

    def prova(self):
        print "scrivo 'prova'!!"

if __name__=="__main__":
    print "\n-------------------------------------"*2
    print "sA=subArray[1,-2,3]) [non subclassa]"
    print "---------------------------------------"
    print "---------------------------------------"
    sA=subArray([1,-2,3])
    print "---- sA, non subclassato --"
    print "- sA.data: ",sA.data
    print "\n- sA.sort: ",sA.sort
    print "\n- sA.data: ",sA.data
    print "\n- sA.sort(): ",sA.sort()
    print "\n- sA.data: ",sA.data
    try:
        print "\n- sA.kov (proprieta' inesistente):",sA.kov
    except:
        print "riscontra errore"
    print "\n- sA.prova(): ",sA.prova()