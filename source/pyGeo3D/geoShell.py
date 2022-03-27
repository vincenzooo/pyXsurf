from math import *

class fig(object):
    def __init__ (self,origin):
        self.origin=tuple(map(float,origin))
        object.__init__(self)
    def __repr__(self):
        #s="\n--<"+str(self.__class__) + " instance at " + hex(id(self)) + ">--\n"
        s=object.__repr__(self)+"\n"
        for k in list(vars(self).items()): s=s+ "%s = %s\n" %k
        s=s+"--------------------------\n\n"
        return s
    def __eq__(self,other):
        print("metodo di confronto non implementato")
        return None

class punto(fig):
    def __init__(self,x,y,z,origin=(0,0,0)):
        self.x=float(x)
        self.y=float(y)
        self.z=float(z)
        fig.__init__(self,origin)
    def trasla (self,new_origin):
        #print "translazione non ancora implementata"
        n_o=(self.origin[0]-new_origin[0],self.origin[1]-new_origin[1],self.origin[2]-new_origin[2])
        p=punto(self.x+n_o[0],self.y+n_o[1],self.z+n_o[2],new_origin)
        return p
    
class raggio(fig):
    ''' un raggio di luce e' una retta nel piano // a yz (x=origin[0])
    equazione fissata da due parametri m e q. m,q= None per retta verticale
    (// asse z, equazione x=origin[0], y=origin[1])'''
    def __init__(self,m,q,origin=(0,0,0)):
        if m!= None:
            self.m=float(m)
        else:
            self.m=None
        self.q=float(q)
        fig.__init__(self,origin)
    def trasla (self,new_origin):
        n_o=(new_origin[0]-self.origin[0],new_origin[1]-self.origin[1],new_origin[2]-self.origin[2])
        if self.m != None:
            new_q=self.q+self.m*n_o[1]-n_o[2]
            r=raggio(self.m,new_q,new_origin)
        else:
            r=raggio(None,self.q-n_o[1],new_origin)
        return r
    def getTh(self):
        return math.atan(1/self.m)
    def setTh(self, value):
        if value==math.pi/2:
            self.m=0
        elif value==0:
            self.m=None
        else:
            self.m = 1/math.tan(value)
    theta = property(getTh, setTh, "theta angle with z axis")

class proPlane(fig):
    ''' piano // a asse x con inclinazione rispetto asse z,
    uguale all'angolo off-axis. e' il piano su cui si proietta
    l'apertura dell'ottica. equazione fissata da due parametri m e q.
    '''
    def __init__(self,m,q,origin=(0,0,0)):
        self.m=float(m)
        self.q=float(q)
        fig.__init__(self,origin)
    def projectCircle(self,R,center):
        #da' R valore del raggio e center punto di centro
        center=center.trasla(self.origin)
        #retta perpendicolare al piano passante per il centro
        yc=center.y
        zc=center.z
        m=self.m
        q=self.q
        #r=raggio(-1/m,zc+yc/m,self.origin)
        #coordiate dell'intersezione col piano
        ncy=(zc+yc/m-q)/(m+(1/m))
        ncz=ncy*m+q
        new_cen=punto(center.x,ncy,ncz,self.origin)
        pass
        return new_cen
        
    def getTh(self):
        return math.atan(1/self.m)
    def setTh(self, value):
        if value==math.pi/2:
            self.m=0
        else:
            self.m = 1/math.tan(value)
    theta = property(getTh, setTh, "theta angle with z axis")
    
class hyp(fig) :
    '''iperbole determinata (nel piano) dai due parametri a e b,
    secondo l'equazione (z/a)^2-(y/b)^2=1 '''
    def __init__(self,a,b,origin=(0,0,0)):
        self.a=float(a)
        self.b=float(b)
        fig.__init__(self,origin)
    def rayIntersect(self,raggio):
        '''restituisce uno o piu' punti intersezione tra iperbole
        e raggio'''
        #parametri dell'equazione quadratica associata al sistema
        if self.origin[0] != raggio.origin[0]:print("attenzione! self.origin[0] != raggio.origin[0]:\n",\
            self.origin[0],raggio.origin[0])
        raggio=raggio.trasla(self.origin)
        if raggio.m != None:    
            a = (raggio.m/self.a)**2-(1/self.b)**2
            b = 2*raggio.m*raggio.q/self.a**2
            c = -1 + (raggio.q/self.a)**2
            #calcola delta
            delta = b**2-4*a*c
            if delta >= 0:
                #risolve formula
                solY=((-b-math.sqrt(delta))/(2*a),(-b+math.sqrt(delta))/(2*a))
                solZ=[math.sqrt((1+(y/self.b)**2)*self.a**2) for y in solY]
                p1=punto(0,solY[0],solZ[0],self.origin)
                p2=punto(0,solY[1],solZ[1],self.origin)
                return p1,p2
            else:
                return None
        else:
            solZ=math.sqrt((1+(raggio.q/self.b)**2)*self.a**2)
            return punto(0,raggio.q,solZ,self.origin)
        
    def tanInPmq(self,p):
        '''restituisce m e q del raggio tangente all'iperbole nel punto p
        nelle coordinate con origine self.origin'''
        p=p.trasla(self.origin)
        b=self.b
        a=self.a
        y0=p.y
        z0=p.z
        #controlla appartenenza
        if not(((y0/a)**2-(z0/b)**2)==1):
            print("punto non appartenente all'iperbole")
            #return None
        #calcola con formula
        q=(b**2)/z0
        m=(q*y0)/a**2
        return m,q
        
    def tanInP (self,p):
        '''restituisce il raggio tangente all'iperbole nel punto p'''
        '''simile a tanInP, restituisce l'angolo tra tangente e
        raggio incidente nel punto p '''
        m,q=tanInPmq(self,p)
        return raggio(m,q,self.origin)

    def tanInPang (self,p):
        '''restituisce l'angolo tra tangente e
        raggio incidente nel punto p '''
        m,q=tanInPmq(self,p)
        return math.atan(1/m)

class shell1(fig):
    ''' shell determinata da due parametri tra raggio, focale, angolo
    non considero l'altezza in quanto la prendo come unita' di misura,
    quindi H=1. considero solo la prima semisuperficie '''
    def __init__(self,R,F,origin=(0,0,0),limit=-1):
        self.F=float(F)
        self.alfa=atan2(R/F)/4
        fig.__init__(self,origin)
        if limit==0:        #non considero limiti: cono infinito
            self.limite=0 
        elif limit == -1: #limiti predefiniti 0 e 1
            self.limite=limite()
        else:
            self.limite=limit
    def planeIntersect (self,x0):
        '''intersezione con un piano // a yz, con eq. x=x0
        (rispetto all'origine standard)'''
        x0=x0-self.origin[0]
        h=hyp((x0/tan(self.alfa)),(xo),(0,0,-self.R/tan(alfa)),self.origin)
        return h
    def rayIntersect (self,raggio):
        ''' da' le coordinate del punto di intersezione con un raggio'''
        h=planeIntersect(raggio.origin[0])
        return h.rayIntersect(raggio)

class limite (fig):
    def __init__ (self,zmin=0.,zmax=1.,origin=(0,0,0)):
        '''limite della shell, un po' da definire'''
        self.zmin=float(zmin)
        self.zmax=float(zmax)
        fig.__init__(self,origin)



if __name__ == "__main__":
    pass

    o=punto(3,-2,1)
    pl=proPlane(-2,4)
    print(pl.projectCircle(5,o))

    '''
    #test rette verticali
    h=hyp(2,3)
    r=raggio(0,0)
    r.theta=0
    print h.rayIntersect(r)
    r.q=1
    print h.rayIntersect(r)'''
    
    ''' #test intersezione hyp-retta
    h=hyp(2.,math.sqrt(12.))
    r=raggio(1./3,2.)
    l=limite(1,3)
    p=list( h.rayIntersect(r))
    print p
    psel=[i for i in p if (l.zmin <= i.z <= l.zmax)]
    print psel    
    '''
    
    '''#test traslazione di un punto
    r=raggio(3,2)
    print "prima di traslazione r=raggio(3,2):"
    print r
    print "dopo traslazione r.trasla((0,0,2)):"
    print r.trasla((0,0,2))
    '''

    ''' #test tangente in un punto    
    h=hyp(2,3)
    p=punto(0,3.,math.sqrt(5)*1.5)
    print h.tanInP(p)
    '''


    