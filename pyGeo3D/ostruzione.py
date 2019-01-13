#!/usr/bin/env python
#Boa:PyApp:main
import math
modules ={}

def diretto (F,dmin,SH):
    '''ricavando diametro all' ingersso da diam allintersezione'''
    th=math.atan2(float(dmin/2),float(F))/4
    dmax=2*SH*math.tan(th)+dmin
    return dmax

def inverso(F,dmax,H):
    '''ricava il dianetri allintersezione dal diametro d'ingresso
    con metodo ricorsico.  stessa unita' di misura.'''
    rIn=float(dmax)/2

    def ThSh(R):return math.atan2(float(R),float(F))/4
    def Rint(rIn,th0):return rIn-H*math.tan(th) #trova raggio allintersezione da angolo e rIn
    r=rIn
    rOld=0
    while (abs(r-rOld)>10e-8):
        rOld=r
        th=ThSh(r)        
        r=Rint(rIn,th)
    return r*2   

def daa(y0,angolo):
    '''dato un punto di partenza (0,y0), e l'angolo con l'asse y 
    per una retta con coefficiente angolare negativo, 
    restituisce la x nel di ordinata y1'''
    return y0*math.tan(angolo)

def angsep(F,H,r,theta):
    alfa=math.atan2(r,F)/4
    H1=H2=H
    #trova coordinata minima pupilla uscita, il raggio colpisce lintersezione p-h
    rout=r-daa(H2,alfa+theta)
    #trova prossimo raggio corrispondente
    def ThSh(R):return math.atan2(float(R),float(F))/4
    def Rint(rint,th0,H):return rint-H*math.tan(th) #trova raggio all'uscita
    rOld=0
    while (abs(rout-rOld)>10e-8):
        rOld=rout
        th=ThSh(rout)        
        rout=Rint(rout,3*th,H2)
    #r1 e' il raggio all'uscita
    rint= rout+H2*math.tan(3*th) 
    rin= rint+H1*math.tan(th)
    #calcola angsep
    angsep=math.atan2(r-rin,H1)
    return angsep

def main(F,H,r_vec,theta):
    '''assumendo shell con altezza uguale per parabola e iperbole,
    calcola la minima separazione
    angolare fra le shell per evitare ostruzione all'uscita per i
    raggi del vettore r_vec per l'angolo fuori asse theta'''
    sep_vec=[angsep(F,H,r,theta) for r in r_vec]
    print(r_vec, sep_vec)
    

if __name__ == '__main__':
    rmin,rmax=175,350
    H=300
    F=20000
    npunti=100
    r_vec=[rmin+xx*(rmax-rmin)/(npunti+1) for xx in range(npunti)]
    theta=1/(60*57.) #1 primo
    main(F,H,r_vec,theta)
