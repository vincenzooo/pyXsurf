import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt

"""2014/11/17 moved in libraries. From now on, import it with:
from thermal.vft import VFT.
Moved Material class to its own file. Moved fitVFT to here."""

#2014/05/15 note: VFT(p,T) returns log10 of viscosity, material.viscosity returns viscosity 10**VFT(self.VFTpars,T) 
#   added flag log to return log10
#2014/04/07 first implementation with classes

def VFT(p,T):
    '''From a 3 vector of VFT parameters (p) calculate viscosity for
    a vector of temperatures (T), return logarithm base 10 of viscosity.         
    Vogel-Fulcher-Tamman equation, nu(T)=nu0*exp((T0)/(T-Tinf))'''
    loge=np.log10(np.e)
    return p[0]+loge*(p[1])/(T-p[2])

def fitVFT(T,logEta):
    '''Fit two vectors of temperature and log(viscosity) using
    the Vogel-Fulcher-Tamman equation. Initial guess is calculated from the
    first three parameters. If success, return VFTpars, if failure return None.'''
    
    # define a fitting function where
    # p[0] = log10(nu0) 
    # p[1] = T0
    # p[2] = Tinf

    #print 'fit VFT'
    
    #default fitfunc(p,T) with p VFT parameters and T temperature
    #errfunc = lambda p, x, y: VFT(p,x)-y    
    #errfunc(p,x,y) is the difference between y and the VFT function for p
    def errfunc(pars,temp,eta): return VFT(pars,temp)- eta
    
    # guess some fit parameters
    a1,b1=[(T[0]*logEta[0]-T[1]*logEta[1])/(logEta[0]-logEta[1]),(T[0]-T[1])/(logEta[0]-logEta[1])]
    a2,b2=[(T[2]*logEta[2]-T[1]*logEta[1])/(logEta[2]-logEta[1]),(T[2]-T[1])/(logEta[2]-logEta[1])]
    l0=(a2-a1)/(b1-b2)
    Tinf=a1+b1*l0
    T0=(T[1]-Tinf)*np.log(10**logEta[1]/l0)
    p0=[l0,T0,Tinf]
    #p0 = [3e-3,10000.,256.]
    #p0 = [3e-3,9000.,100.] #it works for D263
    p1, success = opt.leastsq(errfunc, p0,args=(T,logEta)) 
    if success: 
        return p1 
    else:
        return None

if __name__=='__main__':

    # temperature in C, viscosity in Pa s
    TEagle=np.array([669,722,971,1293]) #Eagle
    #TD263=np.array([529,557,736,1051]) #d263
    logEta=np.array([13.5,12,6.6,3])
    
    pars=fitVFT(TEagle,logEta)
    
    xT=np.linspace(500,1300,100)
    plt.clf()
    plt.plot(xT,VFT(pars,xT))
    plt.xlabel('Temperature ($^{\circ}$C)')
    plt.ylabel(r'$\log{(\eta)}$ (Pa s)')
    
    plt.axhline(10,label=r'$\log{(\eta)}=10.0$ Pa s',ls='--')
    plt.legend()
    plt.show()
    plt.savefig('VFT.jpg')


