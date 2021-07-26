import scipy 
import pylab
from scipy import optimize as opt
from pylab import legend
log=scipy.log

def VFT(p,T):
    '''From a 3 vector of VFT parameters (p) calculate viscosity for
    a vector of temperatures (T)'''
    loge=scipy.log10(scipy.e)
    return p[0]+loge*(p[1])/(T-p[2])

fitfunc=VFT
    
def fitVFT(T,logNu):
    '''Fit two vectors of temperature and log(viscosity) using
    the Vogel-Fulcher-Tamman equation. Initial guess is calculated from the
    first three parameters.'''
    
    # define a gaussian fitting function where
    # p[0] = log(nu0) =l0
    # p[1] = T0
    # p[2] = Tinf

    #Vogel-Fulcher-Tamman equation, nu(T)=nu0*exp((T0)/(T-Tinf))
    #fitfunc = lambda p, x: scipy.log10(p[0]*scipy.exp((p[1])/(x-p[2])))
    #fitfunc = lambda p, x: p[0]+loge*(p[1])/(x-p[2])
    errfunc = lambda p, x, y: fitfunc(p,x)-y
    
    # guess some fit parameters
    a1,b1=[(T[0]*logNu[0]-T[1]*logNu[1])/(logNu[0]-logNu[1]),(T[0]-T[1])/(logNu[0]-logNu[1])]
    a2,b2=[(T[2]*logNu[2]-T[1]*logNu[1])/(logNu[2]-logNu[1]),(T[2]-T[1])/(logNu[2]-logNu[1])]
    l0=(a2-a1)/(b1-b2)
    Tinf=a1+b1*l0
    T0=(T[1]-Tinf)*scipy.log(10**logNu[1]/l0)
    p0=[l0,T0,Tinf]
    #p0 = [3e-3,10000.,256.]
    #p0 = [3e-3,9000.,100.] #it works for D263
    p1, success = opt.leastsq(errfunc, p0,args=(T,logNu))
    return p1,success

def plot_curves():
    """plot curves for eagle and d263, Save figure and print results."""
    print(pE,success)
    print(fitfunc(pE,TEagle))
    pylab.title('')
    pylab.clf()
    pylab.xlabel('Temperature ($^{\circ}$C)')
    pylab.ylabel(r'$\log{(\eta)} Pa s$')
    pylab.plot(TEagle,logNu,'o',label='Eagle Data')
    pylab.plot(TD263,logNu,'*',label='D263 Data')
    #define a vector of 100 points for x (temperature)
    xT=scipy.linspace(min((min(TD263),min(TEagle))),max((max(TD263),max(TEagle))),100)
    pylab.plot(xT,fitfunc(pE,xT),label='Eagle VFT fit')
    pylab.plot(xT,fitfunc(pD,xT),label='D263 VFT fit')
    pylab.axhline(10,label=r'$\log{(\eta)}=10.0$',ls='--')
    pylab.axvline(570,label=r'$D263 slumping Temp$',ls=':')

    pylab.grid()
    pylab.legend()
    pylab.show()
    pylab.savefig('VFT.jpg')
    #pylab.xlim([720,800])
    #pylab.ylim([11,13])

    #pylab.savefig('VFT_zoom.jpg')

if __name__=='__main__':
    TEagle=scipy.array([669,722,971,1293]) #Eagle
    TD263=scipy.array([529,557,736,1051]) #d263
    logNu=scipy.array([13.5,12,6.6,3])
          
    pE,success=fitVFT(TEagle,logNu)
    pD,success=fitVFT(TD263,logNu)
    plot_curves()


