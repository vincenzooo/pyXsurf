import numpy as np
from matplotlib import pyplot as plt
from pint import UnitRegistry
import math
"""
This is the frequency version of blackbody enery density.
Use pint for units.
"""


#import quantities as pq #to use quantities
pq=UnitRegistry() #to use pint

h = pq.planck_constant # 6.626e-34*pq.J*pq.s
c = pq.c    #3.0e+8*pq.m/pq.s
k = pq.k     #1.38e-23*pq.J/pq.K

"""
def planck(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity
"""

def planck(nu, T,wl=False):
    if wl: nu=c/nu #accept wavelengths as input.
    return 8*math.pi*h*nu**3/c**2/((np.exp((h*nu/k/T).to_base_units()))-1)

def rj(nu,T,wl=False):
    if wl: nu=c/nu #accept wavelengths as input.
    return 8*math.pi*nu**2/c**2*k*T
    
def wien(nu,T,wl=False):
    if wl: nu=c/nu #accept wavelengths as input.
    return 8*math.pi*h*nu**3/c**2*(np.exp((-h*nu/k/T).to_base_units()))


def bbapprox(wl,T):
    """makes a plot of black-body power distribution for wavelengths (Planck)
        and its approximation (Rayleigh-Jeans and Wien)."""
    plt.figure(1)
    plt.clf()
    pl=planck(wl,T,1).to_base_units()
    plt.plot(wl,pl,'r',label='Planck' )
    plt.loglog(1)
    plt.ylim(plt.ylim())
    plt.grid(1)
    plt.plot(wl,rj(wl,T,1).to_base_units(),'g-.',label='RJ')
    plt.plot(wl,wien(wl,T,1).to_base_units(),'b--',label='Wien')
    plt.xlabel('Wavelength(%s)'%(wl.units))
    plt.ylabel(('Radiance(%s)'%(pl.units)))
    plt.legend(loc=0)
    plt.title('Black body spectrum and approx. (T=%s)'%T)
    plt.show()
    plt.savefig('blackbody.png')



if __name__=='__main__':
    wl=np.arange(100.,10000000,100)*pq.nm
    T=1000.*pq.K
    bbapprox(wl,T)