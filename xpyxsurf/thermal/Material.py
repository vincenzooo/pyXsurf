import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt
from thermal.vft import VFT,fitVFT

class Material(object):
    """Material with thermal properties.
        eagle=Material(TEta=(TEagle,logEta),name='EagleXG')
        eagle.viscosity(T=293.75,log=False)
        eagle.plot_viscosity(T=293.75,log=False)
        """
    name=""
    def __init__(self,**kwargs):
        #quick prototype, replace with actual arguments.
        for k, v in list(kwargs.items()):
            setattr(self, k, v)
    
    def viscosity(self,T=293.75,log=False):
        """ return the viscosity of the material for a given vector of temperatures.
        If log is set to true, the logarithm base 10 is returned instead of the linear value.
        This is useful to avoid overflow at low temperatures (max exp for float is 308)."""
        self.VFTpars=fitVFT(self.TEta[0],self.TEta[1])
        if log: 
            return VFT(self.VFTpars,T) 
        else: 
            return 10**VFT(self.VFTpars,T)

    def plot_viscosity(self,T,overplot=False,noMarkers=False):
        if not overplot:
            plt.clf()
            plt.title('')
            plt.xlabel('Temperature ($^{\circ}$C)')
            plt.ylabel(r'$\log{(\eta)}$ (Pa s)')
        
        self.VFTpars=fitVFT(self.TEta[0],self.TEta[1])
        if not noMarkers: plt.plot(self.TEta[0],self.TEta[1],'o',label=self.name+' Data')
        plt.plot(T,self.viscosity(T,log=True),label='VFT fit')
        if not overplot: plt.grid()
        plt.legend()
        plt.show()
        return plt.gca()

"""define common materials here, they can be used e.g. as Material.eagle """

TEagle=np.array([669,722,971,1293]) #Eagle
TD263=np.array([529,557,736,1051]) #d263
logEta=np.array([13.5,12,6.6,3])

eagle=Material(TEta=(TEagle,logEta),name='EagleXG')
eagle.Young=73.6e9   #Eagle glass 73.6 GPa -> g/mm/s**2
eagle.poissonRatio=0.23
eagle.density=2.38e-3 #g/cm^3 -> g/mm^3

D263=Material(TEta=(TD263,logEta),name='D263')
        
if __name__=='__main__':
    # temperature in C, viscosity in Pa s
    
    xT=np.linspace(min((min(TD263),min(TEagle))),max((max(TD263),max(TEagle))),100)
    plt.clf()
    eagle.plot_viscosity(xT)
    D263.plot_viscosity(xT,overplot=True)
    
    plt.axhline(10,label=r'$\log{(\eta)}=10.0$ Pa s',ls='--')
    plt.axvline(570,label=r'D263 slumping Temp',ls=':')
    plt.legend()
    plt.show()
    plt.savefig('VFT_materials.jpg')
