import numpy as np
#2014/12/30 modeled on the base of slumpingModel.py (first model for azimuthal slumping).
from matplotlib import pyplot as plt
import os

from plotting.plot_labels import plot_labels
from thermal.Mandrel import Mandrel
from thermal.Sample import Sample
from thermal.ThermalCycle import ThermalCycle
import thermal.Material as Material
import thermal.profileSlumping as prS

class SlumpingModelProfile(object):
    """TODO: modify init parameters in a way that simpler types can be passed, not necessarily objects.
        """
        
    def __init__(self,thermalCycle=None,mandrel=None,sample=None,time=None):
        if thermalCycle!=None:self.thermalCycle=thermalCycle
        if mandrel!=None: self.mandrel=mandrel
        if sample!=None: self.sample=sample
        if time!=None: self.time=time
        self.__changed=True
    
    def alpha_cycle(self,viscosity,E,nu):
        """Calculate slumping amount alpha at each time."""
        time=self.time
        return E/2/(1+nu)*np.cumsum(1/(10**viscosity))/(time*3600.)

    def _slump(self):
        thermalCycle=self.thermalCycle 
        temp=thermalCycle.instantTemp(self.time) 
        pslumped,ct,ci=prS.slump(self.mandrel.x,self.mandrel.y,self.sample.y)
        mat=self.sample.material
        self.alpha=self.alpha_cycle(self.time,mat.viscosity(temp),mat.Young,mat.poissonRatio)
        self.temp=temp
        self.viscosity=sample.material.viscosity(temp)
        self.pslumped,self.ct,self.ci=pslumped,ct,ci 
        self.__changed=False
        
        return pslumped,ct,ci
    
    def _time_from_TC(self,npoints=1000,set=False):
        """calculate time from thermal cycle, return it. If set is True, set self.time."""
        time=np.linspace(np.min(self.thermalCycle.partialTimes),np.max(self.thermalCycle.partialTimes),npoints)
        if set:self.time=time
        return time 
    
    def plot(self,iTimes,outfolder=None,delay=0.05):
        x=self.mandrel[0]
        ym=self.mandrel[1]
        prS.plot_steps(self.pslumped,self.ct,self.ci,iTimes,x,ym=ym,outfolder=outfolder,delay=delay)
    
    def __getattribute__(self,attr):
        #see notes in slumpingModel why it is done like this.
        if attr in ('ct','ci','pslumped','viscosity','temp','alpha'):
            self._slump()

        return super(SlumpingModelProfile, self).__getattribute__(attr)
    

if __name__=='__main__':
    R=220  # Enter mandrel radius of curvature (in mm)'
    W=150  # Enter substrate width (in mm)'
    Ntime=1000 #number of time intervals for the calculatio

    programdir=os.path.realpath(os.path.dirname(__file__))
    os.chdir(programdir)
    theta=0.0
    
    #define Mandrel, sample, thermal Cycle, time.
    xm,ym,xg,yg=prS.load_test_profiles()
    OP1=Mandrel(R)
    OP1.x=xm
    OP1.y=ym
    sample=Sample(width=1.,thickness=0.4,material=Material.eagle)
    sample.x=xg
    sample.y=yg
    thermalCycle=ThermalCycle([0,12,12,24,12,12],[150,650,730,730,650,150])
    time=np.linspace(0,thermalCycle.totalTime,Ntime)
    
    m=SlumpingModelProfile(thermalCycle,OP1,sample=sample,time=time)
    print(m.alpha)

    
    
