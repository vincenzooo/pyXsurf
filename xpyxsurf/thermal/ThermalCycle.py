import numpy as np
from matplotlib import pyplot as plt

class ThermalCycle(object):
    """A thermal cycle initialized with two sequences: setpoint times and temperatures."""
    def __init__(self,spTime,spTemp):
        self.spTime=np.array(spTime,dtype=float)
        self.spTemp=np.array(spTemp,dtype=float)
    
    @property
    def totalTime(self):
        """Total time of the thermal cycle."""
        return np.sum(self.spTime)

    @property
    def partialTimes(self):
        """Total time from start for each set point of the thermal cycle."""
        return np.cumsum(self.spTime)
        
    def instantTemp(self,times=None):
        """"return the instantaneous temperature for the thermal cycle, as a function of
            htime (in hours)."""
        times=np.linspace(self.spTime[0],self.totalTime,100) if times==None else times
        return np.interp(times,self.partialTimes,self.spTemp)

    def plot(self,times=None,label=None,overplot=False,fileName=None):
        if times==None:times=self.spTime.cumsum()
        fig=plt.gcf()
        if not(overplot):
            plt.xlabel('Time(h)')
            plt.ylabel('Temperature (C)')
            plt.grid()

        plt.plot(times,self.instantTemp(times),label=label)
        plt.plot(self.spTime.cumsum(),self.spTemp,'o')
        plt.show()
        if fileName != None: plt.savefig(filename)
        return fig.gca()
        
    def __repr__(self):
        s=[super(self.__class__,self).__repr__()]
        s.append('Set point time: %s'%self.spTime)
        s.append('Elapsed time: %s'%self.partialTimes)
        s.append('Set point temperature: %s'%self.spTemp)
        s.append('Total time: %s'%self.totalTime)
        return '\n'.join(s)
        
        
if __name__=="__main__":
    Ntime=1000
    tc=ThermalCycle([0,12,12,24,12,12],[150,650,730,730,650,150])
    time,delta_t=np.linspace(0,tc.totalTime,Ntime,retstep=True)
    tc.plot(time)
    tc2=tc=ThermalCycle([0,12,12,24,12,12],[150,650,680,680,650,150])
    tc2.plot(time[::10],overplot=True)
    tc3=ThermalCycle([0,12,6,24,12,12],[150,650,730,730,650,150])
    tc3.plot(time[::10],overplot=True)
    