from matplotlib import pyplot as plt
import numpy as np
from cycler import cycler
from collections import OrderedDict

from pyProfile.psd import make_psd_plots as make_plots
print ("WARNING: makenicePSDplots was moved in pyProfile.psd, it will be discontinued, modify your import.")

"""
def make_plots(toplot,outfile=None): 
    plt.figure()
    # 1. Setting prop cycle on default rc parameter
    plt.rc('lines', linewidth=1.5)
    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':', '-.'])* cycler('color', ['r', 'g', 'b', 'y']) ))
       
    for ff,ll in toplot.items():    
        f,p=np.genfromtxt(ff,delimiter='',unpack=True,usecols=[0,1])
        plt.plot(f,p,label=ll)
        plt.loglog()
        plt.xlabel('freq. (mm$^-1$)')
        plt.ylabel('mm $\mu$m$^2$')
        plt.grid(1)
        plt.legend(loc=0)
        plt.show()
        
    if outfile is not None:
        plt.savefig(outfile)
        print("results saved in %s"%outfile)
        #plt.savefig(fn_add_subfix(ff,'_psdcomp','.png'))
"""