# 2023/04/11 This module should contain final library functions, the old or incomplete ones are put in repeatability_dev.
#   Unclear why this is needed at all, as useful functions are already included in dlist.
# 
# 2018/11/02 moved to scripts.repeatabilty from WFS_repeatability in POOL\PSD\WFS_repeatability
#v4 2018/09/18 include functions from reproducibility plot. Includes transformations
#  and differences for any number of plots (it was 3).
#v3 added functions all functions from repeatability.
#v1 works, this v2 adds finer control of styles in make_styles with list value:symbol.


import numpy as np
import matplotlib.pyplot as plt
import os
from dataIO.fn_add_subfix import fn_add_subfix



#from pySurf.data2D_class import align_interactive

#from config.interface import conf_from_json


"""2019/04/08 from messy script/repeatability, here sorted function that have been used. 
When a function is needed by e.g. a notebook, move it here from repeatability_dev.
"""
    
def psd_variability(psdlist,psdrange=None,rmsrange=None,fignum=None,units=None,
    outname=None,figsize=(15,5)):
    # to be removed
    """given a list of 2d psd files plot spans of all of them on same plot.
    TODO: replace input using data rather than files. Move to 2DPSD"""

    from plotting.fignumber import fignumber   
    from pySurf.data2D import projection
    from pySurf.psd2d import rms_power 
    from scipy.stats import ttest_ind
    
    units=['mm','mm','$\mu$m']
    labels = [p.name for p in psdlist]

    #gestione rudimentale dell'output troppo stanco per pensare allo standard.
    if outname is not None:
        label=os.path.basename(outname)
        outfolder=os.path.dirname(outname)
        os.makedirs(outfolder,exist_ok=True)

    rms_v=[]
    fig=fignumber(fignum,figsize=figsize)
    plt.subplot(121)
    for (p,x,f),fn in zip(psdlist,labels):
        ps=projection(p,axis=1,span=True)
        rms_v.append(rms_power(f,p,rmsrange))
        c=plt.plot(f,ps[0],label=os.path.basename(fn)+' AVG')[0].get_color()
        plt.plot(f,ps[1],'--',label='point-wise min',c=c)
        plt.plot(f,ps[2],'--',label='point-wise max',c=c)
        #pdb.set_trace()
    plt.ylabel('axial PSD ('+units[2]+'$^2$ '+units[1]+')')
    plt.xlabel('Freq. ('+units[1]+'$^{-1}$)')
    plt.vlines(rmsrange,*plt.ylim(),label='rms range',linestyle=':')
    plt.ylim(psdrange)
    plt.loglog()
    plt.grid(1)
    plt.legend(loc=0)

    print("Statistics for %s, rms range [%6.3g,%6.3g]:"%(label,*rmsrange))
    for rms,fn in zip(rms_v,labels):
        print(os.path.basename(fn)+": rms=%6.3g +/- %6.3g (evaluated on %i profiles)"%
            (np.nanmean(rms),np.nanstd(rms),len(rms)))
    print ("--------------")
    print ("Distribution of rms: %6.3g +/- %6.3g"%(np.nanmean(rms_v,axis=1).mean(),np.nanmean(rms_v,axis=1).std()))
    print ("AGGREGATED: rms=%6.3g +/- %6.3g (evaluated on %i profiles)"%
            (np.nanmean(rms_v),np.nanstd(rms_v),np.size(rms_v)))
    if len(rms_v)==2:
        print ("Student T test t:%6.3g p:%6.3g"%ttest_ind(rms_v[0],rms_v[1]))

    l=[os.path.splitext(os.path.basename(f))[0]+" rms=%3.2g +/- %3.2g (N=%i)"%
        (np.nanmean(r),np.nanstd(r),len(r)) for f,r in zip(labels,rms_v)]
    plt.subplot(122)
    plt.hist([rr[np.isfinite(rr)] for rr in rms_v],density=True,bins=100,label=l)
    plt.legend(loc=0)
    plt.title(label+" distrib. of avg: %3.2g +/- %3.2g"%
        (np.nanmean(rms_v,axis=1).mean(),np.nanmean(rms_v,axis=1).std()))
    #handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
    #labels= ["low","medium", "high"]
    #plt.legend(handles, labels)
    plt.tight_layout()
    if outname is not None:
        plt.savefig(fn_add_subfix(outname,'_stats','.jpg'))

    return rms_v
