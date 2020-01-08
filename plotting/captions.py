# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 00:36:57 2018

@author: Vincenzo
"""

import matplotlib.pyplot as plt
import numpy as np

def original_make_caption():
    if mode=='tagcode': #single string (e.g. for filenames)
        #teh string manipulation is to avoid leading and ending underscores if head or tail are not set
        novo = ((' '.join([head,exp,foc,"".join(re.findall('\d+',fn)),tail])).strip()).replace(" ","_")
        #novo = '_'.join([head,exp,foc,"".join(re.findall('\d+',fn)),tail]) #the re gives all digits in filename, I believe.
    elif mode=='plottitle':
        novo="%s Exp:%s,f/n:%s %s"%(fn,exp,foc,dt)
    elif mode=='multiline': #caption as multiline text
        #texto is a single string with spaces between words to be splitted in lines
        texto=" ".join([head,fn,"exp:%s f#:%s (EV%1.1f) %i X %i %s"%(exp,foc,calculate_EV(self.foc,self.exp,self.iso),abs(int(yl[1]-yl[0])),abs(int(xl[1]-xl[0])),dt),"offset:%s,%s (or. size:%s,%s)"%(int(xl[0]),int(yl[0]),xsize,ysize),tail])
        novo = textwrap.wrap(texto, width=textwidth)
    else:
        raise ValueError('Mode not recognized %s'%mode)

    return novo   
        
        
def legendbox(text,ax=None,loc=0,*args,**kwargs):
    from matplotlib.patches import Patch
    from matplotlib.legend import Legend
    #------ SAME BUT WITH CUSTOM LEGEND (DOESN'T ERASE OTHER LEGENDS)

    #plt.plot([1,2],[2,3],label='cane')
    #plt.legend(loc=1) 
    import pdb 
    #pdb.set_trace()
    if np.size(text) == 1 and not isinstance(text, list): text = [text]
    if ax is None:
        ax=plt.gca()
    p = [Patch(label=ll,visible=False) for ll in text]
    leg = Legend(ax,p,text,loc=loc,handletextpad=0, handlelength=0,
                 *args,**kwargs)
    ax.add_artist(leg)

    return leg
    
def test_legendbox():
    from pySurf.testSurfaces import make_range
    from pySurf.data2D import plot_data, get_stats
    
    plt.clf()
    nx=4
    ny=3
    data,x,y=make_range(nx,ny)
    plot_data(data,x,y)
    legendbox(get_stats(data,units=['mm','mm','um']))
    plt.show()
    
    