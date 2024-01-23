# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 00:36:57 2018

@author: Vincenzo
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image

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
        
def change_labels(newlab):
    """Change labels of an existing legend modifying line labels."""
    
    for ll,c in zip(plt.gca().lines,newlab):
        ll.set_label(c)
    return plt.legend()
        
def legendbox(text,ax=None,loc=0,clear = False, *args,**kwargs):
    """deprecated: Create a legend from class matplotlib.legend.Legend and add it to axis.
        `text` is a list of strings to be passed to `labels` parameter in Legend.
        Unclear how line styles are set (probably none).
    if clear is set to True, remove previous legends.
        """
    '''
    N.B.: this is not useful any more. Not sure if it was added or just I didn't know about it, but the same can be obtained with plt.legend() (adding the old legend to axis, artists), or alternatively this function can be readapted in that sense.

        The original function was written to add legends without lines (e.g. stats boxes in surface plots)
        without erasing previous legends (only one legend could be built per axis).
        
        It might also be useful to use this function to modify label text but not lines labels (not sure if it is of any use).
        https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
    '''
    
    from matplotlib.patches import Patch
    from matplotlib.legend import Legend
    #------ SAME BUT WITH CUSTOM LEGEND (DOESN'T ERASE OTHER LEGENDS)

    #plt.plot([1,2],[2,3],label='cane')
    #plt.legend(loc=1) 
    
    #pdb.set_trace()
    if np.size(text) == 1 and not isinstance(text, list): text = [text]
    if ax is None:
        ax=plt.gca()
    if clear:
        ax.get_legend().remove()
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
    l = legendbox(get_stats(data,units=['mm','mm','um']))
    plt.show()
    return l
    
"""From imageCaption (A module that creates a caption as image):"""

def autofont(novo,csize,minfont=10):
    """Given a text novo (as list of lines), determine the font size that fits an image of size csize."""
    linefraction=0.1 #vertical margin for lines as a fraction of line height
    
    fontsize=minfont
    font = ImageFont.truetype("arial.ttf", fontsize)
    longestline=novo[np.argmax([len(n) for n in novo])]
    while font.getsize(longestline)[0] < csize[0]:
        # iterate until the text size is just smaller than the criteria
        fontsize += 1
        font = ImageFont.truetype("arial.ttf", fontsize)
    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1 
    font = ImageFont.truetype("arial.ttf", fontsize)       
    
    #decrease font size if it doesn't fit 
    while font.getsize(longestline)[1]*len(novo)*(1+linefraction) > csize[1]:
        # iterate until the text size is just larger than the criteria
        fontsize -= 1
        font = ImageFont.truetype("arial.ttf", fontsize)
    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1 
    font = ImageFont.truetype("arial.ttf", fontsize)
    return font
    
def render_caption(novo,csize,minfont=10):
    """return a rendered image of the text novo (as list of lines) determining the font size that fits the box. csize is the size of the returned image (x,y) in pixels."""

    offset = 10  #margin in pixels between linesheight   
    font=autofont(novo,csize,minfont)
    capim=Image.new("RGB",csize)
    d = ImageDraw.Draw(capim)
    #d = ImageDraw.Draw(newim)
    for line in novo:
        d.text((0,offset),line,(255,255,255),font=font)
        offset += font.getsize(line)[1]
    del d
    
    return capim    
    
if __name__ == "__main__":
    l = test_legendbox()