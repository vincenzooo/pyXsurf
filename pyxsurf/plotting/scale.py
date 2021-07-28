# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from dataIO.span import span

class Scale(object):
    '''
    2018/12/10 N.B.: this almost works, but scale is not accurate when zooming.
    probably something with how the length is rounded. use grid for comparison to 
    see the effect.
    
    A scale that can be plotted as line and text.
    Can be turned on and off setting the `visible` property. 
    .pixsize is the size of a pixel in the units used for the caption.
    The string from the unit is in self.un. 
    The method draw draws the scale if self.visible is True and 
    remove it if not. Instances of line and text are returned 
    (and can be modified, remember to call plt.draw to apply changes).
    Note that scale is independent on image, so it is plotted on current
    axis, it is care of the caller to set that.
    The scale updates itself on the plot on zooming, through callback
    connection through ax.callbacks.connect
    
    Vincenzo Cotroneo vcotroneo@cfa.harvard.edu 2016/03/07
    '''
    
    
    def __init__(self,pixsize=None,un="",visible=False):
        self._scaleFraction=0.25 #fraction of image used by scale 
        self._scaleRounding=1.1 #factor for rounding of scale, typically 1.1 for 10% works 
        self._scaleMargin= 0.05 #fraction of image to keep as distance from border in drawing the scale
        self.visible=visible
        self.pixsize=pixsize
        self.un=un
        self.l=None   #matplotlib line object
        self.t=None   #matplotlib text object
        ax=plt.gca()
        ax.callbacks.connect('xlim_changed', self.draw)
        if self.visible: self.draw()
   
    def draw(self,ax=None, pixsize=None):
        """from size of pixel pixsize (default to self.pixsize) 
        and corresponding real size imsize calculate a a suitable size for a 
        scale mark and the corresponding value and draw it. 
        The text for the unit is provided in un.
        Note that only matplotlib objects are used, an image is not necessary 
        and there are no pixels: pixsize represents the physical size of a unit of 
        axis coordinate (it matches pixel size if an image is plotted with pixel 
        coordinates, as default in imshow)  """
        
        if ax is None:
            ax=plt.gca()
        
        scaleFraction=self._scaleFraction #fraction of image used by scale 
        scaleRounding=self._scaleRounding #factor for rounding of scale, typically 1.1 for 10% works
        scaleMargin=  self._scaleMargin  #fraction of image to keep as distance from border in drawing the scale
        
        #pixsize is set to the correct value:
        if not (pixsize is None): 
            self.pixsize=pixsize
        if self.pixsize is None:
            pixsize=1.
        else: pixsize=self.pixsize
        imsize=(span(ax.get_xlim(),size=1),span(ax.get_ylim(),size=1))
        scalesizeunits=int(round(scaleRounding*imsize[0]*scaleFraction*pixsize)) #size of the scale in units of un
        scalesizepx=scalesizeunits/pixsize   #size of the scale in pixels
        #determine position on plot
        ##top right 

        #xpos=plt.xlim()[0]+imsize[0]*(1-scaleMargin)-scalesizepx
        #ypos=imsize[1]*scaleMargin
        scaleFraction=scalesizepx/imsize[0] #update to correct the rounding
        xpos=1-scaleFraction-scaleMargin
        ypos=1-scaleMargin
        scaletxt="%5.3f %s"%(scalesizeunits,self.un)
        
        if not(self.l is None):
            lcol=self.l.get_color()  #ugly solution to keep color (but not other props)
            self.l.remove()
            self.l=None
        else:
            lcol='yellow'
        if not(self.t is None):
            tcol=self.t.get_color()  #ugly solution to keep color (but not other props)
            self.t.remove()
            self.t=None
        else:
            tcol='yellow'
        if self.visible:
            aux=ax.get_autoscalex_on()
            auy=ax.get_autoscaley_on()
            ax.set_autoscale_on(False)
            #self.l=plt.plot([xpos,xpos+scalesizepx],[ypos,ypos],'y',lw=3)[0]
            #self.t=plt.text(xpos,ypos+imsize[0]*scaleMargin,scaletxt,color='yellow')  
            self.l=ax.plot([xpos,xpos+scaleFraction],[ypos,ypos],'y',lw=3,transform=ax.transAxes,color=tcol)[0]
            self.t=ax.text(xpos,(ypos-scaleMargin),scaletxt,transform=ax.transAxes,color=lcol)
            aux=ax.set_autoscalex_on(aux)
            auy=ax.set_autoscaley_on(auy)
            
        plt.draw()

    def remove(self):
        v=self.visible
        self.visible=False
        self.draw()
        self.visible=v
    
        return self.l,self.t


def test_with_image(infile,outfolder=None):
    """apply a scale on a image. Image can be aligned and zoomed to verify scale
    and scale update."""
    
    infolder=os.path.dirname(infile)
    imgfile=os.path.basename(infile)    
    img=plt.imread(infile)
    
    #plot green channel
    plt.clf()
    plt.imshow(img[:,:,1])
    
    #fix pixsize: 300 um is 810 units of axis (pixels).
    #visible is set, scale should be immediately plotted.
    s=Scale(300/810.,'um',True) #810 pixel for 30 small ruler units, each unit is 10 um. pixsize is 300/810.
    
    if outfolder is not None:
        plt.savefig(os.path.join(outfolder,fn_add_subfix(imgfile,'_01','.png')))
    
    #zoom 2x 
    plt.xlim(np.array(plt.xlim())//2)
    plt.ylim(np.array(plt.ylim())//2)
    s.draw()
    if outfolder is not None:
            plt.savefig(os.path.join(outfolder,fn_add_subfix(imgfile,'_02','.png')))
    return s    

def test_with_plot(outfolder=None):
    """Apply scale to a line plot, it should work the same as an image."""
    
    N=100  #nr. of points in plot
    L=20.  #length of x axis
    x=np.arange(N)*L/(N-1)
    y=np.random.random(N)
    plt.clf()
    plt.plot(x,y)
    
    #fix pixsize: 300 um is 810 units of axis (pixels).
    #visible is set, scale should be immediately plotted.
    s=Scale(L/span(plt.xlim(),True),'units of x',True) #The entire axis span is L
    
    if outfolder is not None:
        plt.savefig(os.path.join(outfolder,'test_with_plot_01.png'))
    
    #zoom 3x 
    #plt.xlim(plt.xlim()[0]/3.,plt.xlim()[1]/3)
    #plt.ylim(plt.ylim()[0]/3.,plt.ylim()[1]/3)    
    s.t.set_color('black')
    s.l.set_color('black')
    s.draw()
    
    plt.grid(1)
    
    if outfolder is not None:
            plt.savefig(os.path.join(outfolder,'test_with_plot_02.png'))
    return s    

    
if __name__=='__main__':
    from dataIO.fn_add_subfix import fn_add_subfix
    import os
    #plt.ion()
    
    infolder=r'test\input_data'
    outfolder=r'test\Scale'
    imgfile='03_calib_20x.jpg'  #img of a ruler at 20x

    plt.close('all')
    
    # this partially fails: 1) color is set and drawn, but reset to yellow on 
    # scale.draw(); 
    # 2) starts from the zoomed version, that is considered home in zoom history even if the image/plot is larger. 
    # 3) makes wrong placement of scale on strong zoom.
    s=test_with_image(os.path.join(infolder,imgfile),outfolder)
    plt.show(block=True) #this goes in main or in scripts that are called by command line

    #this also 
    plt.figure()
    s2=test_with_plot(outfolder)
    s2.t.set_color('black')
    s2.l.set_color('black')
    plt.show(block=True) #this goes in main or in scripts that are called by command line    
   