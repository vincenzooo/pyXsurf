
import numpy as np
from matplotlib import pyplot as plt


def add_clickable_markers(propertyname='markers',fig=None,ax=None,key=None):
    """Makes pickable the image (image only) in the current figure.
    add a list property to the figure.
    Propertyname doesn't really work yet, but works fine with a single set
    of markers named markers"""
    
    #key='ctrl'
    
    def onclick(event):
        button=event.button
        x=event.xdata
        y=event.ydata
        fig=event.canvas.figure
        ax=event.inaxes

        if button==1: 
            if key is None or event.key==key:
            #if event.key==key:
                l=getattr(fig,propertyname)
                l.append([event.xdata, event.ydata])
                #add point
                ax.set_autoscale_on(False)
                ax.plot(x,y,'or',picker=5,markerfacecolor=None)
        if button==3:
            if key is None or event.key==key:
            #if event.key==key:
                l=getattr(fig,propertyname)
                #print 'list',l
                dist=((np.array(l)-(x,y))**2).sum(axis=1)
                pop=l.pop(np.argmin(dist))
                #np.delete(l,np.argmin(dist),0)
                #print 'clicked:',x,y
                #print 'nearest:', pop
                #dist2=[((np.array(pop)-(a.get_xdata(),a.get_ydata()))**2).sum()  for a in ax.lines]
                dist2=[(np.array((pop[0]-a.get_xdata(),pop[1]-a.get_ydata()))**2).sum()  for a in ax.lines]
                #print dist2
                
                ax.lines.pop(np.argmin(dist2))
        
        plt.draw()
        #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        #event.button, event.x, event.y, event.xdata, event.ydata)
    
    if fig is None:
        fig=plt.gcf()

    if ~hasattr(fig,propertyname):
        setattr(fig,propertyname,[])
        
    #ax.imshow(im)
    #ax.autoscale(False)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    return fig
    
    
    #to disconnect:
    #fig.canvas.mpl_disconnect(cid)
        
    def on_pick(event):
        # not used. this is an alternative way, it automatically manage the selection of the plotted point.
        import matplotlib
        thisline = event.artist
        if isinstance(thisline,matplotlib.lines.Line2D):
            #code to remove point from markers list
            thisline.remove()
            plt.show()
        
        
if __name__=="__main__":

    def retax():
        """diagnostic routine. return coordinates of all plotted points,
        assuming this are the symbols for the plotted markers."""
        ax=plt.gca()
        return [(a.get_xdata(),a.get_ydata()) for a in ax.lines]


    #from generalPlotting.pickPointList import add_clickable_markers
    imfile=r'test\input_data\IMG_4509_crop.jpg'
    a=plt.imread(imfile)
    plt.clf()
    plt.imshow(a)
    f=add_clickable_markers()
    
    plt.show()
    ax=plt.gca()
    
    print('add points with left button, remove closest with right. retax() prints the points plotted on current axes, f.markers print marker list.')
    