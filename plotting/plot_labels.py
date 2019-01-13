import matplotlib.pyplot as plt
import numpy as np


def plot_labels(labels, xdata, ydata, xoffset=None, yoffset=None,plain=False):
    """"Add fancy labels with arrows pointing at xdata,ydata.
    offsets can be passed as offsets in data coordinates as scalar or vectors.
    default is a fraction of axes, or can use """
    
    #set offsets format as vectors
    frac=1/20. #fraction of full axis range used as default offset 
    xyc='data' #'offset points'
    if xoffset is None:
        #xyc = 'axes fraction'
        #xoffset = - frac
        xoffset = -(plt.xlim()[1]-plt.xlim()[0])*frac
    if yoffset is None:
        #xyc = 'axes fraction'
        #yoffset = frac
        yoffset = (plt.ylim()[1]-plt.ylim()[0])*frac
    if np.size(xoffset)==1:
        xoffset=np.repeat(xoffset,len(labels))
    if np.size(yoffset)==1:
        yoffset=np.repeat(yoffset,len(labels))
    
    for label, x, y, xo, yo in zip(labels, xdata, ydata, xoffset, yoffset):
        if plain:
            plt.text(x+xo, y+yo, label)
        else:
            plt.annotate(
                label, 
                xy = (x, y), xytext = (x+xo, y+yo), 
                textcoords = xyc, ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            
if __name__=="__main__":
    #this is what I want to pass as label, it must be a 3 element list with [string, xpos, ypos]
    xdata=(1.,2.,3.)
    ydata=(1.,4.,9.)
    labels=('uno','due','tre')
    
    plt.figure(1)    
    plt.plot(xdata,ydata,'o')
    plt.plot(xdata,ydata)
    plot_labels(labels, xdata, ydata, plain=True)
    plt.show()
    
    plt.figure(2)
    plt.plot(xdata,ydata,'o')
    plt.plot(xdata,ydata)
    plot_labels(labels, xdata, ydata, plain=True)
    plt.show()