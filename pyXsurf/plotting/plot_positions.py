import matplotlib.pyplot as plt
import numpy as np

"""Plot markers and shapes in dependance on positions of a group of points."""

'''
def plot_poly(v,ls='',ms=''):
    """plot a polygon defined by a list of vertices `v` as coordinate couples.
    linestyle and markerstyle can be passed.
    """
    
    mh=np.array(mh)
    # duplicate first point at the end: 
    mh=np.vstack([mh,mh[0,:]]) 
    # plt.plot return a list of one element
    # if there are markers, plot them:
    if ms is not None:
        m = plt.plot(mh[:-1,0],mh[:-1,1],ms)[0]
    
    #plot lines
    return plt.plot(mh[:,0],mh[:,1],ls)[0]
'''
    
def plot_poly(v,*args,**kwargs):
    """plot a polygon defined by a list of vertices `v` as coordinate couples.
    Just a thin wrapper around `plt.plot`, it just accepts points in transpose form and adds a copy of first point at the."""
    
    v=np.array(v)
    # duplicate first point at the end: 
    v=np.vstack([v,v[0,:]]) 
    # plt.plot return a list of one element
    
    #plot lines
    return plt.plot(*v.T,*args,**kwargs)

def plot_rect(c1,c2,*args,**kwargs):
    """plot a polygon defined by a 2 element list with coordinates of opposite corners.
    linestyle and markerstyle can be passed.
    es. 
    
    2020/10/13 changed to coordinates of opposite corners from [xspan,yspan]"""
    
    mh=[[c1[0], c1[1]],[c2[0],c1[1]],[c2[0],c2[1]],[c1[0],c2[1]]]
    #mh=[[xs[0],ys[0]],[xs[1],ys[0]],[xs[1],ys[1]],[xs[0],ys[1]]]
    return plot_poly(mh,*args,**kwargs)
    
def plot_circle_ROI(c,r,color='r'):
    """plot center and circunference (color)."""
    plt.plot(c[0],c[1],'o',color=color,alpha=0.5)
    c1=plt.Circle(c,r,fill=None,color=color)
    plt.gca().add_artist(c1)
    #plt.draw()
    return c1

def circle3points(fiducials):
    """Calculate circle passing by three points.
    Return center and radius."""
    
    fid=np.array(fiducials)
    
    #plot circle on three points on wafer edge and measurement point
    x,y,z=(fid*[1,1j]).sum(axis=1)
    w = z-x
    w /= y-x
    c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
    return (-c.real, -c.imag), abs(c+x)
    
def plot_circle3points(fiducials):
    """given three fiducial points, plot them and the circle passing by them.
    Return line (passing through points) and circle artists."""
    
    fid=np.array(fiducials)
    pfid=plt.plot(fid[:,0],fid[:,1],'x')
    ax=plt.gca()
    
    c,r=circle3points(fiducials)
    ax.set_aspect('equal')
    cir=plt.Circle(c,r,fill=0)
    ax.add_artist(cir)
    return pfid,cir
    
def plot_positions(fiducials,positions,labels=None,*args,**kwargs):
    """
    Obsolete: use a combinatio of plt commands, or make better separated functions to plot circles, markers, labels, etc.
    
    plot circle on three point `fiducials` and mark the three points, plot `labels` at `positions`.
    fiducials is a list of three coordinate couples.
    positions a list of n points in same format.
    label can be set to empty string to generate numeric labels (useless, can pass e.g. `[str(i).zfill(2) for i in range(N)]`).
    
    This can be replaced by a combination of markers and plot_circle3points.
    Praticamente una funzione bruttina che plotta un combinazione inutile di roba.
    Si potrebbe notevolmente migliorare.
    """
    
    pfid,cir=plot_circle3points(fiducials)
    pos=np.array(positions)    
    fs=kwargs.pop('fontsize',20)
    
    if labels is not None:    
        if labels=='':
            labels=['%3i'%ll for ll in range(pos.shape[0])]
        plab=[]
        for p,l in zip(pos,labels):
            plab.append(plt.text(p[0],p[1],l,fontsize=fs))
    
    ppos=plt.plot(pos[:,0],pos[:,1],*args,**kwargs)
    plt.draw()
    return pfid,cir,ppos,plab

if __name__=="__main__":
    fiducials=((77.312, 87.499), (77.899,102.718), (29.208,103.132))
    positions=[[58.158,97.466],
    [59.652,104.989],
    [63.947,95.724],
    [68.501,97.656],
    [69.928,88.653],
    [48.077,88.663],
    [48.029,109.160]]

    plt.clf()
    pfid,cir,ppos,plab=plot_positions(fiducials,positions,c='y',ls='none',marker='o',fontsize=20,labels='')

    #if circle is not wanted can be removed
    cir.set_color('r')
    r = plot_rect([40,90],[50,100])
    r[0].set_linestyle(":")
    
    p = plot_poly([[45,95],[55,100],[35,105]])
    p[0].set_linestyle("-.")
    plt.draw()

