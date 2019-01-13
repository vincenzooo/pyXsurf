import matplotlib.pyplot as plt
import numpy as np

def plot_poly(mh,ls='',ms=''):
    """plot a polygon defined by an array of vertices as list of x and y.
    linestyle and markerstyle can be passed"""
    
    mh=np.array(mh)
    mh=np.vstack([mh,mh[0,:]])
    if ms is not None:
        plt.plot(mh[:-1,0],mh[:-1,1],ms)
    
    return plt.plot(mh[:,0],mh[:,1],ls)

def plot_rect(xs,ys,ls='',ms=''):
    """plot a polygon defined by an array of vertices as list of x and y.
    linestyle and markerstyle can be passed"""
    
    mh=[[xs[0],ys[0]],[xs[1],ys[0]],[xs[1],ys[1]],[xs[0],ys[1]]]
    return plot_poly(mh,ls=ls,ms=ms)
    
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

def plot_circle_ROI(c,r,color='r'):
    """plot center and circunference (color)."""
    plt.plot(c[0],c[1],'o',color=color,alpha=0.5)
    c1=plt.Circle(c,r,fill=None,color=color)
    plt.gca().add_artist(c1)
    #plt.draw()
    return c1

def plot_circle3points(fiducials):
    """given three fiducial points, plot them and the circle passing by them.
    Return line (for points) and circle artists."""
    
    fid=np.array(fiducials)
    pfid=plt.plot(fid[:,0],fid[:,1],'x')
    ax=plt.gca()
    
    c,r=circle3points(fiducials)
    ax.set_aspect('equal')
    cir=plt.Circle(c,r,fill=0)
    ax.add_artist(cir)
    return pfid,cir

def plot_positions(fiducials,positions,labels=None,*args,**kwargs):
    """plot circle on three point and position of measurements.
	This can be replaced by a combination of markers and plot_circle3points."""
    
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
	fid=((77.312, 87.499), (77.899,102.718), (29.208,103.132))
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

