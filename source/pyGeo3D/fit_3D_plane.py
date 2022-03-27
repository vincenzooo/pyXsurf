import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

def fit_3D_plane(points,plot=False):
    """Find best fit plane for points. Points is a Nx3 array.
       If a plane is defined by the equation a*x+b*y+c*z+d=0, then the normal has
       direction [a,b,c]. The plane is returned as a tuple (normal,ctr).
       Note that there is some ambiguity in the orientation of the normal. I believe it is related to the sign of d. Also some mistake in plot of normal. """
    # a plane is a*x+b*y+c*z+d=0 [a,b,c] is the normal.
    # The normal is obtained by svd.
    # Note that probably the results can be obtained by same calculation as
    #   line fit, but taking a different row/column
    ctr = points.mean(axis=0)
    x=points-ctr
    M=np.dot(x.T,x)
    normal=np.linalg.svd(M)[0][:,-1]
    if (plot):
        ax=plt.gca()
        ax.scatter3D(*points.T,color='y')
        plt.xlabel('X')
        plt.ylabel('Y')
        normalArrow=np.vstack([ctr,ctr+normal])
        ax.plot3D(*normalArrow.T)
        """
        xx, yy = np.meshgrid(range(10), range(10))
        span=((points.max(axis=0)-points.min(axis=0))/vv[0]).max()
        
        # calculate corresponding z
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

        # plot the surface
        plt3d = plt.figure().gca(projection='3d')
        plt3d.plot_surface(xx, yy, z)
        
        """
        #calcolate coordinates of corners of the plane to plot
        xy0=points.min(axis=0)[0:2]
        xy1=points.max(axis=0)[0:2]
        xyvert=np.array([(xy0[0],xy0[1]),(xy0[0],xy1[1]),(xy1[0],xy1[1]),(xy1[0],xy0[1])])
        d=-ctr.dot(normal)
        z = (-normal[0] * xyvert[:,0] - normal[1] * xyvert[:,1] - d)/normal[2] #questo e' cannato.
        p=np.vstack([xyvert.T,z]).T
        vert=[[(list(p[i,:])) for i in range(p.shape[0])]]
        poly=m3d.art3d.Poly3DCollection(vert)
        ax.add_collection3d(poly)
        ax.scatter3D(*p.T,color='r')
        plt.show()   
        
    return  normal,ctr
    """z= a + b*y +c*x
    x,y,zi=points[:,0],points[:,1],points[:,2]
    A = np.column_stack((np.ones(x.size), x, y))
    c, resid,rank,sigma = np.linalg.lstsq(A,zi)
    return c"""
