import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

def fit_3D_line(points,plot=False):
    '''return direction vector and point for the fitting line.
    Note that it should contain also somewhere'''
    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = points.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(points - datamean)
    
    if (plot):
        ax=plt.gca()
        ax.scatter3D(*points.T)  #this must be     ax = m3d.Axes3D(plt.figure())
        span=((points.max(axis=0)-points.min(axis=0))/vv[0]).max()
        linepts = vv[0] * np.mgrid[-span/2:span/2:2j][:, np.newaxis]+datamean
        ax.plot3D(*linepts.T)
        plt.show()
        
    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.
    return vv[0],datamean
