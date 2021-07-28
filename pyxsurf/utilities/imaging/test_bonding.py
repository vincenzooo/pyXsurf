from utilities.imaging.userKov.pySurf.points import *
#from userKov.calibrate_align import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

def closest_point_on_line(points,lVersor,lPoint):
    """From a list of points in 3D space as Nx3 array, returns a Nx3 array with the corresponding closest points on the line."""
    vd=lVersor/((np.array(lVersor)**2).sum())  #normalize vector, not sure it is necessary.
    return lPoint+(points-lPoint).dot(vd)[:,np.newaxis]*(vd)

def cylinder_error3(odr=(0,0,0,0),points=None,extra=False,xy=False):  
#from v1
#origin=(0,0,0),direction=(0,1.,0),radius is calculated from best fit
    """Given a set of N points in format Nx3, returns the rms surface error on the cylinder defined by origin (intercept of the axis with x=0) and direction, 
    passed as 4-vector odr (origin_y,origin_z,direction_x,direction_z). 
    Best fit radius for odr is calculated as average. 
    If extra is set, additional values are returned :
        radius: best fit radius for the cylinder.
        deltaR[N,3]: deviation from radius for each point.
    """
    #ca 400 ms per loop
    if xy:
        origin=(0,odr[0],odr[1])
        direction=(1.,odr[2],odr[3])
    else:
        origin=(odr[0],0,odr[1])
        direction=(odr[2],1.,odr[3])
    
    #vd=direction/((np.array(direction)**2).sum())  #normalize vector, not sure it is necessary.
    x,y,z=np.hsplit(points,3)
    Paxis=closest_point_on_line(points,direction,origin)
    #origin+(points-origin).dot(vd)[:,np.newaxis]*(vd) #points on the cylinder axis closer to points to fit
    R=np.sqrt(((points-Paxis)**2).sum(axis=1)) #distance of each point from axis
    radius=R.mean()  #
    #plt.plot(R-radius)
    deltaR=(R-radius)  #square difference from expected radius for each point
    fom=np.sqrt(((deltaR)**2).sum()/len(deltaR))
    residuals=np.hstack([x,y,deltaR[:,None]])
    if extra: return fom,residuals,radius
    else: return fom  #,deltaR,radius


if __name__=="__main__":
    #always run this
    gfile='/home/rallured/Dropbox/WFS/SystemAlignment/Bonding/150529BottomLeft_Center.txt'
    fit_func=cylinder_error3

    pts=get_points(gfile,matrix=1, xrange=[0,127],yrange=[0,127])
    pts=crop_points(pts,[23,80],[15,93])
    pts[:,2]=pts[:,2]/1000.   #data are um, convert to mm before fit

    #level plane
    pts,pl=level_points(pts,returnPars=1)
    pts[:,2]=pts[:,2]-pts[np.argsort(pts[:,2])[0],2]

    pts[:,2]=-pts[:,2] #invert concavity

    plt.figure(1)
    plt.clf()
    plot_points(pts,shape=[50,50])

    def p(x): print(x) # callback function 
        #passed to minimization to print each step.

    odr2=(0,5000.,0,0.) #use nominal value for guess direction
    result=minimize(fit_func,x0=(odr2),
                    args=(pts,),options={'maxiter':500},method='Nelder-Mead',callback=p)
    print('-----------------------------------')
    print('Results of fit on region:')
    print(result)    
    odr=result.x
    fom,deltaR,coeff=fit_func(odr,pts,extra=True)
    print('Angle of cylinder axis with y axis (deg):')
    print(np.arccos(np.sqrt(1-(odr[-2:]**2).sum()))*180/np.pi)
    print('Cylinder parameters (angle(deg), R@y=0):')
    print(np.arctan(coeff)*180/np.pi,coeff)

    plt.figure(2)
    plt.clf()
    plot_points(deltaR,shape=[50,50])
    plt.show()

    
