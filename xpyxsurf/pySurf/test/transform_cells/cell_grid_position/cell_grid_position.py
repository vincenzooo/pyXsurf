import numpy as np
import math
#from points import *

def rotate_points(points,theta,center=(0,0)):
    """returns rotated coordinates of 2D point(s) x ([Npoints x 2]) about a center with anticlockwise angle theta in rad. If 3D points are passed, z coordinate is maintained."""
    tx,ty=center
    if (points.shape[-1]==3):
        return np.hstack([rotate_points(points[:,0:2],theta,center),points[:,2:]])
    else:
        if(points.shape[-1]!=2):
            raise Exception
    x,y=np.array(points[:,0]),np.array(points[:,1])
    cost=math.cos(theta)
    sint=math.sin(theta)
    x1=x*cost-y*sint + ty*sint - tx*(cost-1)
    y1=x*sint+y*cost - tx*sint - ty *(cost-1)
    return np.vstack((x1,y1)).T

def translate_points(x,offset=(0,0)):
    """returns translated coordinates of 2D point(s) x ([Npoints x 2]) by an offset."""
    points=np.matrix(x)
    x,y=np.array(points[:,0]),np.array(points[:,1])
    x1=x+offset[0]
    y1=y+offset[1]
    return np.hstack((x1,y1))
    
def _angpos2(xy):
    '''given an image and some notable points, return the angular positions of points with respect to barycenter.
    The angle returned is in the range [-pi:pi]'''

    #N xy points are expressed as array N x 2
    b=(np.sum(xy,0))/(xy.shape[0])    #calculate barycenter
    deltaxy=xy-(b.repeat(xy.size/2).reshape(2,-1)).T
    r=np.sqrt(np.sum(deltaxy**2,1))
    theta=np.arctan2(deltaxy[:,1],deltaxy[:,0])
    return theta,r,b
    
def rototrans_func(theta,center=(0,0),offset=(0,0)):
    """return a function that rotate by theta about center THEN translate by offset"""
    def func(x):
        if (x.shape[-1]==3):
            xy=x[:,0:2]
            return np.hstack([translate_points(rotate_points(xy,theta,center),offset),x[:,2,np.newaxis]]) 
        else:  
            return translate_points(rotate_points(x,theta,center),offset)
    return func
    
def find_rototrans(primary, secondary,verbose=False):
    """Return a function that can transform points from the first system to the second by means of a rototranslation. Also (not implemented yet, but kept for interface consistency with find_affine) return the matrix A of the transformation, that can be applied to a vector x with unpad(np.dot(pad(x), A)).
    primary and secondary are sets of points in format [Npoints, 2]. Transformation matrix A is [Ndim+1 x Ndim+1]."""
    
    ang1,r1,b1=_angpos2(primary[:,0:2])   #glass-> 1, mandrel-> 2
    ang2,r2,b2=_angpos2(secondary[:,0:2])
    mrot=(ang2-ang1).mean()
    bartrans=b2-b1
    #define the transformation function. I will transform primary data
    transform=rototrans_func(mrot,b1,bartrans)
    #if gmarkers.shape[0]: transform=a2d.find_affine(mmarkers,gmarkers)
    if verbose:
        print("stdv of markers distance errors from barycenter: %s"%((r1-r2).std()))
        print('rotation angle (degrees): %s +-%s'%(mrot*180./math.pi,(ang1-ang2).std()*180./math.pi))
        print("errors in markers position after rotations:")
        print(secondary-transform(primary))
    return transform,[mrot,b1,bartrans] 

def distance_table(points):
    for ip,p in enumerate(points): 
        for iq,q in enumerate(points):
            if ip<iq: print(ip+1,iq+1,np.sqrt((((p-q)**2)[0:2]).sum())/2)

if __name__ == '__main__':
    
    #these are the points 
    primary = np.array([[-26,26., 0.],
                        [-26.,-26., 0.],
                        [26., -26., 0.],
                        [26., 26., 0.]])
                        
    xyz=np.genfromtxt('reference_points_input.txt')
    cells=np.genfromtxt('cell_center_input.txt')
        
    transform2,A2= find_rototrans(primary, xyz,verbose=True)
    print('Input position of reference points:')
    print(xyz)
    print('\nDistance table (measured reference points):\npoint1 point2 distance')
    distance_table(xyz)
    
    t2=transform2(primary)    
    print('\nTransformed coordinates of reference points in measurement system')
    print(t2)
    print('\nDistance table (converted nominal reference points):\npoint1 point2 distance')
    distance_table(t2)
    
    print('\nTransformed coordinates of input points')
    print(transform2(cells))
