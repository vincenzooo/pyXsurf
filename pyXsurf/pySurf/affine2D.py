import numpy as np
import math
from .points import rotate_points,translate_points
import logging
import itertools

from matplotlib import pyplot as plt

## rotate_center -> points.rotate_points
## translate -> points.translate_points

def rot_center_func(theta,center=(0,0)):
    def func(x):
        return rotate_points(x,theta,center)
    return func
    
def translate_func(offset=(0,0)):
    def func(x):
        return translate(x,offset)
    return func
    
def rototrans_func(theta,center=(0,0),offset=(0,0)):
    """return a function that rotate by theta about center THEN translate by offset"""
    def func(x):
        if (x.shape[-1]==3):
            xy=x[:,0:2]
            return np.hstack([translate_points(rotate_points(xy,theta,center),offset),x[:,2,np.newaxis]]) 
        else:  
            return translate_points(rotate_points(x,theta,center),offset)
    return func

def apply2D(func,*args,**kwargs):
    """given a function acting on 2D points as array[N,2], returns a function
    that do same action on first two coordinates of a 3D array, leaving the third unchanged.
    e.g. rototrans_func=apply2D(translate_points(rotate_points()))
    """
    
    def func3d(x):
        if (x.shape[-1]==3):
            xy=x[:,0:2]
            return np.hstack([func(xy,*args,**kwargs),x[:,2,np.newaxis]]) 
        else:  
            return func(x,*args,**kwargs)
            
    return func3d
    
def _angpos2(xy):
    '''given some notable points, return the angular positions 
    of points with respect to barycenter. The angle returned is in the range [-pi:pi].
    Return angle and radius for each point and barycenter position.'''

    #N xy points are expressed as array N x 2
    b=(np.sum(xy,0))/(xy.shape[0])    #calculate barycenter
    deltaxy=xy-(b.repeat(xy.size/2).reshape(2,-1)).T
    r=np.sqrt(np.sum(deltaxy**2,1))
    theta=np.arctan2(deltaxy[:,1],deltaxy[:,0])
    return theta,r,b    


def plot_transform(points,plotLines=None,labels=None,transform=None):
    """plot a set of points or shapes and their transformed.
    points is a list of sets of coordinates, one for each groups of points to be plotted with same style and transformation. Stile and transformation can be passed as single value or list.
    plotlines is an array of flags to plot as lines(True) or points (False,default).
    labels set of labels of same length
    transform list (same len as points) of transformation functions or single function to be 
    applied respectively to each set of points passed.
    
    #this plots some markers and a rectangle (see examples in module):
    plt.figure() plot_transform([primary,secondary,rect1],[0,0,1],['markers1','markers2','ROI'],[transform2,None,transform2])
    plt.show()
    display(plt.gcf())
    """
    
    syms = itertools.cycle(('^', '+', 'x', 'o', '*')) 

    if not(hasattr(points,'__len__')): #transform in list if passed as scalar
        points=[points]
    elif not(hasattr(transform,'__len__')):
        transform=np.repeat(transform,len(points))
    points= [np.array(p) for p in points]
    if hasattr(transform,'__len__') and len(transform)==1:
        transform=np.repeat(transform,len(points))
    elif not(hasattr(transform,'__len__')):
        transform=np.repeat(transform,len(points))
    if plotLines is None:
        plotLines=np.repeat(False,len(points))
    if labels is None:
        labels=np.repeat('_nolegend_',len(points))
        
    #plt.clf()
    plt.grid(1)
    plt.gca().set_aspect('equal')
    for markers1,pl,l,trans in zip (points,plotLines,labels,transform):
            
        if pl:
            plt.plot(markers1[:,0],markers1[:,1],label=l)
        else:
            plt.plot(markers1[:,0],markers1[:,1],next(syms),label=l)

        if trans is not None:
            m=trans(markers1)
            if pl:
                plt.plot(m[:,0],m[:,1],label=l+' trans.')
            else:
                plt.plot(m[:,0],m[:,1],next(syms),label=l+' trans.')           
    
    # Shrink current axis by 20%
    if np.any(points):
        ax=plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.legend(loc=0)
    else:
        print('no markers to plot')

def find_rototrans(markers1,markers2,pars=None):
    """Return the transformation that applied to markers1 gives markers2.
    If list is passed in pars, transformation parameters mrot,b1,bartrans are appended."""
    #used by calibrate_align as find_transform
  
    markers1=np.array(markers1)
    markers2=np.array(markers2)
    
    assert np.ndim(markers1)==2
    assert markers1.shape==markers2.shape
    
    gang0,gr,gb=_angpos2(markers1)
    mang0,mr,mb=_angpos2(markers2)
    
    #determine the rotation angle mrot.
    #Note that this is not correct, angles should be weighted by distance from barycenter:
    #   an error on x and y has differenct impact on the angular position of point according
    #   to distance from the center.
    #note that if there is a large rotation, angles can be messed up by preiodicity,
    #   need to account for that.
    mrot0=gang0[0]-mang0[0] #angular displacement between first elements
    #g-gang0[0] and g-mang0 are in the range [-2pi,2pi], transform them in range [0,2pi]
    gang=np.array([(g-gang0[0]) if (g-gang0[0])>=0 else (g-gang0[0]+2*math.pi) for g in gang0])
    mang=np.array([(g-mang0[0]) if (g-mang0[0])>=0 else (g-mang0[0]+2*math.pi) for g in mang0])
    mrot=(gang-mang).mean()+mrot0
    
    bartrans=gb-mb
    try: 
        logger
    except NameError:
        logger=logging.getLogger()
        logger.setLevel(logging.DEBUG)
    #logger=logging.getLogger()
    logger.info("stdv of markers distance errors from barycenter: %s"%((gr-mr).std()))
    logger.info('rotation angle (degrees): %s +-%s'%(mrot*180./math.pi,(gang-mang).std()*180./math.pi))
    logger.info('translation: %s'%(bartrans))

    #define the transformation function. I will rotate glass data
    trans=rototrans_func(-mrot,gb,-bartrans)  #(mrot,mb,bartrans) to rotate mandrel 
    #if gmarkers.shape[0]: transform=a2d.find_affine(mmarkers,gmarkers)
    err=markers2-trans(markers1)
    logger.info("errors in markers position after rotations: \n%s"%(err))
    logger.info("mean of error abs val: \n%s"%(np.sqrt((err**2).sum(axis=1)).mean()))
    if pars is not None:
        pars.extend([mrot,b1,bartrans])
    return trans 

def find_affine(markers1, markers2,pars=None):
    """Return a function that can transform points from the first system to the second. 
    if pars is set to a list, append the matrix A of the transformation, that can be applied to a vector x with:
    unpad(np.dot(pad(x), A))
    markers1 and markers2 are sets of points in format [Npoints, Ndim]. Transformation matrix A is [Ndim+1 x Ndim+1].
    
    http://stackoverflow.com/questions/20546182/how-to-perform-coordinates-affine-transformation-using-python-part-2
    """
    markers1=np.array(markers1)
    markers2=np.array(markers2)
    
    # Pad the data with ones, so that our transformation can do translations too
    n = markers1.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])  #adds a column of ones to the right 
    unpad = lambda x: x[:,:-1]   #remove rightmost column
    X = pad(markers1)
    Y = pad(markers2)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond = None)
    
    #note: the transformation to and from matrix is needed to make it work also if x is a single point.
    trans = lambda x: np.array(unpad(np.dot(pad(np.array(x)), A)))
    if pars is not None:
        pars.append(A)
    
    #calculate parameters and log copied from find_rototrans, ideally should be 
    #   derived from transformation matrix. see e.g.
    # https://trac.osgeo.org/postgis/wiki/DevWikiAffineParameters
    #
    # empirically:
    # A[2,0]= offset
    # A[2,1]= rotation angle
    # A[0,1], A[1,0] = x,y (y,x?) scales
    
    gang0,gr,gb=_angpos2(markers1)
    mang0,mr,mb=_angpos2(markers2)
    bartrans=gb-mb
    try: 
        logger
    except NameError:
        logger=logging.getLogger()
        logger.setLevel(logging.DEBUG)
    logger.info('- INPUT -' )
    logger.info("stdv of markers distance errors from barycenter: %s"%((gr-mr).std()))
    logger.info('translation: %s'%(bartrans))
    logger.info('- OUTPUT -' )  
    tmarkers=trans(markers1)
    tang0,tr,tb=_angpos2(tmarkers)
    logger.info("Affine Transformation Matrix:\n %s"%A)
    logger.info("stdv of markers distance errors from barycenter: %s"%((gr-tr).std()))
    logger.info('translation of barycenter: %s'%(gb-tb))
    err=markers2-trans(markers1)
    logger.info("errors in markers position after transformation: \n%s"%(err))
    logger.info("mean of error abs val: \n%s"%(np.sqrt((err**2).sum(axis=1)).mean()))
    
    return apply2D(trans)  


if __name__=="__main__":
    primary = np.array([[40., 1160., 0.],
                        [40., 40., 0.],
                        [260., 40., 0.],
                        [260., 1160., 0.]])

    secondary = np.array([[610., 560., 0.],
                          [610., -560., 0.],
                          [390., -560., 0.],
                          [390., 560., 0.]])
    
    print("-----------")
    print("Test affine")             
    print("-----------")         
    transform,A= find_affine(primary, secondary)
    print("Starting points:")
    print(primary)
    print("Target:")
    print(secondary)
    print("Result:")
    print(transform(primary))
    print("Max error:", np.abs(secondary - transform(primary)).max())
    
    
    print("-----------")
    print("Test rototrans")   
    print("-----------")
    secondary2=rotate_points(primary,math.pi/6,[-10,2])
    transform2,A2= find_rototrans(primary, secondary2,verbose=True)
    print("Result:")
    print(transform2(primary))
    

    rect1=np.array([[30,1200],[30,-600],[650,-600],[650,1200],[30,1200]])
    
    plt.figure()
    plot_transform([primary,secondary,rect1],[0,0,1],['markers1','markers2','ROI'],[transform2,None,transform2])
    plt.show()
    display(plt.gcf())
   
