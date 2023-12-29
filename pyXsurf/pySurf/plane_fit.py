import numpy as np

def plane_fit (x, y, z):
    """
    #modified by kov
    #x,y,z 3 vectors with coordinates of points (same number of elements).
    #return value [A,B,C] of plane Ax + By + C = z
    #planesurf is a vector with z of plane points

    #compute the average surface, calculate statistical indicator
    # z can be nan, if there are nan in x and y it is likely not to woork.

      # M. Katz 1/26/04
    # IDL function to perform a least-squares fit a plane, based on
    # Ax + By + C = z
    #
    # ABC = plane_fit(x, y, z, error=error)
    """
    #check length of vectors.
    if len(y) != len(x):
        raise ValueError("wrong number of elements for x and y.")
    if len(z) != len(x):
        raise ValueError("wrong number of elements for z.")
    
    gi=~np.isnan(z)
    x=x[gi]
    y=y[gi]
    z=z[gi]
    
    tx2 = np.sum(x**2)
    ty2 = np.sum(y**2)
    txy = np.sum(x*y)
    tx = np.sum(x)
    ty = np.sum(y)
    N = len(z)

    a = np.array([[tx2, txy, tx],
        [txy, ty2, ty],
        [tx, ty, N ]])

    b = np.array([np.nansum(z*x), np.nansum(z*y), np.nansum(z)])
    out = np.array(np.linalg.inv(a) @ b[:,None])
    #plane=(out[0]*x+out[1]*y+out[2])

    return out

if __name__=='__main__':
    #plane parameters
    a=555.5
    b=-3
    c=-15

    #grid
    npx=100
    npy=100

    xp=np.arange(npx)
    yp=np.arange(npy)
    x,y=list(map(np.ndarray.flatten, np.meshgrid(xp,yp)))
    z=x*a+y*b+c

    print("starting value:",a,b,c)
    print(" fit results: ",plane_fit(x,y,z))

  