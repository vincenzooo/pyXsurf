#from stackoverflow
import numpy as np


def find_affine(primary, secondary):
    """Return a function that can transform points from the first system to the second
    """
    # Pad the data with ones, so that our transformation can do translations too
    n = primary.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(primary)
    Y = pad(secondary)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))

    print("Target:")
    print(secondary)
    print("Result:")
    print(transform(primary))
    print("Max error:", np.abs(secondary - transform(primary)).max())
    return transform,A

    
def original_test():
    primary = np.array([[40., 1160., 0.],
                        [40., 40., 0.],
                        [260., 40., 0.],
                        [260., 1160., 0.]])

    secondary = np.array([[610., 560., 0.],
                          [610., -560., 0.],
                          [390., -560., 0.],
                          [390., 560., 0.]])
                          
    return find_affine(primary, secondary)

if __name__=="__main__":
    original_test()
    """
    markers1=[]
    markers2=[]
    points=[]
    transform=find_affine(markers1,markers2)
    points_transformed=trasform(points)
    """
