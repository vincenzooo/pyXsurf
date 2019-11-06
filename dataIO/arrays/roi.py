from dataIO.span import span

def external_roi_rect(points):
    """
    finds the smallest (straight to axis) rectangle containing all points.
    
    Points are passed as N x 2 array, returns two couples (x0,x1) (y0,y1)
    """
    return span(points,axis=0)


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
    
