def intersect_plane_line(pNormal,pPoint,lVersor,lPoint, epsilon=0.000001):
    
    w = lPoint-pPoint
    dot = pNormal.dot( lVersor)

    if abs(dot) > epsilon:
        # the factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        fac = -pNormal.dot(w) / dot
        u=lVersor*fac
        return lPoint+u
    else:
        # The segment is parallel to plane
        return None
