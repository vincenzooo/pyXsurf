import numpy as np
import matplotlib.pyplot as plt
import np.polynomial.chebyshev as cheb

def chebFit(d,xcoeff,ycoeff):
    """Perform a fit to 2D data with 2D Chebyshev polynomials.
    """
    
