import numpy as np
import scipy
import scipy.signal
import pdb
from astropy.modeling import models,fitting

def legendre2d(d,x=None,y=None,xo=None,yo=None,xl=None,yl=None,
    indices=None,model=False,components=None,
    x_domain=None,y_domain=None,*args,**kwargs):
    """
    Fit a set of 2d Legendre polynomials to a 2D array. `x` and `y` coordinates can be passed and used to determine the domain of the Legendre components. 
    
    NaNs are automatically excluded from the analysis.
    xo = 2 fits up to quadratic in row axis
    yo = 3 fits up to cubic in column axis
    This function uses `astropy.modeling.models.Legendre2D` objects and call `fitting.LinearLSQFitter()` after setting model fixed parameters.
    
    Note also that `Legendre2D` class fits all combination of indices up to xo,yo, including mixed terms.
    As a consequence, a fit with xo=1,yo=1 will not necessarily remove a plan, as the x*y term can be non null (this still makes every slice to be a straight line, however the surface itself is curved, i.e. it has curved level curves). 
    
    `xl` and `yl` allow to specify to fit only the polynomials with
    orders `xl[i],yl[i]`. 
    Some notable examples are: 
    `xl=[0,0,1],yl=[0,1,0]` fits a plane, `xl=[0,0,1,2],yl=[0,1,0,0]` fits a cylinder with axis along y 
    `xl=[0,0,1,2,1],yl=[0,1,0,0,1]` fits a misaligned (rotation about x) cylinder with axis along y 
    `xl=[0,0,1,2,2],yl=[0,1,0,0,1]` fits a cone with axis along y.
    `xl=[0,0,1,2,1,2],yl=[0,1,0,0,1,1]` fits a misaligned (rotation about x) cone with axis along y 
    
    TODO: implement components,
    `components` offers an alternative (better) alternative to xl, yl.
    `   `, for example for the cascomponents = list(zip(xl,yl))e above, 
    `degree = (2,2)` gives 
    `components=[(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]`.
    
    `indices` can be used in alternative to the same purpose. It is a list of indices pointing to `model.parameters` and it sets them as variables to fit. 
    It has the advantage that parameters can be found by introspection. 
    N.B.: it might be convenient to use a boolean vector instead.
    For example:
    
        > models.Legendre2D(1,1).param_names
        Out[170]: ('c0_0', 'c1_0', 'c0_1', 'c1_1')
        indices = [0,1,2]  #plane
        > models.Legendre2D(2,2).param_names
        Out[169]: ('c0_0', 'c1_0', 'c2_0', 'c0_1', 'c1_1', 'c2_1', 'c0_2', 'c1_2', 'c2_2')
        indices = [0,1,2,3,5]  #cone
    
    Note also that, due to orthonormality of legendre polynomial, non fitting some terms is equivalent to set to zero the corresponding coefficient in the returned value: using xl,yl, or indices is equivalent to not using them and set the corresponding coefficients to zero in the best fit.
    
    The default aperture is assumed to be +-1 over each dimension,
    but can be changed passing keywords `x_domain, y_domain` and `x_window,y_window` `as kwargs` are passed to `Legendre2D`.
    
    2020/07/15 V. Cotroneo add `indices` keyword. Added `x` and `y` 
    .
    """
    
    def make_legendre_keys(xl,yl):
        """Taken two lists of coefficients (same length), return a list of indices 
        according to model parameters name.
        e.g. 
        p = models.Legendre2D
        pl_init = models.Legendre2D(xo2,yo2)
        key = 'c'+str(xx)+'_'+str(yy)
        """
        keys=[]    
        for xx,yy in zip(xl,yl):
            keys.append( 'c'+str(xx)+'_'+str(yy))
        return keys
    
    fit_p = fitting.LinearLSQFitter()  #Define fitting algorithm
    #pdb.set_trace()
    
    if xo is None:  #set maximum degree if not specified in xo, yo
        if xl is not None:
            xo = max(xl)
    else:
        if xo > max(xl):
            print("WARNING: xo cannot be smaller than xl. Adjusted to %i"%max(xl)) 
            xo=max(xl)
    if yo is None:  
        if yl is not None:
            yo = max(yl)
    else:
        if yo > max(yl):
            print("WARNING: yo cannot be smaller than yl. Adjusted to %i"%max(yl)) 
            xo=max(yl)
            
    p_init = models.Legendre2D(xo,yo,x_domain=x_domain,y_domain=y_domain,*args,**kwargs)
    if xl is not None and yl is not None or indices is not None: #has some fixed coefficients
        #pdb.set_trace()
        # First set all coefficients in .fixed to True.  
        for k in p_init.fixed.keys() : p_init.fixed[k]=True
        #The ones in xl,yl are set to False, so they are taken as independent variables.
        if indices is not None:
            ## Handle xl,yl
            ## set which variables are varied in fit
            assert (xl is None and yl is None)
            kl = [p_init.param_names[i] for i in indices]
        else:
            kl = make_legendre_keys(xl,yl)
        for k in kl:
            p_init.fixed[k] = False
        
    sh = np.shape(d)
    #pdb.set_trace()
    xg = np.linspace(-1,1,sh[1]) if x is None else x
    yg = np.linspace(-1,1,sh[0]) if y is None else y
    x,y = np.meshgrid(xg,yg)
                      
    index = ~np.isnan(d)
    p = fit_p(p_init,x[index],y[index],d[index])
    
    if model:
        return p(x,y),p
    else: 
        return p(x,y),p.parameters.reshape((yo+1,xo+1))
def RAlegendre2d(d,xo=2,yo=2,xl=None,yl=None):
    """
    Fit a set of 2d Legendre polynomials to a 2D array.
    The aperture is assumed to be +-1 over each dimension.
    NaNs are automatically excluded from the analysis.
    x0 = 2 fits up to quadratic in row axis
    y0 = 3 fits up to cubic in column axis
    If xl and yl are specified, only the polynomials with
    orders xl[i],yl[i] are fitted, this allows for fitting
    only specific polynomial orders.
    """
    #Define fitting algorithm
    fit_p = fitting.LinearLSQFitter()
    #Handle xl,yl
    if xl is not None and yl is not None:
        xo,yo = max(xl),max(yl)
        p_init = models.Legendre2D(xo,yo)
        #Set all variables to fixed
        for l in range(xo+1):
            for m in range(yo+1):
                key = 'c'+str(l)+'_'+str(m)
                p_init.fixed[key] = True
        #p_init.fixed = dict.fromkeys(p_init.fixed.iterkeys(),True)
        #Allow specific orders to vary
        for i in range(len(xl)):
            key = 'c'+str(xl[i])+'_'+str(yl[i])
            p_init.fixed[key] = False
    else:
        p_init = models.Legendre2D(xo,yo)
    
    sh = np.shape(d)
    x,y = np.meshgrid(np.linspace(-1,1,sh[1]),\
                      np.linspace(-1,1,sh[0]))
    index = ~np.isnan(d)
    p = fit_p(p_init,x[index],y[index],d[index])
    
    return p(x,y),p.parameters.reshape((yo+1,xo+1))
    
def fitPlane(d,x=None,y=None):
    """
    Fit a plane.
    Piston, tip, tilt are fit
    """
    return legendre2d(d,x,y,xl=[0,0,1],yl=[0,1,0])

def fitCylMisalign(d):
    """
    Fit cylindrical misalignment terms to an image
    Piston, tip, tilt, cylindrical sag, and astigmatism
    are fit
    """
    return legendre2d(d,xl=[0,0,1,2,1],yl=[0,1,0,0,1])

#def fitConic(d):
def fitConeMisalign(d):
    """
    Fit conical misalignment terms to an image
    Piston, tip, tilt, cylindrical sag, and astigmatism
    are fit.
    modified by VC from cylinder adding term 2,1
    """
    return legendre2d(d,xl=[0,0,1,2,1,2],yl=[0,1,0,0,1,1])

def fitLegendreDistortions(d,xo=2,yo=2,xl=None,yl=None):
    """
    Fit 2D Legendre's to a distortion map as read by 4D.
    If sum of orders is odd, the coefficient needs to be negated.
    """
    #Find and format coefficient arrays
    fit = legendre2d(d,xo=xo,yo=yo,xl=xl,yl=yl)
    # az,ax = np.meshgrid(range(xo+1),range(yo+1))
    az,ax = np.meshgrid(list(range(xo+1)),list(range(yo+1)))
    az = az.flatten()
    ax = ax.flatten()
    coeff = fit[1].flatten()

    #Perform negation
    coeff[(az+ax)%2==1] *= -1.
    
    return [coeff,ax,az]

def sgolay2d ( z, window_size, order, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')

def circle(x,y,xc,yc):
    """Fit a circle to a set of x,y coordinates
    Supply with a guess of circle center
    Returns [xc,yc],[rmsRad,rad]
    """
    fun = lambda p: circleMerit(x,y,p[0],p[1])[0]
    res = scipy.optimize.minimize(fun,np.array([xc,yc]),method='Nelder-Mead')
    return res['x'],circleMerit(x,y,res['x'][0],res['x'][1])

def circleMerit(x,y,xo,yo):
    rad = np.sqrt((x-xo)**2+(y-yo)**2)
    mrad = np.mean(rad)
    return np.sqrt(np.mean((rad-mrad)**2)),mrad
