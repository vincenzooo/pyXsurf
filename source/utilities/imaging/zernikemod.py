import numpy as np
from math import factorial
from numpy.ma import masked_array
import scipy
from scipy.misc import factorial
import pdb

#Define Zernike radial polynomials
def rnm(n,m,rho):
    """
    Return an array with the zernike Rnm polynomial calculated at rho points.
    
    
    **ARGUMENTS:**
    
        === ==========================================
        n    n order of the Zernike polynomial
        m    m order of the Zernike polynomial
        rho  Matrix containing the radial coordinates. 
        === ==========================================       
    
    .. note:: For rho>1 the returned value is 0
    
    .. note:: Values for rho<0 are silently returned as rho=0
    
    """
    
    if(type(n) is not int):
        raise Exception("n must be integer")
    if(type(m) is not int):
        raise Exception("m must be integer")
    if (n-m)%2!=0:
        raise Exception("n-m must be even")
    if abs(m)>n:
        raise Exception("The following must be true |m|<=n")
    mask=np.where(rho<=1,False,True)
    
    if(n==0 and m==0):
        return  masked_array(data=np.ones(np.shape(rho)), mask=mask)
    rho=np.where(rho<0,0,rho)
    Rnm=np.zeros(rho.shape)
    S=(n-abs(m))/2
    for s in range (0,S+1):
        CR=pow(-1,s)*factorial(n-s)/ \
            (factorial(s)*factorial(-s+(n+abs(m))/2)* \
            factorial(-s+(n-abs(m))/2))
        p=CR*pow(rho,n-2*s)
        Rnm=Rnm+p
    return masked_array(data=Rnm, mask=mask)
    
def zernike(n,m,rho,theta):
    """
    Returns the an array with the Zernike polynomial evaluated in the rho and 
    theta.
    
    **ARGUMENTS:** 
    
    ===== ==========================================     
    n     n order of the Zernike polynomial
    m     m order of the Zernike polynomial
    rho   Matrix containing the radial coordinates. 
    theta Matrix containing the angular coordinates.
    ===== ==========================================
 
    .. note:: For rho>1 the returned value is 0
    
    .. note:: Values for rho<0 are silently returned as rho=0
    """
    
    
    Rnm=rnm(n,m,rho)
    
    NC=np.sqrt(2*(n+1))
    S=(n-abs(m))/2
    
    if m>0:
        Zmn=NC*Rnm*np.cos(m*theta)
    #las funciones cos() y sin() de scipy tienen problemas cuando la grilla
    # tiene dimension cero
    
    elif m<0:
        Zmn=NC*Rnm*np.sin(m*theta)
    else:
        Zmn=np.sqrt(0.5)*NC*Rnm
    return Zmn

def zmodes(N):
    """
    Construct Zernike mode vectors in standard ordering
    Includes all modes up to radial order N
    """
    r = 0 #Starting radial mode
    radial = []
    azimuthal = []
    z = []
    while np.size(radial) < N:
        if r % 2 == 0:
            m = 0 #Set starting azimuthal mode to 0
        else:
            m = 1 #Set starting azimuthal mode to 1
        while m <= r and np.size(radial) < N:
            #Get current z number
            z = np.size(azimuthal) + 1
            #If z is odd, append sine first
            #Append negative and positive m
            if z % 2 == 1:
                azimuthal.append(-m)
            else:
                azimuthal.append(m)
            radial.append(r)
            if m > 0:
                if z % 2 == 1:
                    azimuthal.append(m)
                else:
                    azimuthal.append(-m)
                radial.append(r)
            m = m + 2
        r = r + 1 #Increment radial order
    return np.array(radial[:N],order='F').astype('int'),\
           np.array(azimuthal[:N],order='F').astype('int')

def zmatrix(rho,theta,N,r=None,m=None):
    """
    Formulate Zernike least squares fitting matrix
    Requires rho and theta vectors, normalized to rhomax=1
    """
    #Create mode vectors
    if r is None:
        r,m = zmodes(N)

    #Form matrix
    A = np.zeros((np.size(rho),np.size(r)))

    #Populate matrix columns
    for i in range(np.size(r)):
        A[:,i] = zernike(int(r[i]),int(m[i]),rho,theta)

    return A

def carttopolar(x,y,cx,cy,rad):
    #Convert to polar
    r = (sqrt((x-cx)**2+(y-cy)**2))/rad
    #Remove invalid points
    mask = r <= 1
    r = r[mask]
    x = x[mask]
    y = y[mask]
    theta = arctan2((y-cy),(x-cx))

    return r, theta, mask

#Reconstruct surface on unique x and y vectors for zernike coefficients
def zernsurf(x,y,cx,cy,rad,coeff,r=None,m=None):
    if size(unique(x)) == size(x):
        x,y = meshgrid(x,y)
    rho = sqrt((x-cx)**2+(y-cy)**2)/rad
    theta = arctan2((y-cy),(x-cx))
    heights = zeros(shape(x))

    if r is None:
        r,m = zmodes(size(coeff))

    for i in range(size(r)):
        heights = heights + coeff[i]*zernike(int(r[i]),int(m[i]),rho,theta)

    #Set outside pixels to NaN
    heights[where(rho>1.)] = NaN

    return heights.data

def fitimg(img,N=20,r=None,m=None):
    """
    Perform Zernike fit on an image.
    Zernike domain is defined over the full image array.
    """
    #Construct rho and theta vectors
    x,y = np.meshgrid(np.linspace(-1.,1.,np.shape(img)[0]),\
                      np.linspace(-1.,1.,np.shape(img)[1]))
    rho = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)

    #Flatten and remove NaNs
    rho = rho.flatten()
    theta = theta.flatten()

    #Get b vector
    b = img.flatten()

    #Get A matrix
    A = zmatrix(rho[~np.isnan(b)],theta[~np.isnan(b)],N,r=r,m=m)

    #Solve for coefficients
    c = scipy.linalg.lstsq(A,b[~np.isnan(b)])

    #Reconstruct fit image
    coeff = c[0]
    A = zmatrix(rho,theta,N,r=r,m=m)
    fit = np.dot(A,coeff)
    fit = fit.reshape(np.shape(x))
    fit[np.isnan(img)] = np.nan

    return c,fit.reshape(np.shape(x))

def fitvec(x,y,z,N=20,r=None,m=None):
    """
    Perform Zernike fit on an image.
    Zernike domain is defined over the full image array.
    """
    #Construct rho and theta vectors
    rho = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    rho = rho/rho.max()

    #Get A matrix
    A = zmatrix(rho,theta,N,r=r,m=m)

    #Solve for coefficients
    c = scipy.linalg.lstsq(A,z)

    return c
    
    

#Function to output Zernike coefficients and RMS fit error for x,y,z txt file
def zcoeff(filename,save=False,cx=0.,cy=0.,rad=1.,order=20,r=None,m=None,**kwags):
    #Read in surface data
    if type(filename) is str:
        d = genfromtxt(filename,**kwags)
    else:
        d = filename

    if shape(d)[0]==3:
        sagx, sagy, sagz = d[0],d[1],d[2]
    else:
##        #Strip NaN rows/columns
##        while sum(isnan(d[0]))==shape(d)[1]:
##            d = d[1:]
##        while sum(isnan(d[-1]))==shape(d)[1]:
##            d = d[:-1]
##        newsize = shape(d)[0]
##        while sum(isnan(d[:,0]))==newsize:
##            d = d[:,1:]
##        while sum(isnan(d[:,-1]))==newsize:
##            d = d[:,:-1]
        x,y=meshgrid(arange(shape(d)[0],dtype='float'),\
                     arange(shape(d)[1],dtype='float'))
##        ind = invert(isnan(d))
##        d2 = d[ind]
##        x = x[ind]
##        y = y[ind]
        x = x.flatten()
        y = y.flatten()
        d2 = d.flatten()
        sagx = []
        sagy = []
        sagz = []
        for i in range(size(d2)):
            if invert(isnan(d2[i])):
                sagx.append(x[i])
                sagy.append(y[i])
                sagz.append(d2[i])
        sagx = array(sagx)
        sagy = array(sagy)
        sagz = array(sagz)

    #Convert to normalized polar coordinates for Zernike fitting
    rho, theta, mask = carttopolar(sagx,sagy,cx,cy,rad)

    #Create Zernike polynomial matrix for least squares fit
    #Using all Zernike polynomials up to radial order 20
    pdb.set_trace()
    A = zmatrix(rho,theta,order,r=r,m=m)

    #Perform least squares fit
    #0th element is the coefficient matrix
    #1st element is sum of squared residuals
    #2nd element is rank of matrix A
    #3rd element is singular values of A
    fit = scipy.linalg.lstsq(A,sagz[mask])

    #Compute fitted surface from Zernike coefficients
    if shape(d)[0] != 3:
        y,x=meshgrid(arange(shape(d)[0],dtype='float'),\
                         arange(shape(d)[1],dtype='float'))
    else:
        x,y = d[0],d[1]
    fitsurf = zernsurf(x.flatten(),y.flatten(),cx,cy,rad,fit[0],r=r,m=m)
##
##    #Do residuals match up with those from fit?
##    print sum((fitsurf-sagz)**2)
##    print fit[1]

    rms = sqrt(fit[1]/size(sagz))

    #If save=True, save coefficients to a txt file
    #First line is number of coefficients
    if save==True:
        savetxt(filename.split('.')[0]+'Coeff.txt'\
                ,insert(fit[0],0,size(fit[0])))

    return fit[0],fit[1],rms,fitsurf
