"""Artificially generated test surfaces.
Can be incorporated in readers."""

import numpy as np
import matplotlib.pyplot as plt
from dataIO.span import span

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)



def make_sag(nx,ny):
    """ create surface with sag along y with peak-to-valley 1.
    Note that this is the range of data, not of the surface (i.e. if 
    number of points in y is even, the analytical minimum of the surface lies
    between the two central pixels and it is not included in data)"""
    
    ss=(np.arange(ny)-(ny-1)/2.)**2 #makes a parabola centered on 0
    ss=ss/span(ss,size=1)-np.min(ss) #rescale height to 1
    sag=np.repeat(ss[np.newaxis, :], nx, axis=0).T #np.tile(ss,nx).reshape(ny,nx)
    
    return sag,np.arange(nx),np.arange(ny)


    
def make_range(nx,ny):
    """make a data,x,y with increasing integer values for each pixel.
    
    example:

    >>> make_range(2,3)
    
    (array([[0., 1.],
        [2., 3.],
        [4., 5.]]), array([0, 1]), array([0, 1, 2]))"""
        
    data,x,y=np.arange(ny*nx,dtype=float).reshape(ny,nx),np.arange(nx),np.arange(ny)
    x=np.arange(nx)
    y=np.arange(ny)
    return data,x,y
    
def make_prof_legendre(x,coeff,inanp=[]):      
    """create a profile on x with legendre coefficients coeff.
    Add nans on inanp indices."""
    yp=np.polynomial.legendre.legval(x,coeff)
    yp[inanp]=np.nan
    return yp

def make_surf_legendre(x,y,coeff,inanp=[],inanl=[]):
    """Create a test surface from profiles along x, created with
    legendre polynomial with coefficients 'coeff'. 
    """
       
    data=np.empty((len(y),len(x)))
    nrep=len(x)//len(coeff)
    for i,cc in enumerate(coeff):
        yp=make_prof_legendre(y,cc,)
        data[:,i*nrep:(i+1)*nrep]=yp[:,None]
    data[inanp,inanp]=np.nan
    data[:,inanl]=np.nan

    return data

def make_surf_plane(x,y,coeff):
    """Create a test surface from plane coefficients `coeff` as returned from `pySurf.plane_fit`.
    based on
        # Ax + By + C = z    
    """
    
    A,B,C = coeff
    xx,yy = np.meshgrid(x,y)
    z = A * xx + B * yy + C
    data = z.reshape(len(y),len(x))

    return data
    
    
def test_profile_legendre(nanpoints=0):
    """test my routines with profile
    """
    
    x=np.arange(12)
    y=np.arange(20)
    coeff=[[0,3,-2],[12,0.01,-0.5],[-1,0.5,0]]
    inanp=[4,7,10] # indices for point nans if nanpoints != 0
    inanl=[2,5]     #indices for nan lines 

    plt.figure("fit and reconstruction")
    plt.clf()
    for cc in coeff:
        yp=np.polynomial.legendre.legval(y,cc)
        if nanpoints>0:
            yp[inanp]=np.nan
        plt.plot(y,yp,label='build %5.4f %5.4f %5.4f'%tuple(cc))
        yrec=fitlegendre(y,yp,2)
        plt.plot(y,yrec,'x')
    plt.legend(loc=0)

def test_profile_legendre(nans=True,fixnans=True):
    """test on a list of 2D coefficients creating profile then fitting with 
        numpy routines and with data2D routine fitlegendre 
        (wrapper around numpy to handle nans)."""
    
    y=np.arange(20)
    coeff=[[0,3,-2],[12,0.01,-0.5],[-1,0.5,0]]
    if nans:
        inanp=[4,7,10] # indices for point nans if nanpoints != 0
    else:
        inanp=[]
        
    plt.figure("fit and reconstruction")
    plt.clf()
    testdata=[y]
    for cc in coeff:
        yp=make_prof_legendre(y,cc,inanp)
        testdata=testdata+[yp]
        plt.plot(y,yp,label='build %5.4f %5.4f %5.4f'%tuple(cc))
        ccfit=np.polynomial.legendre.legfit(y,yp,2)
        print('fit:%5.4f %5.4f %5.4f'%tuple(ccfit))
        yrec=np.polynomial.legendre.legval(y,ccfit)
        yrec2=fitlegendre(y,yp,2,fixnans=fixnans)
        plt.plot(y,yrec,'x',label='rec  %5.4f %5.4f %5.4f'%tuple(ccfit))
        plt.plot(y,yrec2,'o',label='data2D.fitlegendre')
    plt.legend(loc=0)
    
    return testdata
    
def test_surf_legendre(nans=True,fixnans=True):
    """test how 1D (line) routines in polynomial.legendre work on 1D and 2D data.
    If nanpoints=0 nan are not put in data. If nanpoints=1 or 2 nans are added on some points and
    some lines, with value value of nanpoints determineing the option nanstrict of levellegendre
    (1=false, 2=true).
    """
    x=np.arange(12)
    y=np.arange(20)
    coeff=[[0,3,-2],[12,0.01,-0.5],[-1,0.5,0]]
    if nans:
        inanp=[4,7,10] # indices for point nans if nanpoints != 0
        inanl=[2,5]     #indices for nan lines    x=np.arange(12)
    else:
        inanp,inanl=[]
    
    data=make_surf_legendre(x,y,coeff,inanp,inanl)
    
    #test on surface
    ccfit=np.polynomial.legendre.legfit(y,data,2)
    print("shape of data array: ",data.shape)
    print("shape of coefficient array:",ccfit.shape)
    #print 'fit:%5.4f %5.4f %5.4f'%tuple(ccfit)
    #xx,yy=np.meshgrid(x,y)
    
    datarec=np.polynomial.legendre.legval(y,ccfit).T
    #datarec=np.polynomial.legendre.legval2d(yy,xx,ccfit)

    plt.figure("2D fit and reconstruction with numpy")
    plt.clf()    
    plt.subplot(131)
    plt.title('data')
    plt.imshow(data,origin='lower',interpolation='none')
    plt.colorbar()
    plt.subplot(132)
    plt.title('reconstruction')
    plt.imshow(datarec,origin='lower',interpolation='none')
    plt.colorbar()
    plt.subplot(133)
    plt.title('difference')
    plt.imshow(data-datarec,origin='lower',interpolation='none')
    plt.colorbar()
    plt.tight_layout()
    
    datarec=fitlegendre(y,data,2,fixnans=fixnans)
    plt.figure("2D fit and reconstruction with fitlegendre")
    plt.clf()    
    plt.subplot(131)
    plt.title('data')
    plt.imshow(data,origin='lower',interpolation='none')
    plt.colorbar()
    plt.subplot(132)
    plt.title('reconstruction')
    plt.imshow(datarec,origin='lower',interpolation='none')
    plt.colorbar()
    plt.subplot(133)
    plt.title('difference')
    plt.imshow(data-datarec,origin='lower',interpolation='none')
    plt.colorbar()
    plt.tight_layout()
    
    #test my routine now
    plt.figure("legendre removal with levellegendre")
    plt.imshow(levellegendre(y,data,2),origin='lower',interpolation='none')
    plt.colorbar()
    plt.show()
    
    #compare with old remove legendre
    plt.figure("compare removal routines")
    plt.clf()
    
    data=data+np.random.random(data.shape)
    datarec=levellegendre(y,data,2)
    rem4=removelegendre(y,2)
    rldata=level_by_line(data,rem4)
    
    plt.subplot(131)
    plt.imshow(datarec,origin='lower',interpolation='none')
    plt.title('levellegendre')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(rldata,origin='lower',interpolation='none')
    plt.title('level_by_line')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(datarec-rldata,origin='lower',interpolation='none')
    plt.title('difference')
    plt.colorbar()
    plt.tight_layout()

def test_makeGaussian(N=100,rnd=3.):
    a=makeGaussian(N)*20
    a=a+np.random.random((N,N))*rnd
    fig=plt.figure()
    fig.canvas.set_window_title('Gaussian surface N=%i, noise=%f'%(N,rnd))
    title="min:%4.1f max:%4.1f rms:%4.3f"%(np.nanmin(a),np.nanmax(a),np.nanstd(a))
    plt.title(title)
    plt.imshow(a,interpolation='none')
    plt.colorbar()
    plt.show()
    return a
    
if __name__=="__main__":
    N=100
    data=test_makeGaussian(100)
    print(data)

