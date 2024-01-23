#This submodule includes routines to manipulate image arrays
import numpy as np
from scipy.interpolate import griddata
from utilities.imaging.fitting import legendre2d


def unpackimage(data,xlim=[-1,1],ylim=[-1,1],remove=True):
    """Convert a 2D image into x,y,z coordinates.
    x will relate to 2nd index in order to correspond to abscissa in imshow
    y will relate to 1st index in order to correspond to oordinate in imshow
    if remove is True, NaNs will not be returned in the list of coordinates
    """
    x,y = np.meshgrid(np.linspace(xlim[0],xlim[1],np.shape(data)[1]),\
                   np.linspace(ylim[0],ylim[1],np.shape(data)[0]))
##    y = y.flatten()
##    x = x.flatten()
    
    if remove is True:
        ind = np.invert(np.isnan(data.flatten()))
        return x.flatten()[ind],y.flatten()[ind],data.flatten()[ind]

    return x.flatten(),y.flatten(),data.flatten()

def autoGrid(d,xr=[-1,1],yr=[-1,1]):
    """
    Create a meshgrid based on the shape of the image d
    """
    xspan = np.linspace(xr[0],xr[1],np.shape(d)[1])
    yspan = np.linspace(yr[0],yr[1],np.shape(d)[0])
    return np.meshgrid(xspan,yspan)
    

def shiftNaN(img,n=1,axis=0):
    """This function shifts an image in a NaN padded array
    Specify which axis to shift, and specify wh
    ich direction
    """
    #Construct array to insert
    if axis is 0:
        ins = np.repeat(np.nan,np.abs(n)*\
                     np.shape(img)[1]).reshape(np.abs(n),np.shape(img)[1])
    else:
        ins = np.repeat(np.nan,np.abs(n)*\
                     np.shape(img)[0]).reshape(np.abs(n),np.shape(img)[0])
    #If direction=0, shift to positive
    if n > 0:
        img = np.delete(img,np.arange(np.shape(img)[1]-\
                                      n,np.shape(img)[1]),axis=axis)
        img = np.insert(img,0,ins,axis=axis)
    else:
        n = np.abs(n)
        img = np.delete(img,np.arange(n),axis=axis)
        img = np.insert(img,-1,ins,axis=axis)
    return img

def padNaN(img,n=1,axis=0):
    """Pads an image with rows or columns of NaNs
    If n is positive, they are appended to the end of
    the specified axis. If n is negative, they are
    appended to the beginning
    """
    #Construct array to insert
    if axis is 0:
        ins = np.repeat(np.nan,np.abs(n)*\
                        np.shape(img)[1]).reshape(np.abs(n),np.shape(img)[1])
    else:
        ins = np.repeat(np.nan,np.abs(n)*\
                        np.shape(img)[0]).reshape(np.abs(n),np.shape(img)[0])
        ins = np.transpose(ins)
    #If direction=0, shift to positive
    if n < 0:
        img = np.concatenate((ins,img),axis=axis)
    else:
        img = np.concatenate((img,ins),axis=axis)
    return img

def padRect(img,nan_num = 1):
    """Pads an image with an outer NaN rectangle of width nan_num"""
    img = padNaN(img,n=nan_num,axis=0)
    img = padNaN(img,n=-nan_num,axis=0)
    img = padNaN(img,n=nan_num,axis=1)
    img = padNaN(img,n=-nan_num,axis=1)
    return img

def borderFill(img,n = 1,fill_value = np.NaN):
    img[:n],img[-n:] = fill_value,fill_value
    img[:,:n],img[:,-n:] = fill_value,fill_value
    return img

def tipTiltPiston(img,piston,tip,tilt,tx=None,ty=None):
    """This function adds a constant and
    tip and tilt to an array
    This makes use of tilt arrays tx,ty
    If not provided, compute using meshgrid
    Updated
    """
    if tx is None:
        ty,tx = np.meshgrid(np.arange(np.shape(img)[1]),\
                                np.arange(np.shape(img)[0]))
        tx = (tx-np.mean(tx)) / tx.max()
        ty = (ty-np.mean(ty)) / ty.max()

    return img + piston + tip*tx + tilt*ty

def nearestNaN(arr,method='nearest'):
    """Fill the NaNs in a 2D image array with the griddata
    nearest neighbor interpolation"""
    ishape = np.shape(arr)
    #Unpack image both with and without removing NaNs
    x0,y0,z0 = unpackimage(arr,remove=False)
    x1,y1,z1 = unpackimage(arr,remove=True)

    #Interpolate onto x0,y0 grid
    newarr = griddata((x1,y1),z1,(x0,y0),method=method)

    return newarr.reshape(ishape)

def rebin(a,shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return nanmean(nanmean(a.reshape(sh),axis=3),axis=1)

def stripnans(d1,removeAll=False):
    """
    Need to fix removeAll. Likely need to remove rows/columns
    in a circular fashion until all perimeter NaNs are eliminated.
    """
    d = np.copy(d1)
    if len(np.shape(d)) is 1:
        return d[~np.isnan(d)]
    if removeAll is False:
        newsize = np.shape(d)[1]
    else:
        newsize = 1
    while sum(np.isnan(d[0]))>=newsize:
        d = d[1:]
    while sum(np.isnan(d[-1]))>=newsize:
        d = d[:-1]
    if removeAll is False:
        newsize = np.shape(d)[0]
    else:
        newsize = 1
    while sum(np.isnan(d[:,0]))>=newsize:
        d = d[:,1:]
    while sum(np.isnan(d[:,-1]))>=newsize:
        d = d[:,:-1]
    return d

def transformation(x,y,r=0.,tx=0.,ty=0.):
    """Return x and y vectors after applying a rotation about
    the origin and then translations in x and y
    """
    x,y = np.cos(r)*x+np.sin(r)*y,-np.sin(r)*x+np.cos(r)*y
    x,y = x+tx,y+ty
    return x,y

def rotateImage(img,rot):
    """Apply a rotation about the center of an image using
    griddata
    """
    sh = np.shape(img)
    x,y = np.meshgrid(np.linspace(-1,1,sh[1]),np.linspace(-1,1,sh[0]))
    dx = 2./(sh[1]-1)
    dy = 2./(sh[0]-1)
    x,y = transformation(x,y,r=rot)
    x2,y2 = np.meshgrid(np.arange(x.min(),x.max()+dx,dx),\
                        np.arange(y.min(),y.max()+dy,dy))
    #Interpolate from x,y to x2,y2
    img2 = griddata((x.flatten(),y.flatten()),img.flatten(),(x2,y2))
    return stripnans(img2)

def newGridSize(img,newshape,method='linear'):
    """
    Interpolate an image onto a new shape size
    """
    shape1 = np.shape(img)
    x1,y1 = np.meshgrid(list(range(shape1[1])),list(range(shape1[0])))
    x2,y2 = np.meshgrid(np.linspace(0,shape1[1]-1,newshape[1]),\
                        np.linspace(0,shape1[0]-1,newshape[0]))
    img2 = griddata((x1.flatten(),y1.flatten()),img.flatten(),(x2,y2),\
                    method=method)
    return img2

def nanflatten(img):
    """
    Automatically remove NaNs when flattening an image
    """
    d = img.flatten()
    d = d[~np.isnan(d)]
    return d

def removePoly(l,order=2):
    """
    Remove a polynomial up to order from a slice.
    NaNs are ignored.
    """
    if np.sum(~np.isnan(l))<2:
        return
    ind = np.invert(np.isnan(l))
    x = np.arange(len(l))
    fit = np.polyfit(x[ind],l[ind],order)

    return l - np.polyval(fit,x)

def removeLegSlice(din,order=2,axis=0):
    """
    Remove a Legendre polynomial up to order from each
    slice of an image.
    """
    d = np.copy(din)
    if axis is 0:
        d = np.transpose(d)
    for i in range(len(d)):
        d[i,:] = removePoly(d[i,:],order=order)
    if axis is 0:
        d = np.transpose(d)
    
    return d

def remove2DLeg(din,xo=2,yo=0):
    """
    Remove a 2D Legendre fit to din up to
    xo and yo.
    """
    f = legendre2d(din,xo=xo,yo=yo)[0]
    
    return din-f

def removeDS9Regions(img,filename):
    """
    Read in an SAOImage region file and set all
    pixels within regions to NaN.
    File should look like:
    circle(x,y,rad)
    box(x,y,dx,dy,0)
    ellipse(x,y,dx,dy,0)
    """
    #Construct index grid
    y,x = np.meshgrid(list(range(np.shape(img)[1])),list(range(np.shape(img)[0])))
    
    #Get region data
    f = open(filename,'r')
    lines = f.readlines()

    for l in lines:
        t = l.split('(')[0]
        n = np.array(l.split('(')[1].split(','))
        # Note: this can throw an error based on Mac vs. Windows -- if there's a carriage return
        # plus a new line, the second value should be -3.
        n[-1] = n[-1][:-2]
        n = n.astype('float')
        if t == 'circle':
            ind = (x-n[1])**2+(y-n[0])**2 < n[2]**2
        elif t == 'box':
            yind = np.logical_and(y<n[0]+n[2]/2,y>n[0]-n[2]/2)
            xind = np.logical_and(x<n[1]+n[3]/2,x>n[1]-n[3]/2)
            ind = np.logical_and(xind,yind)
        elif t == 'ellipse':
            ind = []
        img[ind] = np.nan

        
    return img
