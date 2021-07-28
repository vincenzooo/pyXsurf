#This submodule facilitates stitching of metrology images
#The images must have fiducials to compute translation and rotation
#Piston/tip/tilt is done by arrays after translation/rotation is fixed
#by fiducials
import utilities.transformations as tr
import numpy as np
import matplotlib.pyplot as plt
from . import man
from scipy.optimize import minimize
from .analysis import getPoints
from scipy.interpolate import griddata
from utilities.plotting import nanmean

import pdb

def transformCoords(x,y,tx,ty,theta):
    """Transforms coordinates x,y by translating tx,ty
    and rotation theta about x
    Returns: x,y of new coords
    """
    trans = tr.translation_matrix([tx,ty,0])
    rot = tr.rotation_matrix(theta,[0,0,1],point=[np.mean(x),np.mean(y),0])
    pos0 = np.array((x,y,np.repeat(0.,np.size(x)),np.repeat(1.,np.size(x))))
    pos1 = np.dot(trans,np.dot(rot,pos0))

    return pos1[0],pos1[1]

def transformCoords_wMag(x,y,tx,ty,theta,mag):
    """Transforms coordinates x,y by magnifying x,y by a constant factor,
    then translating tx,ty and rotating theta about x
    Returns: x,y of new coords
    """
    mag_x,mag_y = x*mag,y*mag
    trans = tr.translation_matrix([tx,ty,0])
    rot = tr.rotation_matrix(theta,[0,0,1],point=[np.mean(x),np.mean(y),0])
    pos0 = np.array((mag_x,mag_y,np.repeat(0.,np.size(x)),np.repeat(1.,np.size(x))))
    pos1 = np.dot(trans,np.dot(rot,pos0))
    return pos1[0],pos1[1]

def transformCoords_wSeparateMag(x,y,tx,ty,theta,x_mag,y_mag):
    """Transforms coordinates x,y by magnifying x,y by a constant factor,
    then translating tx,ty and rotating theta about x
    Returns: x,y of new coords
    """
    new_x,new_y = x*x_mag,y*y_mag
    trans = tr.translation_matrix([tx,ty,0])
    rot = tr.rotation_matrix(theta,[0,0,1],point=[np.mean(x),np.mean(y),0])
    pos0 = np.array((new_x,new_y,np.repeat(0.,np.size(x)),np.repeat(1.,np.size(x))))
    pos1 = np.dot(trans,np.dot(rot,pos0))
    return pos1[0],pos1[1]

def matchFiducials(x1,y1,x2,y2):
    """This function will compute a rotation and translation
    to match a list of fiducial coordinates
    Returns: translation tx,ty and rotation theta about zhat
    to bring x2,y2 to x1,y1
    """
    #Make function to minimize
    fun = lambda p: sumOfSquares(x1,y1,*transformCoords(x2,y2,*p))

    #Make starting guess
    start = zeros(3)
    start[0] = mean(x1-x2)
    start[1] = mean(y1-y2)
    start[2] = .0001

    #Run minimization and return fiducial transformation
    res = minimize(fun,start,method='nelder-mead',\
                   options={'disp':True,'maxfev':1000})
    
    return res['x']

def matchFiducials_wMag(x1,y1,x2,y2):
    '''
    This function will compute a rotation, translation and
    magnification needed to match a list of fiducial coordinates.
    Returns:
    tx, ty - translations
    theta - rotation
    mag - magnification factor
    These transformations are needed to bring x2,y2 to x1,y1
    '''

    #Make function to minimize
    fun = lambda p: sumOfSquares(x1,y1,*transformCoords_wMag(x2,y2,*p))
    
    #Make starting guess
    start = np.zeros(4)
    start[0] = np.mean(x1-x2)
    start[1] = np.mean(y1-y2)
    start[2] = .0001
    start[3] = 1.0

    #Run minimization and return fiducial transformation
    res = minimize(fun,start,method='nelder-mead',\
                   options={'disp':True,'maxfev':1000})
    
    return res['x']

def matchFiducials_wSeparateMag(x1,y1,x2,y2):
    '''
    This function will compute a rotation, translation and
    magnification needed to match a list of fiducial coordinates.
    Returns:
    tx, ty - translations
    theta - rotation
    mag - magnification factor
    These transformations are needed to bring x2,y2 to x1,y1
    '''

    #Make function to minimize
    fun = lambda p: sumOfSquares(x1,y1,*transformCoords_wSeparateMag(x2,y2,*p))
    
    #Make starting guess
    start = np.zeros(5)
    start[0] = np.mean(x1-x2)
    start[1] = np.mean(y1-y2)
    start[2] = .0001
    start[3] = 1.0
    start[4] = 1.0

    #Run minimization and return fiducial transformation
    res = minimize(fun,start,method='nelder-mead',\
                   options={'disp':True,'maxfev':1000})
    
    return res['x']

def sumOfSquares(x1,y1,x2,y2):
    """Computes the sum of the squares of the residuals
    for two lists of coordinates
    """
    return sum(np.sqrt((x1-x2)**2+(y1-y2)**2))

def matchPistonTipTilt(img1,img2):
    """This function applies piston and tip/tilt
    to minimize RMS difference between two arrays
    Returns: img2 matched to img1
    """
    #Make function to minimize
    fun = lambda p: nanmean((img1 - man.tipTiltPiston(img2,*p))**2)

    #Run minimization and return matched image
    res = minimize(fun,[1.,.1,.1],method='nelder-mead',\
                   options={'disp':True,'maxfev':1000})

    return man.tipTiltPiston(img2,*res['x'])

def stitchImages(img1,img2):
    """Allows user to pick fiducials for both images.
    Function then computes the transform to move img2
    to img1 reference frame.
    Updated
    """
    #Get both fiducials
    xf1,yf1 = getPoints(img1)
    xf2,yf2 = getPoints(img2)

    #Match them
    tx,ty,theta = matchFiducials(xf1,yf1,xf2,yf2)

    #Pad img1 based on translations
    img1 = man.padNaN(img1,n=round(tx),axis=1)
    img1 = man.padNaN(img1,n=round(ty),axis=0)
    #Shift img1 fiducial locations
    if tx<0:
        xf1 = xf1 - tx
    if ty<0:
        yf1 = yf1 - ty

    #Get x,y,z points from stitched image
    x2,y2,z2 = man.unpackimage(img2,xlim=[0,shape(img2)[1]],\
                           ylim=[0,shape(img2)[0]])

    #Apply transformations to x,y coords
    x2,y2 = transformCoords(x2,y2,ty,tx,theta)

    #Get x,y,z points from reference image
    x1,y1,z1 = man.unpackimage(img1,remove=False,xlim=[0,shape(img1)[1]],\
                           ylim=[0,shape(img1)[0]])

    #Interpolate stitched image onto expanded image grid
    newimg = griddata((x2,y2),z2,(x1,y1),method='linear')
    print('Interpolation ok')
    newimg = newimg.reshape(shape(img1))

    #Images should now be in the same reference frame
    #Time to apply tip/tilt/piston to minimize RMS
    newimg = matchPistonTipTilt(img1,newimg)

    #Would like list of enlarge image showing all valid data, this is final step
    #Avoid overwritting fiducials
    #Save indices of NaNs near fiducials
    find = logical_and(sqrt((y1-xf1[0])**2+(x1-yf1[0])**2) < 15.,\
                       isnan(img1).flatten())
    for i in range(1,size(xf1)):
        find = logical_or(find,\
                logical_and(sqrt((y1-xf1[i])**2+(x1-yf1[i])**2) < 15.,\
                        isnan(img1).flatten()))

    #Include newimg data
    img1[isnan(img1)] = newimg[isnan(img1)]
    #Reset fiducials to NaNs
    img1[find.reshape(shape(img1))] = NaN

    #Return translations to automatically pad next image
    return img1,tx,ty

def overlapTrans(x,y,tx,ty,theta,sx,sy):
    """
    Transform coordinates with a rotation about mean coordinate,
    followed by translation, followed by scaling
    """
    x2,y2 = x*np.cos(theta)+y*np.sin(theta),\
            -x*np.sin(theta)+y*np.cos(theta)
    x2,y2 = x2+tx, y2+ty
    x2,y2 = x2*sx, y2*sy
    
    return x2,y2

def overlapMerit(x1,y1,z1,x2,y2,z2,tx,ty,theta,sx,sy):
    """
    Apply transformation on img2 and return RMS error
    """
    #Apply coordinate transformation
    x3,y3 = overlapTrans(x2,y2,tx,ty,theta,sx,sy)
    #Interpolate onto img1 grid
    x3,y3,z2 = x3.flatten(),y3.flatten(),z2.flatten()
    ind = ~np.isnan(z2)
    x3,y3,z2 = x3[ind],y3[ind],z2[ind]
    z4 = griddata((x3.flatten(),y3.flatten()),\
                  z2.flatten(),(x1,y1),method='cubic')
    #Determine overlap area
    area = np.sum(np.logical_and(~np.isnan(z1),~np.isnan(z4)))
    #Return error
    resid = z1-z4
    resid = resid - np.nanmean(resid)
    return np.sqrt(np.nanmean(resid**2)),z4

def overlapImages(img1,img2,scale=False):
    """Function to interpolate a second image onto the first.
    This is used to compare metrology carried out on a part
    before and after some processing step. The first image may
    be translated, rotated, and scaled with respect to the first
    image. Scaling is due to magnification changes.
    Procedure is to:
    1) Set NaNs to zeros so that they want to overlap
    2) Set up merit function as function of transformation
       and scaling. Return RMS error.
    3) Use optimizer to determine best transformation
    4) Interpolate img2 to img1 using this transformation
       and return transformed img2.
    """
    #Unpack images
    x1,y1 = np.meshgrid(np.linspace(-1.,1.,np.shape(img1)[1]),\
                        np.linspace(-1.,1.,np.shape(img1)[0]))
    x2,y2 = np.meshgrid(np.linspace(-1.,1.,np.shape(img2)[1]),\
                        np.linspace(-1.,1.,np.shape(img2)[0]))
    #Get centroids of each image
    cx1 = np.mean(x1[~np.isnan(img1)])
    cy1 = np.mean(y1[~np.isnan(img1)])
    cx2 = np.mean(x1[~np.isnan(img2)])
    cy2 = np.mean(y1[~np.isnan(img2)])

    #Save NaN index
    nans = np.isnan(img1)

    #Set up merit function
    if scale is False:
        fun = lambda p: overlapMerit(x1,y1,img1,x2,y2,img2,\
                                     p[0],p[1],p[2],1,1)[0]
        start = [cx1-cx2+.01,cy1-cy2+.01,0.1]
        print(start)
    else:
        fun = lambda p: overlapMerit(x1,y1,img1,x2,y2,img2,\
                                     *p)[0]
        start = [0.1,0.1,.1,1,1]

    #Optimize!
    pdb.set_trace()
    res = minimize(fun,start,method='Powell',\
                   options={'disp':True,'maxfev':1000,\
                            'ftol':.0001,\
                            'xtol':.0001})

    #Construct transformed img2
    imgnew = overlapMerit(x1,y1,img1,x2,y2,img2,\
                          res['x'][0],res['x'][1],res['x'][2],\
                          1,1)[1]
    img1[nans] = np.nan
    imgnew[nans] = np.nan

    pdb.set_trace()

    return imgnew

def AlignImagesWithFiducials(img1,img2,xf1,yf1,xf2,yf2):
    """
    Aligns img2 to img1 based on an array listing the x,y coordinates of common fiducials.
    Arguments:
    img1 - the reference image to be aligned to.
    img2 - the image to be aligned to the reference image.
    xf1 - an array containing the x coordinates of the fiducials in img1
    yf1 - an array containing the y coordinates of the fiducials in img1.
    xf2 - an array containing the x coordinates of the fiducials in img2.
    yf2 - an array containing the y coordinates of the fiducials in img2.
    Returns:
    newimg - img2 as aligned and interpolated to the coordinates of img1.
    """
    #Match them
    tx,ty,theta,mag = matchFiducials_wMag(yf1,xf1,yf2,xf2)

    x2_wNaNs,y2_wNaNs,z2_wNaNs = man.unpackimage(img2,remove = False,xlim=[0,np.shape(img2)[1]],\
                           ylim=[0,np.shape(img2)[0]])
    #Apply transformations to x,y coords
    x2_wNaNs,y2_wNaNs = transformCoords_wMag(x2_wNaNs,y2_wNaNs,ty,tx,theta,mag)
    
    #Get x,y,z points from reference image
    x1,y1,z1 = man.unpackimage(img1,remove=False,xlim=[0,np.shape(img1)[1]],\
                           ylim=[0,np.shape(img1)[0]])

    #Interpolate stitched image onto expanded image grid
    newimg = griddata((x2_wNaNs,y2_wNaNs),z2_wNaNs,(x1,y1),method='linear')
    print('Interpolation ok')
    newimg = newimg.reshape(np.shape(img1))

    #Images should now be in the same reference frame
    #Time to apply tip/tilt/piston to minimize RMS
    newimg = matchPistonTipTilt(img1,newimg)

    return newimg

def AlignImagesWithFiducials_SeparateMag(img1,img2,xf1,yf1,xf2,yf2):
    """
    Aligns img2 to img1 based on an array listing the x,y coordinates of common fiducials.
    Arguments:
    img1 - the reference image to be aligned to.
    img2 - the image to be aligned to the reference image.
    xf1 - an array containing the x coordinates of the fiducials in img1
    yf1 - an array containing the y coordinates of the fiducials in img1.
    xf2 - an array containing the x coordinates of the fiducials in img2.
    yf2 - an array containing the y coordinates of the fiducials in img2.
    Returns:
    newimg - img2 as aligned and interpolated to the coordinates of img1.
    """
    #Match them
    tx,ty,theta,x_mag,y_mag = matchFiducials_wSeparateMag(yf1,xf1,yf2,xf2)

    x2_wNaNs,y2_wNaNs,z2_wNaNs = man.unpackimage(img2,remove = False,xlim=[0,np.shape(img2)[1]],\
                           ylim=[0,np.shape(img2)[0]])
    #Apply transformations to x,y coords
    x2_wNaNs,y2_wNaNs = transformCoords_wSeparateMag(x2_wNaNs,y2_wNaNs,ty,tx,theta,x_mag,y_mag)
    
    #Get x,y,z points from reference image
    x1,y1,z1 = man.unpackimage(img1,remove=False,xlim=[0,np.shape(img1)[1]],\
                           ylim=[0,np.shape(img1)[0]])

    #Interpolate stitched image onto expanded image grid
    newimg = griddata((x2_wNaNs,y2_wNaNs),z2_wNaNs,(x1,y1),method='linear')
    print('Interpolation ok')
    newimg = newimg.reshape(np.shape(img1))

    #Images should now be in the same reference frame
    #Time to apply tip/tilt/piston to minimize RMS
    newimg = matchPistonTipTilt(img1,newimg)

    return newimg