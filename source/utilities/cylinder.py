import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pdb
import utilities.imaging.man as man
import utilities.imaging.fitting as fit
import scipy.ndimage as nd
from utilities.plotting import scatter3d
import utilities.plotting
from utilities.imaging.analysis import rms
from scipy.interpolate import griddata

#This module contains routines to fit a cylinder to
#2D metrology data

def cyl(shape,curv,rad,yaw,pitch,roll,piston):
    """Create a cylindrical surface on a 2D array.
    Specify shape of array, and other parameters in
    pixels or radians where appropriate.
    Radius is assumed to be large enough to fill
    provided array.
    Curv is +1 or -1, with +1 indicating a convex
    cylinder from user's point of view (curving negative)"""
    #Construct coordinate grid
    x,y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
    x,y = x-np.mean(x),y-np.mean(y)
    #Apply yaw rotation
    x,y = x*np.cos(yaw)+y*np.sin(yaw),-x*np.sin(yaw)+y*np.cos(yaw)
    #Create base cylinder
    cyl = np.sqrt(rad**2-x**2)*curv
    #Apply pitch, roll, and piston
    cyl = cyl + piston + x*roll + y*pitch

    return cyl

def findGuess(d):
    """Find initial guess parameters for cylindrical
    metrology data. Use a quadratic fit in each axis
    to determine curvature sign and radius.
    Assume zero yaw"""
    sh = np.shape(d)
    yaw = 0.
    #Fit x and y slices
    xsl = d[sh[0]/2,:]
    ysl = d[:,sh[1]/2]
    xsl,ysl = xsl[np.invert(np.isnan(xsl))],ysl[np.invert(np.isnan(ysl))]
    Nx = np.size(xsl)
    Ny = np.size(ysl)
    xfit = np.polyfit(np.arange(Nx),xsl,1)
    yfit = np.polyfit(np.arange(Ny),ysl,1)
    #Subtract off linear fit
    xsl = xsl - np.polyval(xfit,np.arange(Nx))
    ysl = ysl - np.polyval(yfit,np.arange(Ny))
    #Figure out largest radius of curvature and sign
    Xsag = xsl.max()-xsl.min()
    Ysag = ysl.max()-ysl.min()
    if Xsag > Ysag:
        radius = Nx**2/8./Xsag
        curv = -np.sign(np.mean(np.diff(np.diff(xsl))))
    else:
        radius = Ny**2/8./Ysag
        curv = -np.sign(np.mean(np.diff(np.diff(ysl))))
        yaw = -np.pi/2
        
    return curv,radius,yaw,np.arctan(yfit[0]),\
                           np.arctan(xfit[0]),\
                           -radius+np.nanmean(d)

def fitCyl(d):
    """Fit a cylinder to the 2D data. NaNs are perfectly fine.
    Supply guess as [curv,rad,yaw,pitch,roll,piston]
    """
    guess = findGuess(d)
    fun = lambda p: np.nanmean((d-cyl(np.shape(d),*p))**2)
##    guess = guess[1:]
    pdb.set_trace()
    res = scipy.optimize.minimize(fun,guess,method='Powell',\
                    options={'disp':True,'ftol':1e-9,'xtol':1e-9})

    return res

def transformCyl(x,y,z,yaw,pitch,lateral,piston):
    """Transform x,y,z coordinates for cylindrical fitting"""
    #Yaw
    y,z = y*np.cos(yaw)+z*np.sin(yaw), -y*np.sin(yaw)+z*np.cos(yaw)
    #Pitch
    x,y = x*np.cos(pitch)+y*np.sin(pitch), -x*np.sin(pitch)+y*np.cos(pitch)
    #Lateral
    x = x + lateral
    #Piston
    z = z + piston
    return x,y,z

def itransformCyl(x,y,z,yaw,pitch,lateral,piston):
    """Transform x,y,z coordinates for cylindrical fitting"""
    #Lateral
    x = x + lateral
    #Piston
    z = z + piston
    #Pitch
    x,y = x*np.cos(-pitch)+y*np.sin(-pitch), -x*np.sin(-pitch)+y*np.cos(-pitch)
    #Yaw
    y,z = y*np.cos(-yaw)+z*np.sin(-yaw), -y*np.sin(-yaw)+z*np.cos(-yaw)

    return x,y,z

def meritFn(x,y,z):
    """Merit function for 3d cylindrical fitting"""
    rad = np.sqrt(x**2 + z**2)
    return rms(rad)

def fitCyl3D(d):
    """Fit a cylinder to the 2D data. NaNs are fine. Image is
    first unpacked into x,y,z point data. Use findGuess to
    determine initial guesses for cylindrical axis.
    """
    #Get cylindrical axis guess
    g = findGuess(d)
    #Convert to 3D points
    sh = np.shape(d)
    x,y,z = man.unpackimage(d,xlim=[-sh[1]/2.+.5,sh[1]/2.-.5],\
                            ylim=[-sh[0]/2.+.5,sh[0]/2.-.5])
    xf,yf,zf = man.unpackimage(d,xlim=[-sh[1]/2.+.5,sh[1]/2.-.5],\
                            ylim=[-sh[0]/2.+.5,sh[0]/2.-.5],remove=False)
    #Apply transformations to get points in about the right
    #area
    z = z - np.mean(z)
    x,y = x*np.cos(g[2])+y*np.sin(g[2]), -x*np.sin(g[2])+y*np.cos(g[2])
    y,z = y*np.cos(g[3])+z*np.sin(g[3]), -y*np.sin(g[3])+z*np.cos(g[3])
    x,z = x*np.cos(g[4])+z*np.sin(g[4]), -x*np.sin(g[4])+z*np.cos(g[4])
    z = z + g[0]*g[1]
    #Now apply fit to minimize sum of squares of radii
    fun = lambda p: meritFn(*transformCyl(x,y,z,*p))
    res = scipy.optimize.minimize(fun,[.01,.01,.01,.01],method='Powell',\
                    options={'disp':True})
    x,y,z = transformCyl(x,y,z,*res['x'])
    #Subtract mean cylinder
    rad = np.mean(np.sqrt(x**2+z**2))
    z = z - np.sqrt(rad**2-x**2)
    #Repack into 2D array
    yg,xg = np.meshgrid(np.arange(y.min(),y.max()+1),\
                        np.arange(x.min(),x.max()+1))
    newz = griddata(np.transpose([x,y]),z,\
                    np.transpose([xg,yg]),method='linear')
    pdb.set_trace()
##    xg,yg = np.meshgrid(np.linspace(-sh[1]/2.+.5,sh[1]/2.-.5,sh[1]),\
##                        np.linspace(-sh[0]/2.+.5,sh[0]/2.-.5,sh[0]))
##    pdb.set_trace()
##    newz = griddata(np.transpose([x,y]),z,\
##                    np.transpose([xg,yg]),method='linear')

    return newz
