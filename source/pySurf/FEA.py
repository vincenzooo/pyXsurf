from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import span
from pySurf.points import get_points, resample_grid,rebin_points,subtract_points,plot_points,save_points
import numpy as np
from matplotlib import pyplot as plt

def FEAreader1(FEAfile):
    """read V. Marquez files. Perform column shifting and knvert x returning result in operator coordinates
    with all units in mm.
    """
    a=np.genfromtxt(FEAfile,delimiter=',',skip_header=1)
    yy,xx,zz,dy,dx,dz=np.hsplit(a[:,2:8],6)
    x=-xx-dx
    y=yy+dy  #Correct for gravity
    z=dz  #here I keep only the deformation, and ignore nominal cylinder shape
    p=np.hstack([x,y,dz])*1000. #convert to mm, all coordinates
    return p

def plotFEA(FEAfile,FEAreader,datafile=None,outname=None,markers=None):
    """ read FEA and resample/rebin both points and FEA on the grid defined by steps, subtract gravity from pts 
    and plot and returns the corrected points. Simulation and data are also plotted if valid. 
    If datafile is not passed only simulation data are plotted and returned.
    FEA reader is a function that accept a single argument FEAfile and return an array of points describing 
    changes in shape (x,y,dz)."""
    rebin=False
    
    fpts=FEAreader(FEAfile)    
    if trans is not None:
        fpts=trans(fpts)
    if  datafile is not None:
        pts=get_points(datafile,delimiter=' ')
        
        if rebin:
            pts=rebin_points(pts,steps=steps)
            fp=rebin_points(fpts,steps=steps)
        else:
            xl,yl=span(pts,axis=0)[:2,:]
            xg=np.arange(xl[0],xl[1],steps[0])
            yg=np.arange(yl[0],yl[1],steps[1])
            pts=resample_grid(pts,xg,yg)
            fp=resample_grid(fpts,xg,yg)

        pts0=subtract_points(pts,fp)
    else:
        pts0=pts
        pts=None
    
    #diagnostic plots for gravity subtraction
    if pts is not None:
        plt.figure('pts0')  #corrected
        plt.clf()
        plot_points(pts0,scatter=1,aspect='equal')
        plt.title('points from file')
        if markers is not None:
            plt.plot(markers[:,0],markers[:,1],'+w',label='support')

    if datafile is not None:        
        plt.figure('pts')  #data
        plt.clf()
        plot_points(pts,scatter=1,aspect='equal')
        plt.title('resampled and gravity corrected')
        plt.savefig(fn_add_subfix(outname,'_corrected','.png'))
        if markers is not None:
            plt.plot(markers[:,0],markers[:,1],'+w',label='support')
        
    plt.figure('fp')  #simulation
    plt.clf()
    plot_points(fp,scatter=1,aspect='equal')
    plt.title('gravity effect from FEA')
    if markers is not None:
        plt.plot(markers[:,0],markers[:,1],'+',label='support') #here x and y inverted
    plt.savefig(fn_add_subfix(outname,'_FEA','.png'))

    if outname is not None:
        save_points(pts0,fn_add_subfix(outname,'','.dat'))
        save_points(pts0,fn_add_subfix(outname,'_matrix','.dat'),matrix=1)
        save_points(pts0,fn_add_subfix(outname,'_matrix','.fits'),matrix=1)

    return pts0