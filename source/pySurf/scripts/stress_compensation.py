"""here coating stress fit and optimization functions from C1S05_stress_compensation_target.ipynb"""


import numpy as np 
import matplotlib.pyplot as plt
import os 
import pandas as pd
from plotting.captions import legendbox
from pySurf.data2D import get_stats
from dataIO.outliers import remove_outliers
from pySurf.data2D import level_data
from dataIO.span import span
from dataIO.fn_add_subfix import fn_add_subfix
from utilities.imaging import fitting as fit
from plotting.backends import maximize
from dataIO.any_is_none import any_is_none


def fom_rms(tdata,x=None,y=None):
    """Fit surface after plane level"""
    #tdata=tdata[ci0:ci1,ci0:ci1]
    #tdata=tdata-fit.legendre2d(tdata,1,1)[0]
    return np.nanstd(level_data(tdata)[0])

def fom_sl(tdata,x=None,y=None):
    """Fit slope of surface after plane level"""
    """This was taking a second argument dy, changed 2019/03/28 to data,x,y,
    do a workaround to keep old compatibility."""
    #tdata=tdata[ci0:ci1,ci0:ci1]
    
    if y is None and x is not None:
        dy=x
    else:    
        dy=np.diff(diff.y)[0]
    tdata=tdata-fit.legendre2d(tdata,1,1)[0]
    grad=np.gradient(tdata)
    slopeax=grad[0][:-1,:]/1000.*206265/dy  #in arcseconds from um
    return np.nanstd(slopeax)

def comp2(d1,d2,roi=None):
    #adapted (partially) to data2d
    """make 4 panel plot with data, differences and difference slope for two data sets on same x and y.
    Return four axis for plot customization.
    all data and all stats are leveled.
    
    roi: if passed, plot a corresponding rectangle."""
    partial=False  #not implemented, if a roi is defined, adds a legend with partial statistics
    if roi is not None:
        rect=np.array([(roi[0][0],roi[1][0]),
                       (roi[0][0],roi[1][1]),
                       (roi[0][1],roi[1][1]),
                       (roi[0][1],roi[1][0]),
                       (roi[0][0],roi[1][0])])   
    
    plt.figure(6)
    plt.clf()
    maximize()
    
    #pdb.set_trace()
    diff=d1-d2
    ax1=plt.subplot(141)
    d1.level().plot(title='Data')
    plt.clim(*remove_outliers(d1.level().data,nsigma=2,itmax=3,span=True))
    if roi is not None:
        plt.plot(rect[:,0], rect[:,1], '--',c='black', lw=2)
        legendbox(get_stats(d1.crop(*roi).level()()[0]),loc=4)
        
    ax2=plt.subplot(142,sharex=ax1,sharey=ax1)
    d2.level().plot(title='Simulation')
    
    plt.clim(*remove_outliers(d2.level().data,nsigma=2,itmax=3,span=True))
    if roi is not None:
        plt.plot(rect[:,0], rect[:,1], '--',c='black', lw=2)
        legendbox(get_stats(d2.crop(*roi).level()()[0]),loc=4)
        
    ax3=plt.subplot(143,sharex=ax1,sharey=ax1)
    diff.level().plot(title='Difference')    
    plt.clim(*remove_outliers(diff.level().data,nsigma=2,itmax=3,span=True))
    if roi is not None:
        plt.plot(rect[:,0], rect[:,1], '--',c='black', lw=2)
        legendbox(get_stats(diff.crop(*roi).level()()[0]),loc=4)
    
    ax4=plt.subplot(144,sharex=ax1,sharey=ax1)
    
    #2019/03/28 replaced with object
    #grad=np.gradient(diff.level().data)
    #dy=np.diff(diff.y)[0]
    #slopeax=grad[0]/1000.*206265/dy
    #x,y=diff.x,diff.y
    #plot_data(slopeax,x,y,units=['mm','mm','arcsec'],title='Axial Slope',stats=True) #,
              #vmin=-250,vmax=250)
    #plt.clim(*remove_outliers(slopeax,nsigma=2,itmax=3,span=True))
    #if roi is not None:
    #    plt.plot(rect[:,0], rect[:,1], '--',c='black', lw=2)
    #    legendbox(get_stats(*crop_data(slopeax,x,y,*roi),units=['mm','mm','arcsec']),loc=4)
    
    slopeax=diff.slope(scale=(1.,1.,1000.))[1]
    slopeax.name='Axial Slope'
    slopeax.plot()
    if roi is not None:
        plt.plot(rect[:,0], rect[:,1], '--',c='black', lw=2)
        legendbox(get_stats(slopeax.crop(*roi).level()()[0]),loc=4)   
    
    axes = [ax1,ax2,ax3,ax4]
    '''
    if roi is not None:  #update legend to include total and partial stats.
        for ax in axes:
            for art in ax.artists:
                if isinstance(art,matplotlib.legend.Legend):
                    p = art.get_patches()
                    t = art.get_texts()
                    newleg=Legend(ax,p,t,loc=2,handletextpad=0, handlelength=0)
                    ax.add_artist(newleg)
                    art.remove()
    '''
    return axes

def simple_regressor(d1,d2):
    """Use analytical for simple regression formula to determine best fit scale factor.
       Data are leveled before calculation of fom, but note that leveling can be altered by nan. 
       The function is defined on 2D data (not on Data2D objects), d1 and d2 must have same shape."""
    #pdb.set_trace()
    d2[np.isnan(d1)]=np.nan #this is to account in leveling for points that are nan in difference (and product, but non in d2)
    tbest=np.nansum(level_data(d1)[0]*level_data(d2)[0])/np.nansum(level_data(d2)[0]**2)
    return tbest
    
def scale_fit2(d1,d2,testval=None,fom=fom_rms,outfile='',roi=None, crop=False, dis= True):
    """d1 and d2 are Data2D objects. Returns value for scaling tbest of d2
    that minimizes fom(d1-d2*tbest). e.g. fom = rms of surface/slope.
    Plot rms as a function of scale factor (if testval are passed for raster scan) 
        and 4 panel plot of best result (use comp2) either cropped (if crop is True) 
        or full.
    
    if testval is set to None, best fit is determined by analytical formula (minimizing
        rms of surface difference). If tesval is array, all values are tested. 
    
    if `roi` is passed, fom is calculated on the roi only.
    `crop` flag controls the plotting. If True, only the roi is plotted and 
    color scale is adjusted accordingly. Otherwise, entire region is plotted and
    a rectangle around the roi is plotted.
    dis=True print information and display plots in notebook.
    
    all fom and stats are calculated on leveled data, ideally I want to have stats in 
    roi unleveled, but it is safer this way for now.
    """
    
    #dy=np.diff(d1.y)[0]
    d2=d2.resample(d1)
    
    res=[]
    if roi is None: 
        diff=d1-d2
        roi = [span(diff.x),span(diff.y)]  #stupid way of doing it
    
    if testval is None:
        #use analytical formula
        tbest=simple_regressor(d1.crop(*roi).data,d2.crop(*roi).data)
        dtest=(d1-(d2*tbest)).crop(*roi).level()
        rbest=fom(*dtest())
    else:
        #test all values
        for i,t in enumerate(testval):  
            #test each testval, create a plot (on file only). If dis is True, print vals
            diff=d1-d2*t 
            f=fom(*(diff.crop(*roi))())
            res.append(f)
            axes=comp2(d1,d2*t,roi=roi)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle('factor: %s'%t)
            if outfile:
                plt.savefig(fn_add_subfix(outfile,'_%03i'%(i+1),'.jpg'))
            if dis: print (i,t,f)        
        #select best value
        res=np.array(res)
        ibest=np.argmin(res)   
        tbest=testval[np.argmin(res)]
        rbest=res[ibest]
        #make plot with rms as a function of testval
        plt.figure(5)
        plt.clf()
        plt.plot(testval,res)
        plt.title('best factor:%6.3f, FOM surf:%6.3f'%(tbest,res[ibest]))
        plt.grid()
        if outfile:
            plt.savefig((fn_add_subfix(outfile,'','.jpg',pre='FOMtrend_')))
        if dis: display(plt.gcf())
    
    #plot and display (if dis) best fit data (cropped if crop is selected, full otherwise).
    if crop:  
        d1 = d1.crop(*roi).level()
        d2 = d2.crop(*roi).level()
    plt.figure()
    a=comp2(d1,d2*tbest,roi=roi)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('best factor:%6.3f, FOM surf:%6.3f'%(tbest,rbest))
    if dis: display(plt.gcf())
    if outfile: plt.savefig(fn_add_subfix(outfile,'_fit','.jpg'))
    
    return tbest



def trioptimize(d1,d2,fom,testvals=None,rois=None,outfile=None,noplot=False,dis=True): 
    """run optimizations using scale_fit2 over a set of cases
    defined by the values in matching lists of testvals (array of scaling test values) 
    and rois ([[xmin,xmax],[ymin,ymax]]). 
    
    Returns a pandas data structure with parameters and best scaling values for each case,
        with fields ['tbest','testval','roi','fom_roi'].
    
    Plot 
 
    d1 and d2 are Data2D objects.
    testvals: list of vectors to use as test values for each of the ROI
    rois:    list of rois (same len as testvals), in form [[xmin,xmax],[ymin,ymax]], 
        set to None to use for range.
    outfile:
    noplot:
    dis:
    """
    
    if outfile is not None:
        outname = os.path.basename (outfile)
        outfolder = os.path.dirname (outfile)
    else:
        outname = None
    
    plt.close('all')
    res=pd.DataFrame(index=['tbest','testval','roi','fom_roi','fom_initial']).transpose()

    if any_is_none(testvals): 
        if fom != fom_rms:
            raise ValueError('analytical fit (without testval) supported only for fom_rms')
        if testvals is None: testvals=[None]*len(rois)
    
    #Minimizes on each subaperture
    for i,(roi,testval) in enumerate(zip(rois,testvals)):  
        name = 'crop%02i'%(i+1) if outname is None else outname+'_crop%02i'%(i+1)
        #if dis print optimization parameters
        if dis:
            print ("\n\n\n===   CASE %i: %s  ===\n"%(i+1,name))
            if roi is None:
                print ('FULL APERTURE')
            else:
                print ("CROP: [%.2f:%.2f] [%.2f:%.2f]"%tuple(roi[0]+roi[1]))
            print ("TEST RANGE:",span(testval))
            dtest=(d1-d2).crop(*roi) if roi is not None else d1-d2
            fom0=fom(*dtest())
            print ("INITIAL FOM:",fom0)
        
        #do optimization, plot bestfit on ROI, displaying plot if dis is True
        s=scale_fit2(d1,d2,testval, dis=dis if roi is not None else False,  #it will plot later if no roi 
                  fom=fom,outfile='' if outname is None else fn_add_subfix (outfile,
                  '_crop%02i'%(i+1)),roi=roi,crop=dis) #if dis is set, plot roi, otherwise skip

        #calculate and store best fit
        ftest =fom(*((d1-d2*s).crop(*roi))()) if roi is not None else fom(*(d1-d2*s)())
        b=pd.Series([s,testval,roi,ftest,fom0],index=['tbest','testval','roi','fom_roi','fom_initial'], #fom_initial added 04/02
            name=name)
        res = res.append(b)
        
        #plot best fit case full range unless noplot
        if not noplot:
            plt.close('all')
            comp2(d1,d2*s,roi=roi) # plots entire area with rectangle
            dtest=(d1-d2*s)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle('best factor:%6.3f, FOM surf:%6.3f'%(s,fom(*dtest())))
            if outname:
                plt.savefig(fn_add_subfix(outfile,'_croproi_%02i'%(i+1),'.png'))
            if dis: display(plt.gcf())
        if dis: print ("BEST SCALE: %.3f"%s," FOM:",ftest)
        
    if dis: print ("\n===\nSUMMARY:")
    #a=pd.DataFrame(index=['tbest','testval','roi','fom_roi']).transpose()
    for roi,r in zip(rois,res.iterrows()):
        if dis:
            if roi is None:
                print ('FULL APERTURE--> BEST SCALE: ', r[1]['tbest'])
            else:
                #pdb.set_trace()
                #not sure why this is needed, but r[0] is the dataseries name, r[1] the dataseries
                print ("CROP: [%.2f:%.2f] [%.2f:%.2f]"%tuple(roi[0]+roi[1]),"--> BEST SCALE: ",r[1]['tbest'])
    print ("=======================")
    
    return res
    
def reoptimize(scales, step=None, nstep=5):
    """run a numerical optimization starting from a centered value and a number or steps and stepsize.
    Optimization parameters are taken from a pandas dataFrame that has fields 
        It can be used to numerically rerun an analytical fit (or to refine a previous fit).
        2019/03/28 quickly taken from notebook. """
    
    #initialize allres
    #step=0.001
    #nsteps=5 

    print(scales['tbest','fom_roi','roi'])
    tv=scales['tbest'].values
    rois=[None if np.isnan(s).all() else s for s in scales['roi'].values] #pd makes them nan when they were None, convert back

    scales2=trioptimize(sdiff,df,fom=fom,
               testvals=[np.arange(x0,y0,step) for x0,y0 in zip(tv-step*5,tv+step*5)],
               rois=rois,dis =True,
               outfile=os.path.join(outfolder,prfolder,'VM_stress_fit'))
    print(scales2['tbest','fom_roi','roi'])
    return scales2