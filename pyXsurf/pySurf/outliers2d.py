import matplotlib.pyplot as plt
import numpy as np
from dataIO.span import span
from pyProfile.profile import remove_profile_outliers
    
def remove_outliers2d(data,x=None,y=None,nsigma=3,fignum=None,name='',includenan=True):
    """remove outliers line by line by interpolation (along vertical lines).
    If fignum is set, plot comparison in corresponding figure."""

    ldata= data.copy()
    mask=np.apply_along_axis( remove_profile_outliers, axis=0, arr=ldata, 
        nsigma=nsigma ,includenan=includenan) #this modifies ldata and return a mask

    #these are used only to determine min and max.
    if x is None:
        x=np.arange(data.shape[1])
    if y is None:
        y=np.arange(data.shape[0])
              
    if fignum:
        #plot data before and after removal of outliers
        plt.figure(fignum)
        plt.clf()
        plt.suptitle('Effects of outliers removal - %s'%name).set_size('large')
        
        
        ax1=plt.subplot(221)
        plt.title('data')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        axim=plt.imshow(data,interpolation='None',aspect='auto',
                        clim=span(ldata),extent=np.hstack([span(x),span(y)]))
        plt.colorbar()

        ax2=plt.subplot(222,sharex=ax1,sharey=ax1)
        plt.title('data outliers removed')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        axim=plt.imshow(ldata,interpolation='None',aspect='auto',
                        clim=span(ldata),extent=np.hstack([span(x),span(y)]))
        plt.colorbar()

        ax3=plt.subplot(223,sharex=ax1,sharey=ax1)
        plt.title('outliers')
        outliers=ldata-data
        plt.imshow(outliers,interpolation='None',aspect='auto',
                        clim=span(ldata),extent=np.hstack([span(x),span(y)]))
        plt.colorbar()
        
        if mask.any():
            ax4=plt.subplot(224)
            plt.title('outliers height distribution')
            #plt.hist(outliers[mask].flatten(),bins=20,label='$\Delta$ correction',alpha=0.5)
            plt.hist(data[np.isfinite(data)].flatten(),bins=20,label='before',alpha=0.3,color='b')
            plt.hist(ldata[np.isfinite(ldata)].flatten(),bins=20,label='after',alpha=0.3,color='r')
            ax4b=ax4.twinx()
            ax4b.hist((data-ldata)[np.isfinite(data-ldata)].flatten(),bins=20,label='difference',alpha=0.3,color='r')
            plt.xlabel('Height (um)')
            plt.ylabel('N')
            plt.legend(loc=0)
        plt.show()
        
    return ldata

def remove_outliers2d2(data,x=None,y=None,nsigma=3,degree=1,fignum=None,name='',includenan=True):
    """remove outliers by comparing with average line profile."""
    #not working

    ldata= data.copy()
    avgprof=np.mean(ldata,axis=1)
    for i in ldata.T:
        d=(i-avgprof)
        d=d-polyfit_profile(d)  #d is the difference from average profile
        m=np.nanmean(d)
        rms=np.nanstd(d)
    mask=np.apply_along_axis( remove_profile_outliers, axis=0, arr=ldata, 
        nsigma=nsigma ,includenan=includenan) #this modifies ldata and return a mask

    #these are used only to determine min and max.
    if x is None:
        x=np.arange(data.shape[1])
    if y is None:
        y=np.arange(data.shape[0])
              
    if fignum:
        #plot data before and after removal of outliers
        plt.figure(fignum)
        plt.clf()
        plt.suptitle('Effects of outliers removal - %s'%name).set_size('large')
        
        
        ax1=plt.subplot(221)
        plt.title('data')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        axim=plt.imshow(data,interpolation='None',aspect='auto',
                        clim=span(ldata),extent=np.hstack([span(x),span(y)]))
        plt.colorbar()

        ax2=plt.subplot(222,sharex=ax1,sharey=ax1)
        plt.title('data outliers removed')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        axim=plt.imshow(ldata,interpolation='None',aspect='auto',
                        clim=span(ldata),extent=np.hstack([span(x),span(y)]))
        plt.colorbar()

        ax3=plt.subplot(223,sharex=ax1,sharey=ax1)
        plt.title('outliers')
        outliers=ldata-data
        plt.imshow(outliers,interpolation='None',aspect='auto',
                        clim=span(ldata),extent=np.hstack([span(x),span(y)]))
        plt.colorbar()
        
        if mask.any():
            ax4=plt.subplot(224)
            plt.title('outliers height distribution')
            #plt.hist(outliers[mask].flatten(),bins=20,label='$\Delta$ correction',alpha=0.5)
            plt.hist(data[np.isfinite(data)].flatten(),bins=20,label='before',alpha=0.3,color='b')
            plt.hist(ldata[np.isfinite(ldata)].flatten(),bins=20,label='after',alpha=0.3,color='r')
            ax4b=ax4.twinx()
            ax4b.hist((data-ldata)[np.isfinite(data-ldata)].flatten(),bins=20,label='difference',alpha=0.3,color='r')
            plt.xlabel('Height (um)')
            plt.ylabel('N')
            plt.legend(loc=0)
        plt.show()
        
    return ldata