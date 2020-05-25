

"""functions operating on a list of Data2D objects"""
# turned into a class derived from list 2020/01/16, no changes to interface,
# everything should work with no changes.

import os
import matplotlib.pyplot as plt
import numpy as np
import pdb


from pySurf.readers._instrument_reader import auto_reader
from pySurf.data2D import plot_data,get_data, level_data, save_data, rotate_data, remove_nan_frame, resample_data
from pySurf.data2D import read_data,sum_data, subtract_data, projection, crop_data, transpose_data, apply_transform, register_data
from plotting.multiplots import find_grid_size, compare_images
from pySurf.psd2d import psd2d,plot_psd2d,psd2d_analysis,plot_rms_power,rms_power

from pySurf.points import matrix_to_points2

from copy import deepcopy
from dataIO.span import span
from pySurf.data2D import projection
from pySurf.data2D_class import Data2D
from pySurf.affine2D import find_rototrans,find_affine

class superlist(list):
    """Test class that vectorizes methods."""
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = [obj.__getattr__(self, name) for obj in self] #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        """
        if hasattr(attr[0], '__call__'):
            attr=attr[0]
            def newfunc(*args, **kwargs):
                #pdb.set_trace()
                print('before calling %s' %attr.__name__)
                result = attr(*args, **kwargs)
                print('done calling %s' %attr.__name__)
                return result
            return newfunc
        else:
            return [a for a in attr]
        """
        
        result = []
        for a in attr:
            #print('loop ',a)
            #pdb.set_trace()
            if hasattr(a, '__call__'):
                result=np.vectorize(a)
                """
                def newfunc(*args, **kwargs):
                    #pdb.set_trace()
                    print('before calling %s' %a.__name__)
                    result = a(*args, **kwargs)
                    print('done calling %s' %a.__name__)
                    return result
                result.append(newfunc)
                """
            else:
                result.append(a)

        return result  #it works as a property. As method, this returns a list of methods, but since the method is called as sl.method(),
        #gives an error as it tries to call the list.
        # deve ritornare una funzione che ritorna una lista.
        
def test_superlist():
    s = superlist([np.arange(4),np.arange(3)])
    print('original data:')
    print(s)
    print('\ntest property (np.shape):')
    print(s.shape)
    print('\ntest method (np.flatten):')
    print(s.flatten())

class Dlist(list):
    """A list of pySurf.Data2D objects on which unknown operations are performed serially."""

    #automatically vectorize all unknown properties
#    def __getattr__(self, name):
#        return [getattr(i,name) for i in self]
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = object.__getattr__(self, name) #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        if hasattr(attr[0], '__call__'):
            def newfunc(*args, **kwargs):
                pdb.set_trace()
                print('before calling %s' %attr.__name__)
                result = attr(*args, **kwargs)
                print('done calling %s' %attr.__name__)
                return result
            return newfunc
        else:
            return attr
        
    def topoints(self):
        """convert a dlist to single set of points containing all data."""
        plist = [d.level((2,2)).topoints() for d in data]    
        return np.vstack(plist)   
    
def topoints(data,level=None):
    """convert a dlist to single set of points containing all data.
    if level is passed, points are leveled and the value is passed as argument (e.g. level=(2,2) levels sag along two axis)."""
    if level is not None:
        plist = [d.level((2,2)) for d in data] 
    plist = [d.topoints() for d in data]    
    return np.vstack(plist)

 


def load_dlist(rfiles,reader=None,*args,**kwargs):
    """Extracted from plot_repeat. Read a set of rfiles to a dlist.
    readers and additional arguments can be passed as scalars or lists.

    You can pass additional arguments to the reader in different ways:
     - pass them individually, they will be used for all readers
         load_dlist(.. ,option1='a',option2=1000)
     - to have individual reader parameters pass them as dictionaries (a same number as rfiles),
         load_dlist(.. ,{option1='a',option2=1000},{option1='b',option3='c'},..)

    Example:
        dlist=load_dlist(rfiles,reader=fitsWFS_reader,scale=(-1,-1,1),
                units=['mm','mm','um'])

        dlist2=load_dlist(rfiles,fitsWFS_reader,[{'scale':(-1,-1,1),
                'units':['mm','mm','um']},{'scale':(1,1,-1),
                'units':['mm','mm','um']},{'scale':(-1,-1,1),
                'units':['mm','mm','$\mu$m']}])
    2019/04/08 made function general from plot_repeat, moved to dlist.
    """

    import pdb
    

    if reader is None:
        reader=auto_reader(rfiles[0])
        
    if np.size(reader) ==1:
        reader=[reader]*len(rfiles)
    '''
    if kwargs : #passed explicit parameters for all readers
        pdb.set_trace()
        kwargs=[kwargs]*len(rfiles)
    else:
        if np.size(args) ==1:
            kwargs=[args]*len(rfiles)
        else:
            if np.size(args) != len(rfiles):
                raise ValueError
    '''

    if kwargs : #passed explicit parameters for all readers
        #pdb.set_trace()
        #vectorize all values
        for k,v in kwargs.items():
            if (np.size(v) == 1):
                kwargs[k]=[v]*len(rfiles)    
            elif (len(v) != len(rfiles)):
                kwargs[k]=[v]*len(rfiles)
    
    #pdb.set_trace()
    #transform vectorized kwargs in list of kwargs
    kwargs=[{k:v[i] for k,v in kwargs.items()} for i in np.arange(len(rfiles))]
    
    #kwargs here is a list of dictionaries {option:value}, matching the readers
    dlist=[Data2D(file=wf1,reader=r,**k) for wf1,r,k in zip(rfiles,reader,kwargs)]

    return dlist

def test_load_dlist(rfiles):

    dlist=load_dlist(rfiles,reader=fitsWFS_reader,scale=(-1,-1,1),
            units=['mm','mm','um'])

    dlist2=load_dlist(rfiles,fitsWFS_reader,[{'scale':(-1,-1,1),
            'units':['mm','mm','um']},{'scale':(1,1,-1),
            'units':['mm','mm','um']},{'scale':(-1,-1,1),
            'units':['mm','mm','$\mu$m']}])
    return dlist,dlist2


def mark_data(datalist,outfile=None,deg=1,levelfunc=None,propertyname='markers',direction=0):
    """plot all data in a set of subplots. Allows to interactively put markers
    on each of them. Return list of axis with attached property 'markers'"""
    import logging
    from matplotlib.widgets import MultiCursor
    from plotting.add_clickable_markers import add_clickable_markers2

    try:
        logger
    except NameError:
        logger=logging.getLogger()
        logger.setLevel(logging.DEBUG)
    #datalist = [d() for d in datalist]]
    axes=list(compare_images([[levellegendre(y,wdata,deg=6),x,y] for (wdata,x,y) in datalist],
        commonscale=True,direction=direction))
    fig=plt.gcf()
    multi = MultiCursor(fig.canvas, axes, color='r', lw=.5, horizOn=True, vertOn=True)
    plt.title('6 legendre removed')

    if outfile is not None:
        if os.path.exists(fn_add_subfix(outfile,'_markers','.dat')):
            np.genfromtxt(fn_add_subfix(outfile,'_markers','.dat'))

    for a in axes:
        a.images[0].set_clim(-0.01,0.01)
        add_clickable_markers2(ax=a,propertyname=propertyname,block=True)

    #display(plt.gcf())

    #Open interface. if there is a markers file, read it and plot markers.
    #when finished store markers

    for a in axes:
        logger.info('axis: '+str(a))
        logger.info('markers: '+str(a.markers))
        print(a.markers)

    if outfile is not None:
        np.savetxt(fn_add_subfix(outfile,'_markers','.dat'),np.array([a.markers for a in axes]))
        plt.savefig(fn_add_subfix(outfile,'_markers','.png'))

    return axes

def add_markers(dlist):
    """interactively set markers, when ENTER is pressed,
    return markers as list of ndarray.
    It was align_active interactive, returning also trans, this returns only markers,
    transforms can be obtained by e.g. :
    m_trans=find_transform(m,mref) for m in m_arr]
      """

    #set_alignment_markers(tmp)
    xs,ys=find_grid_size(len(dlist),5)[::-1]

    fig,axes=subplot_grid(len(dlist),(xs,ys),sharex='all',sharey='all')

    axes=axes.flatten()
    maximize()
    for i,(d,ax) in enumerate(zip(dlist,axes)):
        plt.sca(ax)
        ll=d.level(4,byline=True)
        ll.plot()
        plt.clim(*remove_outliers(ll.data,nsigma=2,itmax=1,span=True))
        add_clickable_markers2(ax,hold=(i==(len(dlist)-1)))

    return [np.array(ax.markers) for ax in axes]

def align_interactive(dlist,find_transform=find_affine,mref=None):
    """plot a list of Data2D objects on common axis and allow to set
    markers. When ENTER is pressed, return markers and transformations"""

    m_arr = add_markers (dlist)

    #populate array of transforms
    mref = mref if mref is not None else m_arr[0]
    m_trans = [find_transform(m,mref) for m in m_arr]

    return m_arr,m_trans

def psd2an(dlist,wfun=np.hanning,
                dis=False,ymax=0.05,outfolder="",subfix='_psd2d',*args,**kwargs):
    """2d psd analysis with threshold on sag removed data. Return a list of psd2d.
    ymax sets top of scale for rms right axis.
    if outfolder is provided, save psd2d_analysis plot with dlist names+subfix"""
    m_psd=[]
    for dd in dlist:
        m_psd.append(dd.level(2,byline=True).psd(analysis=True,
            title=os.path.basename(outfolder),wfun=wfun,*args,**kwargs))
        #psd2d_analysis(*dd.level(2,byline=True)(),
        #                 title=os.path.basename(outfolder),wfun=wfun,*args,**kwargs)
        #m_psd.append((fs,ps))
        ax=plt.gcf().axes[-1]
        ax.set_ylim([0,ymax])
        plt.grid(0)
        ax.grid()
        plt.suptitle(dd.name+' - hanning window - sag removed by line')
        if outfolder is not None:
            plt.savefig(os.path.join(outfolder,dd.name+subfix+'.png'))
        if dis:
            display(plt.gcf())
    return m_psd

def extract_psd(dlist,rmsthr=0.07,rmsrange=None,prange=None,ax2f=None,dis=False):
    """use psd2an, from surface files return linear psd."""
    m_psd=psd2an(dlist,rmsthr=rmsthr,rmsrange=rmsrange,prange=prange,ax2f=ax2f,dis=dis)

    plt.figure()
    plt.clf()
    labels=[d.name for d in dlist]
    m_tot=[]
    #gives error, m_psd is a list of PSD2D objects, there is no f,p, check psd2an
    for (f,p),l in zip(m_psd,labels):
        ptot=projection(p,axis=1)
        #plot_psd(f,ptot,label=l,units=d.units)
        np.savetxt(fn_add_subfix(d.name,'_calib_psd','.dat',
            pre=outfolder+os.path.sep),np.vstack([fs,ptot]).T,
            header='SpatialFrequency(%s^-1)\tPSD(%s^2 %s)'%(u[0],u[2],u[0]))
        m_tot.append((f,ptot))
    plt.title(outname)

    plt.legend(loc=0)
    if dis:
        display(plt.gcf())
    return m_tot
