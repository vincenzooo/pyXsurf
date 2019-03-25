"""
2018/06/07 v1.3
v1.2 was not convenient, switch back to same interface as 1.1:
don't modify self, return copy.

2018/06/06 v 1.2
After attempt to modify interface in a way that modify self, switch back to same interface as 1.1. It is inconvenient to have always to create a copy, in particular in interactive mode when same operation can be repeated multiple times you need to inizialize data every time (e.g. rotate).

2018/06/06 v1.1
methods are written to consistently return a copy without modifying self.
But this doesn't work, always need to assign, cannot chain (with property assignment, it works on methods) or apply to set of data in list. 

Give some attention to inplace operators that can link to external data. At the moment when class is created from data, x, y these are assigned directly to the property resulting in a link to the original data. Some methods have inplace operations that will reflect on initial data, others don't."""

"""
programming notes:
-inplace methods-
A method acting on Data2D (e.g. level) can modify (o reassign to) self and return self or return a copy.

e.g. a method crop (or remove nancols) that return slices of data and x/y reassign to self.data/x/y views of original data.

A method that reassigns self.data=...
most of the time stays linked to original data.
If inplace changes (e.g. element-wise assignment) are then performed on the property, the change is reflected in the original data.
See notebook Python Programming Notes.

To remove this behavior, a copy of the data or view must be performed at some point (e.g. explicitly self.data.copy() or implicitly deepcopy(self).



-attempt to subclass ndarray-
def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None, info=None):
    # Create the ndarray instance of our type, given the usual
    # ndarray input arguments.  This will call the standard
    # ndarray constructor, but return an object of our type.
    # It also triggers a call to InfoArray.__array_finalize__
    obj = super(Data2D, subtype).__new__(subtype, shape, dtype,    #(InfoArray
                                            buffer, offset, strides,
                                            order)
    # set the new 'info' attribute to the value passed
    #obj.info = info
    # Finally, we must return the newly created object:
    return obj

def __array_finalize__(self, obj):
    # ``self`` is a new object resulting from
    # ndarray.__new__(InfoArray, ...), therefore it only has
    # attributes that the ndarray.__new__ constructor gave it -
    # i.e. those of a standard ndarray.
    #
    # We could have got to the ndarray.__new__ call in 3 ways:
    # From an explicit constructor - e.g. InfoArray():
    #    obj is None
    #    (we're in the middle of the InfoArray.__new__
    #    constructor, and self.info will be set when we return to
    #    InfoArray.__new__)
    if obj is None: return
    # From view casting - e.g arr.view(InfoArray):
    #    obj is arr
    #    (type(obj) can be InfoArray)
    # From new-from-template - e.g infoarr[:3]
    #    type(obj) is InfoArray
    #
    # Note that it is here, rather than in the __new__ method,
    # that we set the default value for 'info', because this
    # method sees all creation of default objects - with the
    # InfoArray.__new__ constructor, but also with
    # arr.view(InfoArray).
    #self.info = getattr(obj, 'info', None)
    # We do not need to return anything
"""
    


import os
import matplotlib.pyplot as plt
import numpy as np

#from pySurf._instrument_reader import read_data, csvZygo_reader,csv4D_reader,sur_reader,auto_reader

from pySurf._instrument_reader import auto_reader
from pySurf.data2D import plot_data,get_data, level_data, save_data, rotate_data, remove_nan_frame, resample_data
from pySurf.data2D import read_data,sum_data, subtract_data, projection, crop_data, transpose_data, apply_transform, register_data, data_from_txt

from pySurf.psd2d import psd2d,plot_psd2d,psd2d_analysis,plot_rms_power,rms_power

from pySurf.points import matrix_to_points2

from copy import deepcopy
from dataIO.span import span
from dataIO.fn_add_subfix import fn_add_subfix

import pdb

from pySurf.affine2D import find_affine



class Data2D(object):  #np.ndarrays
    """A class containing 2d data with x and y coordinates. It has a set of methods for analysis operations.
    Function methods: return a copy with new values.
    to use as modifier (procedure), e.g.:
        a=a.level() """
    
    """
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(Data2D, subtype).__new__(subtype, shape, dtype,    #(InfoArray
                                                buffer, offset, strides,
                                                order)
        # set the new 'info' attribute to the value passed
        #obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        #self.info = getattr(obj, 'info', None)
        # We do not need to return anything
    """
    
    def __init__(self,data=None,x=None,y=None,file=None,reader=None,units=None,name=None,*args,**kwargs):
        """can be initialized with data; data,x,y; file; file, x, y.
        if x and y are provided, they override x and y if matching number of elements, or used as range if two element."""
        #from pySurf.instrumentReader import reader_dic
        import pdb
        #pdb.set_trace()
        
        if isinstance (data,str):
            print ('first argument is string, use it as filename')
            file=data
        #pdb.set_trace()
        self.file=file #initialized to None if not provided
        if file is not None:
            assert data is None
            #store in xrange values for x if were passed
            xrange=span(x) if x is not None else None
            yrange=span(y) if y is not None else None
            if reader is None:
                reader=auto_reader(file)
            data,x,y=read_data(file,reader=reader,*args,**kwargs) #calling without arguments skips register, however skips also reader argumnets, temporarily arranged with pop in read_data to strip all arguments for
                #register data and pass the rest to reader
            #pdb.set_trace()
            if np.size(xrange) == data.shape[1]:
                self.x=xrange
            elif np.size(xrange) == 2:
                x=np.linspace(*xrange,data.shape[1])
            elif xrange is not None:
                print("wrong number of elements for x (must be 2 or xsize_data), it is instead",np.size(xrange))
                raise ValueError                
            
            # set self.header to file header if implemented in reader, otherwise set to empty string""         
            try:  
                #kwargs['header']=True
                self.header=reader(file,header=True,*args,**kwargs)
            except TypeError:  #unexpected keyword if header is not implemented
                self.header=""
                #raise 
        else:
            if data is not None:
                if x is None:
                    x=np.arange(data.shape[1])
                if y is None:
                    y=np.arange(data.shape[0])
        
        if data is not None:
            data,x,y=register_data(data,x,y,*args,**kwargs)# se read_data chiamasse register, andrebbe un'indentazione +1
            self.data,self.x,self.y=data,x,y
        
        self.units=units
        if name is not None:
            self.name=name
        elif file is not None:
            self.name=os.path.basename(file)
        else:
            self.name=""
        #print(name)
    
    def __call__(self):
        return self.data,self.x,self.y

    def __add__(self,other,*args,**kwargs):
        res = self.copy()
        if isinstance(other,Data2D):
            if other.units is not None:
                assert self.units == other.units
            res=Data2D(*sum_data(self(),other(),*args,**kwargs),units=self.units,name=self.name + " + " + other.name)
        elif len(np.shape(other))<=1:  #add scalar
            if np.size(other) == 1:
                res.data = res.data +other
            else:
                raise ValueError ("wrong number of elements in Data2D sum")
        elif len(np.shape(other))==2:  #sum with matrix
            #completely untested
            
            if other.shape != self.data.shape:
                raise ValueError ('Different size in sum')
            res.data=res.data+other
        else:
            raise ValueError('Multiply Data2D by wrong format!')

        return res    
        
    def __radd__(self,other):
        return self.__add__(other)
        
        
    def __mul__(self,scale,*args,**kwargs):
        res = self.copy()
        if len(np.shape(scale))<=1:  #multiply by scalar(s)
            if np.size(scale) == 3:
                res.x = scale[0] * res.x
                res.y = scale[1] * res.y
                res.data = scale[2] * res.data
            elif np.size(scale) == 1:
                res.data = res.data * scale
            else:
                raise ValueError ("wrong number of elements in Data2D multiplication")
        elif len(np.shape(scale))==2:  #multiply by matrix
            #completely untested
            print ("warning, multiplication between arrays was never tested")
            assert scale.shape == self.data.shape
            res.data=res.data*scale
        elif isinstance(scale,Data2D):
            tmp=scale.resample(self)
            res=self.copy()
            res.data=self.data*tmp.data
        else:
            raise ValueError('Multiply Data2D by wrong format!')
        return res
        
    def __rmul__(self,scale,*args,**kwargs):
        return self.__mul__(scale,*args,**kwargs)
        
    def __neg__(self):
        return self.__mul__(-1)
        
    def __sub__(self,other,*args,**kwargs):
        #assert self.units == other.units
        #res=Data2D(*subtract_data(self(),other(),*args,**kwargs),units=self.units)
        res=self.__add__(self,-other,*args,**kwargs)
        if hasattr(other,name):
            res.name = self.name + " - " + other.name
        return res
        
    def __truediv__(self,other):
        return self*(1./other)
        
        
    def plot(self,title=None,*args,**kwargs):
        
        stats=kwargs.pop('stats',2) #to change the default behavior
        nsigma=kwargs.pop('nsigma',3) #to change the default behavior
        res=plot_data(self.data,self.x,self.y,units=self.units,
            stats=stats,*args,**kwargs)
        if title is None:
            if self.name is not None:
                title = self.name
        plt.title(title)
        return res

    def load(self,filename,*args,**kwargs):
        self.data,self.x,self.y = data_from_txt(filename,*args,**kwargs)
        
    def save(self,filename,*args,**kwargs):
        return save_data(filename,self.data,self.x,self.y,*args,**kwargs)
    save.__doc__=save_data.__doc__
    
    def rotate(self,angle,*args,**kwargs):
        res = self.copy()
        res.data,res.x,res.y=rotate_data(self.data,self.x,self.y,angle,*args,**kwargs) 
        return res

    def rot90(self,k=1,*args,**kwargs):
        """uses numpy.rot90 to rotate array of an integer multiple of 90 degrees in direction
        (from first to second axis)."""
        res = self.copy()
        res.data,res.x,res.y=rotate_data(*res(),k=k,*args,**kwargs)
            
        return res
        
    def transpose(self):
        res = self.copy()
        res.data,res.x,res.y = transpose_data(self.data,self.x,self.y)
        return res
    
    def apply_transform(self,*args,**kwargs):
        res = self.copy()
        res.data,res.x,res.y=apply_transform(self.data,self.x,self.y,*args,**kwargs)
        return res
        
    def apply_to_data(self,func,*args,**kwargs):
        """apply a function from 2d array to 2d array to data."""
        res = self.copy()
        res.data=func(self.data,*args,**kwargs)
        return res
    
    def crop(self,*args,**kwargs):
        res=self.copy()
        res.data,res.x,res.y=crop_data(self.data,self.x,self.y,*args,**kwargs) 
        return res

    def level(self,*args,**kwargs):
        res=self.copy()
        res.data,res.x,res.y=level_data(self.data,self.x,self.y,*args,**kwargs) 
        return res
    
    def resample(self,other,*args,**kwargs):
        """TODO, add option to pass x and y instead of other as an object."""
        res=self.copy()
        if self.units is not None and other.units is not None:
            if self.units != other.units:
                raise ValueError('If units are defined they must match in Data2D resample.')
        res.data,res.x,res.y=resample_data(res(),other(),*args,**kwargs)  
        return res
        
    def psd(self,wfun=None,rmsnorm=True,analysis=False,subfix='',name=None,*args,**kwargs):
        """return a PSD2D object with 2D psd of self.
        If analysis is set True, psd2d_analysis plots are generated and related parameters
          are passed as args.
        subfix and name are used to control the name of returned object."""
        
        if analysis:
            f,p=psd2d_analysis(self.data,self.x,self.y,wfun=wfun,units=self.units,*args,**kwargs)
        else:
            f,p=psd2d(self.data,self.x,self.y,wfun=wfun,norm=1,rmsnorm=rmsnorm) 
        
        newname = name if name is not None else fn_add_subfix(self.name,subfix)
        
        return PSD2D(p,self.x,f,units=self.units,name=newname)
        
    def remove_nan_frame(self,*args,**kwargs):
        res = self.copy()
        res.data,res.x,res.y=remove_nan_frame(self.data,self.x,self.y,*args,**kwargs)  
        return res       
        
    def topoints(self):
        return matrix_to_points2(self.data,self.x,self.y) 
         
        
    def std(self,axis=None):
        """return standard deviation of data excluding nans"""
        return np.nanstd(self.data,axis=axis)
        
    def copy(self):
        """copy.deepcopy should work well."""
        return deepcopy(self)
        
    def printstats(self,label=None,fmt='%3.2g'):
        if label is not None:
            print(label)
        s=("%s PV: "+fmt+", rms: "+fmt)%(self.name,span(self.data,size=True),
                             np.nanstd(self.data))
        print(s)
        return s
        
    def align_interactive(self,other,find_transform):
        """interactively set markers and align self to other.
        Return aligned Data2D object"""
        m1,m2=add_markers([self,other])
        trans=find_transform(m1,m2)
        return self.apply_transform(trans)
        
    def remove_outliers(self,*args,**kwargs):
        """use dataIO.remove_outliers to remove outliers"""
        self.data=self.data[remove_outliers(self.data,mask=True,*args,**kwargs)]
        
    def histostats(self,*args,**kwargs):
        res =data_histostats(self.data,self.x,self.y,units=self.units)
        plt.title(self.name)
        return res
    
from scripts.dlist import add_markers, align_interactive  #functions operating on Data2D
    

from pySurf.psd2d import psd2d,plot_psd2d 
class PSD2D(Data2D):
    """It is a type of data 2D with customized behavoiur and additional properties
    and methods."""
    def __init__(self,*args,**kwargs):
        """needs to be initialized same way as Data2D"""   
        #if a surface or a wdata,x,y are passed, these are interpreted as 
        Data2D.__init__(self,*args,**kwargs)
    
    def plot(self,*args,**kwargs):
        u=kwargs.pop('units',self.units) 
        return plot_psd2d(self.y,self.data,self.x,units=u,*args,**kwargs)
        
    def avgpsd(self,*args,**kwargs):
        """avg, returns f and p. Can use data2D.projection keywords `span` and `expand` to return PSD ranges."""
        return self.y,projection(self.data,axis=1,*args,**kwargs)
        
    def rms_power(self,plot=False,*args,**kwargs):
        """Calculate rms slicec power. 
            If plot is set also plot the whole thing."""
            
        if plot:
            return plot_rms_power(self.y,self.data.self.x,units=self.units,*args,**kwargs)
        else:
            """this is obtained originally by calling rms_power, however the function deals with only scalar inmot for rms range.
            Part dealing with multiple ranges is in plot_rms_power, but should be moved to rms_power."""
            raise NotImplementedError
        
def test_rot90():
    a=np.ones(250).reshape((25,10))
    a[6:7,6:9]=3
    d=Data2D(a)
    
    plt.close('all')
    d.plot()
    plt.title('original')
    
    plt.figure()
    #return a,d
    c=d.rot90()
    c.plot()
    plt.title('rotated')
    
    plt.figure()
    #return a,d
    c=d.rot90(k=2)
    c.plot()
    plt.title('rotated k=2')
    
    plt.figure()
    #return a,d
    c=d.rot90(k=2,center=(10,5))
    c.plot()
    plt.title('rotated k=2 about (10,5)')
        
def test_class_init():
    """test init and plot"""
    from pySurf.instrumentReader import matrixZygo_reader
    from dataIO.fn_add_subfix import fn_add_subfix
    relpath=r'test\input_data\zygo_data\171212_PCO2_Zygo_data.asc'
    outpath=r'test\results\data2D_class'
    
    wfile= os.path.join(os.path.dirname(__file__),relpath)
    (d1,x1,y1)=matrixZygo_reader(wfile,ytox=220/1000.,center=(0,0))
    
    plt.figure(1)
    plt.clf()
    plt.suptitle(relpath)
    plt.title('use plot_data function')
    plot_data(d1,x1,y1,aspect='equal')

    a=Data2D(d1,x1,y1)
    plt.figure(2)
    plt.clf()
    plt.title('From data')
    a.plot(aspect='equal')
    
    b=Data2D(file=wfile,ytox=220/1000.,center=(0,0))
    plt.figure(3)
    plt.clf()
    plt.title('from filename')
    b.plot(aspect='equal')
    
    b.save(os.path.join(outpath,os.path.basename(fn_add_subfix(relpath,"",".txt"))),
        makedirs=True)
    b.remove_nan_frame()
    
    plt.figure()
    plt.title('removed nans')
    b.plot(aspect='equal')
    
if __name__=='__main__':
    plt.close('all')
    test_class_init()