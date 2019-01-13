import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

"""
Classes for markers and collections of markers. Minimum structure for a marker 
    is a list of two coords, for a collection is a list of markers.
    Both can be converted in dicts by adding default labels.
    Graphical aspects of the markers and collections are also handled by 
        the classes.
    Markers is a working/semi-working example of iteration with 

TODO: each marker must have a `visible` property, for a Markers object 
    can be set as list of bool or scalar
TODO: should be implemented as a ordered dict. a collection of markers is a dict
    itself (unordered?). 
TODO: implement a more sofisticated visualization method than `plot`, e.g. set
    a `visible` property and an `update` method that adds or remove markers from 
    plot. This should be built with the idea of being easy to connect to external
    controls, e.g. toolbox buttons, add_clickable_markers, button widget 

    2018 vcotroneo@cfa.harvard.edu
"""

class Markers(object):
    """A marker is a set of points intended as Nx2 array xy.
    It has a name for the group of markers and each marker has a name (tag)
    associated.
    """
    #better to implement markers as arrays and associated tags rather than dict
    #and convert to dict on need.
    def __init__(self,xy=None,name=None,*args,**kwargs):
        if xy is None:
            xy={}
        if isinstance(xy,dict):
            self.xy=xy
        else:
            self.xy={}
            k=np.arange(len(xy))+1
            for tag,point in zip(k,xy):
                self[tag] = point
        self.name=name
        
    def plot(self,subplots=0,points=None,w=None,**kwargs):
        """xy is N points x,y {Nx2}. plot circles around points.
        if subplots is set, generate subplots with zoomed area."""
        #2015/12/12 xy was called m in previous (not very used) versions.
    
        xy=self.xy
        if subplots and (points is not None):
            nrows = int(np.sqrt(len(xy[:,0])))+1 #int(math.ceil(len(subsl) / 2.))
            plt.figure()
            fig, axs = plt.subplots(nrows, nrows,squeeze=True)
            for marker,ax in zip(xy,axs.flatten()[0:len(xy)]):
                plt.sca(ax)
                #scatter=scatter if kwargs.has_key('scatter') else True
                if w is None:
                    frac=0.1
                    w=[(ax.get_xlim()*np.array([-1,1])).sum()*frac,
                       ax.get_ylim()*np.array([-1,1])).sum()*frac]
                elif:
                    np.size(w) == 1:
                        w=np.repeat(w,2)
                else:
                    assert np.size(w) == 2
                    
                plt.xlim(marker[0:1]+[-w/2,w/2])
                plt.ylim(marker[1:2]+[-w/2,w/2])
                #plot_points(points,scatter=scatter,bar=0,**kwargs)
                # Make an axis for the colorbar on the right side
            for ax in axs.flatten()[len(xy):]:
                plt.cla()
            #cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            #plt.colorbar()
        else:
            plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor='none', c='w',lw=20,markersize=20)
            
    def pick(self):
        def onpick4(event):
            #print 'pick!'
            artist = event.artist
            if isinstance(artist, AxesImage):
                im = artist
                A = im.get_array()
                #print('onpick4 image', A.shape)
                print(event.mouseevent.x,event.mouseevent.y)
        fig=plt.gcf()
        fig.canvas.mpl_connect('pick_event', onpick4)
            

class Marker(object):
    """A 2-d array of position with attached grafical properties and ability to plot itself."""
    def __init__(self,x,y, *args, **kw):
        """here it needs to process the possible input types. If dict of dict, create behind
        the scenes a MarkerCollection object."""
        self.x=x
        self.y=y
        self.visible=kw.pop('visible',True)
        self.tag=kw.pop('tag',None)
        self.line=None
        self.ax=plt.gca()
        
        #assign default or automatic graphical properties
        
    def plot(self,ax=None,*args,**kwargs):
        """"""
        if self.visible:
            if self.line is not None:
                i=self.ax.lines.index(self.line) #index of the old line
            self.line.remove()
            #the new line doesn't retain 
            plt.plot(self.x,self.y,*args,**kwargs)
            

            
            
from collections.abc import Mapping
    
class MarkerSet(Mapping):
    """Another way of making a set of markers as dictionary. 
    as explained at http://www.kr41.net/2016/03-23-dont_inherit_python_builtin_dict_type.html
    as a better alternative to inherit dict."""
    
    def __init__(self, *args, **kw):
        """here it needs to process the possible input types. If dict of dict, create behind
        the scenes a MarkerCollection object."""
        self._storage = dict(*args, **kw)
    def __getitem__(self, key):
        return self._storage[key]
    def __iter__(self):
        return iter(self._storage)    # ``ghost`` is invisible
    def __len__(self):
        return len(self._storage)

class MarkersCollection(Mapping):
    """It is basically a dictionary of MarkerSet objects with ability to propagate property assignments. 
    as explained at http://www.kr41.net/2016/03-23-dont_inherit_python_builtin_dict_type.html
    as a better alternative to inherit dict."""
    
    def __init__(self, *args, **kw):
        """here it needs to process the possible input types. If dict of dict, create behind
        the scenes a MarkerCollection object."""
        self._storage = dict(*args, **kw)
    def __getitem__(self, key):
        return self._storage[key]
    def __iter__(self):
        return iter(self._storage)    # ``ghost`` is invisible
    def __len__(self):
        return len(self._storage)
        
def Markers(m):
    """Returns a proper object MarkerSet or MarkersCollection according to the input format."""
    pass
    
def test_markers():
    return
    
        
if __name__=="__main__":
    imgfile=r'test\input_data\03_FS04c_16_100x.jpg'
    data=plt.imread(imgfile)
    plt.imshow(data, picker=True)
    plt.show()
    m=Markers()
    m.pick()

    