
import sys
import numpy as np
import sys
# sys.ps1 = 'SOMETHING' # this redefines the primary prompt of system command line, not sure
# why it was here, but it is called a number of times when importing
print ('cane')
from matplotlib import pyplot as plt


#2017/08/03 made some adjustments, mostly in interactivity. However plt.ion/ioff don't update immediately,
#  it is necessary to close the window.

def add_clickable_markers(fig=None,ax=None,key='enter',propertyname='markers',modkey='control',hold=False):
    """Makes pickable the image (image only) in the current figure.
    add a list `propertyname` to the figure. if axis is passed, markers are added only to specific axis. This way it is possible to associate differently named markers to 
        different axis in same figure.
        
        Deprecated, use add_clickable_markers2"""
    
    #useful references  
    #http://astronomy.nmsu.edu/holtz/a575/ay575notes/node6.html
    #https://matplotlib.org/examples/event_handling/keypress_demo.html
    def press(event):
        #print('press', event.key)
        sys.stdout.flush()
        if event.key == key:
            fig.canvas.mpl_disconnect(cid1)
            fig.canvas.mpl_disconnect(cid2)
            #fig.canvas.stop_event_loop()
            print (plt.isinteractive())
            plt.interactive(interact_store)  #this sets interactive to original status, but the behaviour doesn't update (if set to ON you still need to close the figure)
            #plt.show(block=False)
            print (plt.isinteractive())
            fig.exit=True  #experimental for loop
        
    def onclick(event):
        button=event.button
        x=event.xdata
        y=event.ydata
        fig=event.canvas.figure
        cax=event.inaxes
        
        if ax is None or cax==ax:         
            if button==1: 
                if event.key is not None:
                    if event.key == modkey:
                        #print (id(fig),propertyname)
                        l=getattr(fig,propertyname)
                        l.append([event.xdata, event.ydata])
                        #add point
                        cax.set_autoscale_on(False)
                        cax.plot(x,y,'or',picker=5,markerfacecolor=None)
            if button==3:
                if event.key is not None:
                    if event.key == modkey:
                        l=getattr(fig,propertyname)
                        if len(l)>0:
                            dist=((np.array(l)-(x,y))**2).sum(axis=1)
                            pop=l.pop(np.argmin(dist))
                            dist2=[(np.array((pop[0]-a.get_xdata(),pop[1]-a.get_ydata()))**2).sum()  for a in ax.lines]
                            cax.lines.pop(np.argmin(dist2))
        
        plt.draw()
        #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        #event.button, event.x, event.y, event.xdata, event.ydata)
    
    def handle_close(evt):
        print('Closed Figure!')
        fig.canvas.mpl_disconnect(cid3)

    #to disconnect:
    #fig.canvas.mpl_disconnect(cid)
    """
    def on_pick(event):
        # not used. this is an alternative way, it automatically manage the selection of the plotted point.
        import matplotlib
        thisline = event.artist
        if isinstance(thisline,matplotlib.lines.Line2D):
            #code to remove point from markers list
            thisline.remove()
            plt.show()
    """
    
    if fig is None:
        fig=plt.gcf()
    interact_store=plt.isinteractive()
    #plt.show(block=True) #This makes impossible to select the points because it blocks events handling.
    #plt.show(block=False) #This works to select the points, but it doesn't release afterwards.

    if hold:
        print ('hold')
        plt.ioff()
    else:
        plt.ion()  #this works in setting interactive mode off independently on starting status. However it sets the status forever, setting ioff here and ion at the end alters
    # the status variable, but it is not applied to the window.
    # so it must be set off here if you want to proceed at the end, but it is necessary to 
    # close the windows to continue.
    
    #fig.canvas.start_event_loop(20)  #this is useful only for timed events (ginput?) documentation is
        #not  very clear to this regar, but it's clearly useless here.
    #add a list property (empty) with the chosen name for markers,
    #if property already exist it will automatically append.    
    if not hasattr(fig,propertyname):
        setattr(fig,propertyname,[])
        #setattr(fig.getattr(propertyname),'ax')
    fig.exit=False #experimental for loop
    #ax.imshow(im)
    #ax.autoscale(False)
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', press)
    cid3 = fig.canvas.mpl_connect('close_event', handle_close)
        
    return fig

def add_clickable_markers2(ax=None,key='enter',propertyname='markers',modkey='control',hold=False):
    """Enable adding markers by point and click to the passed or current axis. Returns axis.
    
    Markers are added by mouse left click + modkey and are stored in a list added
        to the axis as a property `propertyname`. They can be removed by modkey
        + mouse right click.
        `key` disconnect user interface stopping point and click operations.
    Markers with different propertyname can be associated to same axis. 
    hold is implemented in experimental way, only one axis should hold. Not sure
        what happens otherwise. Even with this working version there is flickering.

    This is updated version of add_clickable_markers, associate with axis, 
        not with figure so it is safe to use on subplots.
    ex:
    ax1=add_clickable_markers2()    #use current axis
    ax1=add_clickable_markers2(ax=ax1)   #use a specific axis
    print (ax1.markers)  #run this after markers are added
    
    2018 vcotroneo@cfa.harvard.edu
    """

    # 2017/10/26 it is more appropriate to associate markers with axis,
    # not with figures, so it will be possible to handle subplots.

    # useful references
    # http://astronomy.nmsu.edu/holtz/a575/ay575notes/node6.html
    # https://matplotlib.org/examples/event_handling/keypress_demo.html
    
    ## routines to be called on events
    ## Note that events and associations belong to figure, the axis is obtained with
    ##     event.inaxes
    def press(event):   #called on keyboard press event
        #print('press', event.key)
        cax=event.inaxes
        #print("cax:","cax is None!" if cax is None else cax)
        #print("*")
        if cax is None:
            cax=plt.gca() #if cursor was not on any axis when key press
        sys.stdout.flush()
        if event.key == key:
            fig=cax.figure
            print ("disconnecting cids:")
            #print(cids[:2]) #2017/10/26 not sure where cid3 is taken, maybe it works, added other cids.
            print("----")
            fig.canvas.mpl_disconnect(cids[0])   #w
            fig.canvas.mpl_disconnect(cids[1])
            #fig.canvas.stop_event_loop()
            print ("isinteractive?",plt.isinteractive())
            plt.interactive(interact_store)  #this sets interactive to original status, but the behaviour doesn't update (if set to ON you still need to close the figure)
            #plt.show(block=False)
            print ("isinteractive?",plt.isinteractive())  
            if hasattr(fig,'exit'):
                fig.exit=True
        
    def onclick(event):     #called on mouse button press event
        button=event.button
        x=event.xdata
        y=event.ydata
        #fig=event.canvas.figure
        cax=event.inaxes
        
        if ax is None or cax==ax:         
            if button==1: 
                if event.key is not None:
                    if event.key == modkey:
                        #print (id(fig),propertyname)
                        l=getattr(cax,propertyname)  ##getattr(fig,propertyname)
                        l.append([event.xdata, event.ydata])
                        #add point
                        ax.set_autoscale_on(False)
                        ax.plot(x,y,'or',picker=5,markerfacecolor=None)
            if button==3:
                if event.key is not None:
                    if event.key == modkey:
                        l=getattr(cax,propertyname)  ##getattr(fig,propertyname)
                        if len(l)>0:
                            dist=((np.array(l)-(x,y))**2).sum(axis=1)
                            pop=l.pop(np.argmin(dist))
                            dist2=[(np.array((pop[0]-a.get_xdata(),pop[1]-a.get_ydata()))**2).sum()  for a in ax.lines]
                            ax.lines.pop(np.argmin(dist2))
        
        plt.draw()
        #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        #event.button, event.x, event.y, event.xdata, event.ydata)
    
    def handle_close(evt):
        print('Closed Figure!')
        print('cids:')
        print(cids) #2017/10/26 not sure where cid3 is taken, maybe it works, added other cids.
        print("#---")
        #cids are created when function is run. are then they stored in handle_close?
        # to test: call function, then disconnect cids1 and 2 by enter key,
        # then close to call this routine that prints cids.
        # if they are defined, they are retained by function call. 
        for c in cids:
            evt.canvas.mpl_disconnect(c)
    
    # add property `propertyname` as empty list and associate event routines
    # to UI actions that modify list of markers.
    if ax is None:
        ax=plt.gca()
    else:
        if isinstance (ax,plt.Figure):
            raise ValueError   #old implementation was accepting figure
    
    interact_store=plt.isinteractive()
    """
    if hold:  #for test and debuffing
        print ('hold')
        plt.ioff()
    else:
        plt.ion()  
    """
    
    if not hasattr(ax,propertyname):
        setattr(ax,propertyname,[])
        
    fig=ax.figure
    # Deactivate the default keymap
    #fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    
    cids=[
    fig.canvas.mpl_connect('button_press_event', onclick),
    fig.canvas.mpl_connect('key_press_event', press),
    fig.canvas.mpl_connect('close_event', handle_close)
    ]
    
    print("cids established:\n",cids)
    
    def get_key():
        """proper way to implement press key with yield.
        see 
        https://gist.github.com/tacaswell/4545013"""
        pass
        
    if hold:
        setattr(fig,'exit',False)
        #i=0 #safety
        while not(fig.exit): # and i < 500:
            #i=i+1
            plt.pause(0.01)
        print ("exit")  #,i)
    return ax

class modify_markers(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.bound_keys = []
        self.bound_cid = {}

    def add_step_through(self, gen, key):
        
        key = key[0] # make a single char
        if key in self.bound_keys:
            raise RuntimeError("key %s already bound"%key)
        first_data = gen.next()
        self.ax.plot(first_data)
        self.fig.canvas.draw()
        self.bound_keys.append(key)
        def ontype(event):
            if event.key == key:
                try:
                    self.ax.plot(gen.next())
                    self.fig.canvas.draw()
                except StopIteration:
                    self.fig.canvas.mpl_disconnect(self.bound_cid[key])
                    del self.bound_cid[key]
                    self.bound_keys.remove(key)

        #self.bound_cid[key] = self.fig.canvas.mpl_connect('key_press_event', ontype)) 

 
class push_to_advance(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.bound_keys = []
        self.bound_cid = {}

    def add_step_through(self, gen, key):
        key = key[0] # make a single char
        if key in self.bound_keys:
            raise RuntimeError("key %s already bound"%key)
        first_data = gen.next()
        self.ax.plot(first_data)
        self.fig.canvas.draw()
        self.bound_keys.append(key)
        def ontype(event):
            if event.key == key:
                try:
                    self.ax.plot(gen.next())
                    self.fig.canvas.draw()
                except StopIteration:
                    self.fig.canvas.mpl_disconnect(self.bound_cid[key])
                    del self.bound_cid[key]
                    self.bound_keys.remove(key)

        self.bound_cid[key] = self.fig.canvas.mpl_connect('key_press_event', ontype)

def test_interac():
    """2018/09/05 try to solve with new idea of calling add_clickable_markers2 as a function that returns a value only
    when a key is pressed. to do this the interactive mode of matplotlib must be off.
    This works perfectly if imported from ipython shell, but if 
    called in qtconsole from jupyter notebook, doesn't disable
    interactive mode."""
    
    plt.ioff()
    plt.figure()
    plt.plot([1, 2, 3], [3, -3, 6])
    plt.show()
    plt.ion()

    
def test_interac2():
    """This probably works better than anything else. Still has flickering, but works from both ipython and jupyter. 
    
    For some strange reason, pause and coord must be
    made global, while there is no need for cid.
    Maybe because global and pause are in functions linked to mpl,
    while remove_cids is independent?"""

    
    def remove_cids():
        for c in cids:
            fig.canvas.mpl_disconnect(c)        
    
    def onclick(event):
        global coords,pause
        coords.append((event.xdata, event.ydata))
        print(pause,len(coords),coords)
        if (len(coords)==2):
            pause = False
            remove_cids()
    
    def handle_close(evt):
        remove_cids()

    global coords,pause
    coords = []
    pause = True

    fig, ax = plt.subplots()
    plt.plot([1, 2, 3], [3, -3, 6])
    plt.show() 
    
    cids=[
        fig.canvas.mpl_connect('button_press_event', onclick),
        #fig.canvas.mpl_connect('key_press_event', press),
        fig.canvas.mpl_connect('close_event', handle_close)
        ]
    print (cids)
 
    while pause:
        plt.pause(0.01)
    
    return coords

def test_multimarkers():
    """plots two datasets in subplots, return axis with attached clickable markers
    (CTRL+button). return axis ax1 and ax2 with markers properties attached named respectively 
    ax1.m1 and ax2.m2 """

    a = np.random.random(20).reshape((5, 4))
    b = -a
    ax1 = plt.subplot(121)
    plt.imshow(a, interpolation='none')
    ax2 = plt.subplot(122)
    plt.imshow(b, interpolation='none')
    add_clickable_markers2(ax=ax1, propertyname='m1')
    f = add_clickable_markers2(ax=ax2, propertyname='m2')
    plt.show()
    return ax1, ax2


def test_loop():
    """test for first version add_clickable_markers (on figure, deprecated)."""
    # doesn't work in spyder 2018/06/04

    plt.clf()
    a = np.random.random(25).reshape(5, 5)
    plt.imshow(a)
    print('add points with CTRL+left button, remove closest with CTRL+right. retax() prints the points plotted on current axes, f.markers print marker list.')

    f = add_clickable_markers() # has experimental property .exit()
    plt.show()
    # ax=plt.gca()
    print(f.markers)

    import time
    while not (f.exit):
        time.sleep(0.1)


def test_add_clickable_markers(a):

    plt.clf()
    plt.imshow(a)

    f = add_clickable_markers()
    plt.show()
    # ax=plt.gca()
    print(f.markers)
    return f


def test_multiaxis(ax1, ax2, hold=False):
    """plot on two axis in a blocking or non blocking mode."""

    ax1, ax2 = add_clickable_markers2(ax=ax1), add_clickable_markers2(ax=ax2, hold=hold)

    print(ax1.markers)
    print(ax2.markers)

    return ax1, ax2


def test_add_clickable_markers2(a, b):

    ax1 = plt.subplot(121)
    plt.imshow(a)
    ax2 = plt.subplot(122)
    plt.imshow(b)

    return test_multiaxis(ax1, ax2)


def test_hold(a):
    """test hold mechanism independently by how it is implemented inside
    add_clickable_marker2.
    test single axis figure and figure with two subplots holding until enter
        key is pressed."""

    print("test on single figure, will advance only after enter is pressed.")
    plt.close('all')
    plt.figure(1)
    plt.imshow(a)
    ax1 = add_clickable_markers2(hold=True)
    print(ax1.markers)

    print("test on subplot (only last one has hold)")
    plt.figure(2)
    ax2 = plt.subplot(121)
    plt.imshow(a)
    ax3 = plt.subplot(122)
    plt.imshow(a)
    
    ax2,ax3=add_clickable_markers2(ax=ax2),add_clickable_markers2(ax=ax3,hold=True)  
    print (ax2.markers)
    print (ax3.markers)
    return ax1,ax2,ax3

def test_disconnect():
    """test access to connection ids from functions in event handling. these
    are not stored anywhere, so they must be stored at event function definition
    from main function body."""
    
    imfile=r'test\input_data\IMG_4509_crop.jpg'
    a=plt.imread(imfile)
    ax1,ax2=test_add_clickable_markers2(a,a) 
    print ("function is called and axis returned, events functions connected to cids.")
        # the body of the function connect cids. event handling functions are defined
        #     and reference directly to cids, are these accessible?
    print("get disconnection of cid1 and 2 by pressing enter with focus on figure,")
        # this disconnected the cids associated
    print("then close the figure (disconnect also the 3rd cid).")
    return ax1,ax2


    
    
    
    return markers1,markers2

if __name__=="__main__":

    def retax():
        """diagnostic routine. return coordinates of all plotted points,
        assuming this are the symbols for the plotted markers."""
        ax=plt.gca()
        return [(a.get_xdata(),a.get_ydata()) for a in ax.lines]


    #from generalPlotting.pickPointList import add_clickable_markers
    imfile=r'test\input_data\IMG_4509_crop.jpg'
    a=plt.imread(imfile)

    print ('add points with CTRL+left button, remove closest with CTRL+right. retax() prints the points plotted on current axes, f.markers print marker list.')
    plt.figure(1)
    f1=test_add_clickable_markers(a)

    plt.figure(2)
    ax1,ax2=test_add_clickable_markers2(a,a)    
    
    
