import matplotlib
import matplotlib.pyplot as plt
import pdb



def demo_display(testfunc):
    """A wrapper function that makes the wrapped function work independently on environment (Ipython, Jupyter notebook, or shell). Use as a decorator on test functions to generate an output appropriate for the situation."""
    
    pass

def test_demo_display():
    pass

def get_max_size():
    """returns max size for figure, in OS independent way."""
    try:
        import psutil
        if psutil.OSX:   #True (OSX is deprecated) VC:MACOS doesn't work
            import AppKit
            screen=AppKit.NSScreen.screens()[0]  #if multiple screens
            xs,ys=screen.frame().size.width, screen.frame().size.height
        elif psutil.WINDOWS: #False
            from win32api import GetSystemMetrics
            xs,ys=GetSystemMetrics(0),GetSystemMetrics(1)  #screen size in pixel
        elif psutil.LINUX:   #False
            from Xlib import display as Xd
            resolution = Xd.Display().screen().root.get_geometry()
            xs,ys= resolution.width,resolution.height
    except:
        print ("Unrecognized OS") #not tested on different backends (only Qt)
        raise NotImplementedError
        
    return xs, ys    

def maximize(backend=None,fullscreen=False,verbose=False):
    """Maximize window independently on backend.
    
    Fullscreen sets fullscreen mode, that is same as maximized, but it doesn't have title bar (press key F to toggle full screen mode)."""
    
    #see https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python

    if backend is None:
        backend=matplotlib.get_backend()
    mng = plt.get_current_fig_manager()
    if verbose:
        print ("function maximize:\nbackend: %s\nmanager:%s"%(backend,mng))
    if fullscreen:
        mng.full_screen_toggle()
    else:   
        if backend == 'wxAgg':
            mng.frame.Maximize(True)
        elif backend == 'Qt4Agg' or backend == 'Qt5Agg':
            mng.window.showMaximized()
        elif backend == 'TkAgg':
            try:
                mng.window.state('zoomed') #works fine on Windows!
            except:
                m = mng.window.maxsize()
                mng.window.geometry('{}x{}+0+0'.format(*m))
        elif backend == 'module://ipykernel.pylab.backend_inline':
            #it was mng.resize(xsi,ysi) 
            #added 2020/05/26
            #works fine on Windows
            dpi = matplotlib.rcParams['figure.dpi']
            xs,ys = get_max_size()
            
            xsi,ysi = xs/dpi,ys/dpi  #convert from pixel to inches for .resize
            #xsi, ysi = xsi/2, ysi/2 #additional reduction to make fonts visible inline in a notebook
            xsi, ysi = xsi, ysi #additional reduction to make fonts visible inline in a notebook
            #matplotlib.rcParams['figure.figsize'] = [xsi,ysi] #questo funziona su WIN matplotlib inline 2020/05/26
            #mng.resize(xsi,ysi)
            params = {'legend.fontsize': 'x-large',
            'figure.figsize': (xsi,ysi),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large'}
            matplotlib.rcParams.update(params)
        else:
            xs,ys = get_max_size()
            try:
                mng.resize(xs,ys) 
            except:
                print ("Unrecognized backend: ",backend) #not tested on different backends (only Qt)
                raise NotImplementedError
                  
    #plt.show() #2020/05/26 removed to avoid showing empty grid in notebook inline.
    
    plt.pause(0.1) #this is needed to make sure following processing gets applied (e.g. tight_layout)
    #plt.draw()  #I am sure I had used the other version because this was not working in the past. I try again to try to prevent creation of blank screen in jupyter inline terminal.   
    
def list_backends():        
    """list all the available backends."""
    #from https://stackoverflow.com/questions/3285193/how-to-switch-backends-in-matplotlib-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
    non_gui_backends = matplotlib.rcsetup.non_interactive_bk
    print ("Non Gui backends are:", non_gui_backends)
    print ("Gui backends I will test for", gui_env)
    for gui in gui_env:
        print ("testing", gui)
        try:
            matplotlib.use(gui,warn=False, force=True)
            from matplotlib import pyplot as plt
            print ("    ",gui, "Is Available")
            plt.plot([1.5,2.0,2.5])
            fig = plt.gcf()
            fig.suptitle(gui)
            plt.show()
            print ("Using ..... ",matplotlib.get_backend())
        except:
            print ("    ",gui, "Not found")
    return gui_env.extend(non_gui_backends)

def test_backend(gui_env=None):
    """test a list of interactive backend."""
    # from https://stackoverflow.com/questions/3285193/how-to-switch-backends-in-matplotlib-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    if gui_env is None: gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
    for gui in gui_env:
        try:
            print ("testing", gui)
            matplotlib.use(gui,warn=False, force=True)
            from matplotlib import pyplot as plt
            break
        except:
            continue
    print ("Using:",matplotlib.get_backend())
    
    
def test_maximize():
    """make a few subplots and maximize window"""
    plt.figure()
    ax1=plt.subplot(121)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)') 
    plt.plot([1,3,6],[12,33,-1])

    ax2=plt.subplot(122,sharex=ax1,sharey=ax1)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)') 
    plt.plot([1,3,6],[3,32,-13])
    
    maximize()
    plt.tight_layout()
    plt.show()