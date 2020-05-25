import matplotlib
import matplotlib.pyplot as plt



def demo_display(testfunc):
    """A wrapper function that makes the wrapped function work independently on environment (Ipython, Jupyter notebook, or shell). Use as a decorator on test functions to generate an output appropriate for the situation."""
    
    pass

def test_demo_display():
    pass


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
            mng.window.state('zoomed') #works fine on Windows!
        else:
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
                mng.resize(xs,ys)
            except:
                print ("Unrecognized backend: ",backend) #not tested on different backends (only Qt)
                raise NotImplementedError
    plt.show()
    plt.pause(0.1) #this is needed to make sure following processing gets applied (e.g. tight_layout)

    
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