''' Vincenzo Cotroneo vcotroneo@cfa.harvard.edu 2016/03/07

It's a set of classes that add key commands to a matplotlib window.
The class is created and called. On call with an object and a figure as arguments,
links them and add the key commands to the matplotlib figure.

There are 

class BaseImageKey(AbstractImageKey):
    """A simple one with only help and reset (plt.close('all')).
    can be used as base."""

class BaseImageKey(AbstractImageKey):
    """A simple one with only help and reset (plt.close('all')).
    can be used as base."""

    
2018/06/06 it used to work, but dependent on Tk , not on my computer.
It was used to visualize help.
Also, some problem with arguments passed to event functions for BaseImageKey
example b.
'''

import matplotlib
tk_yes=False  #set to true if tkinter is present
try:

    #http://effbot.org/tkinterbook/tkinter-file-dialogs.htm
    import tkinter as tk
    import tkinter.filedialog as filedialog
    import tkinter.simpledialog
    from tkinter import Frame,Label,Entry,Button
    import tkinter.messagebox
    matplotlib.use('TkAgg')
    tk_yes=True
except ImportError:
    print("tkinter not laoded")
    pass
import matplotlib.pyplot as plt 


from dataIO.fn_add_subfix import fn_add_subfix
import re
import shutil
import os
import numpy as np

from dataIO.span import span 


import pdb
  

   
pe_yes=False  #set to true if piexif is present
try:
    import piexif
    pe_yes=True
except ImportError:
    pass

''' Following classes are minimal examples built for debugging, 
not needed any more after AbstractImageKey class works properly.

class minimalExample(object):
    """Try to connect to a function."""
        
    def __call__(self):
        """connect the current figure to an object adding I/O command interface based on keyboard to figure.
        object is passed to be manipulated in routine. Includes standard keys
        use H for list of commands."""
        
        def on_key_press(event):
            print "pressed"
            plt.clf()
            plt.draw()
            pass
                
        figure=plt.gcf()                               
        ax=plt.gca()
        
        print "connect"
        plt.connect('key_press_event', on_key_press)
            
        return figure  #return

class minimalExample2(object):
    """Try to connect to a function."""

  
    def __init__(self):
        def pltrnd():
            plt.plot(np.arange(100),np.random.rand(100))
            plt.draw()
        self.cmds={'c':plt.clf,'r':pltrnd}
        
    def __call__(self):
        """connect the current figure to an object adding I/O command interface based on keyboard to figure.
        object is passed to be manipulated in routine. Includes standard keys
        use H for list of commands."""
            
        def on_key_press(event):
            print "pressed"+event.key
            if event.key in self.cmds.keys(): 
                self.cmds[event.key]()
                print event.key+'!'
            plt.draw()
                
        figure=plt.gcf()                               
        ax=plt.gca()
        
        print "connect"
        plt.connect('key_press_event', on_key_press)
            
        return figure  #return
'''
     
class AbstractImageKey(object):
    """It's a class that when called connects a set of key shortcuts and functions
    to a figure. 
     
    Empty class to be inherited from.
    figure is not stored in any property, the ImageKey object defines the 
        keyboard interface.
    It is initialized with three lists of keys, functions and labels.

    For each of them a key is added connected to figure as shortcut to activate the function.
    In this version, function has self as argumennt: evt are kept in this class
        and in its derived.
    Label is used to generate an entry in the help dictionary, the help function
    is not defined here.
    functions and keys can be added with method add_key,
    if called return figure (unless arguments are provided, see connect).
    
    connect is called when the instance is called as a function.
    """
    
    def __init__(self,keys=None,functions=None,labels=None):
        self.cmds={}
        self.helpstrings={}
        # Deactivate the default keymap
        fig=plt.gcf()
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        if keys and functions and labels:
            for k,f,l in zip(keys,functions,labels):
                self.add_key(k,f,l)
        
    def add_key(self,key,function,label=None):
        if label is None:
            label=key+': '+function +'(?)'
        self.cmds[key]=function
        self.helpstrings[key]=label  
        
    def connect(self,obj=None,figure=None):
        """connect the current figure to an object adding I/O command interface based on keyboard to figure.
        object is passed to be manipulated in routine. (?)
        figure can be a plt.figure, otherwise current figure is used.
        return obj if not None, otherwise figure, I don't know why.
        It is called when self is called.
        
        """
        
        def on_key_press(evt):
            """call the corresponding function in self.cmds try both
            passing evt and witout, not sure it always work.
            it is connected to matplotlilb at the end"""
            #print ("pressed"+evt.key)
            #print (self.cmds)
            #print (evt.key in self.cmds.keys())
            if evt.key in self.cmds: 
                #print (self.cmds[evt.key])
                print ('call')
                try:
                    self.cmds[evt.key]()  
                except TypeError: 
                    #print('cane')
                    self.cmds[evt.key](self)
                    #problem is here the cmd can accept
                        #evt if needed to extract info, but can also be without
                        #parametner, how do I solve? use this limited workaround
                #except Exception as err:
                #    raise err
                plt.draw()                
        
        def handle_close(evt):
            """Useful maybe to save log? At the moment is saved in quit."""
            print('Closed Figure!')
                
        if figure is None: 
            figure=plt.gcf()
        else:
            plt.figure(figure.number)
        
        #ax=plt.gca()
                
        plt.connect('key_press_event', on_key_press)
        plt.connect('close_event', handle_close) 
            
        if obj is None:
            obj=figure
            
        return obj  #return

    def help_str(self):
        #import pdb
        #pdb.set_trace()
        #helpstrings=["-KEY COMMANDS-"].extend(['%s: %s'%(key,self.helpstrings[key]) for key in list(self.cmds.keys())])
        helpstrings=["-KEY COMMANDS-"]+['%s: %s'%(key,self.helpstrings[key]) for key in self.cmds]
        
        helptext= "\n".join(helpstrings)
        return helptext 
        
    def __call__(self,obj=None,figure=None):
        self.connect(obj,figure)
        
        
class BaseImageKey(AbstractImageKey):
    """A simple one with only help and reset (plt.close('all')).
    can be used as base.
    Adds tk interface to Abstract."""

    def __init__(self,*args,**kwargs):
    
        #def call_reset(evt):
        #    plt.close('all')
        
        def call_help(evt=None):
            helptext=self.help_str()  #calls parent self generated string
            try:            
                root = tk.Tk()
                #root.withdraw()
                w = tk.Label(root, text=helptext)
                w.pack()
                root.mainloop()            
            except Exception as err:
                raise err                
                #plt.connect(lambda x: print(help_str(x)))

        super(BaseImageKey, self).__init__(*args,**kwargs)
        #self.add_key('r',call_reset,'reset all plots')
        self.add_key('h',call_help,'Show this help')
        try:
            tk.Tk
        except NameError:
            print('backend_error')
        
    def __call__(self,object=None,figure=None):
        """
        This is very confused.
        connect the current figure to an object adding I/O command interface based on keyboard to figure.
        object is passed to be manipulated in routine. Includes standard keys
        use H for list of commands.
        If object is None is set to current axis (?!) call connect of abstract class
        with object and figure, then return obj."""
             
        def endroot(root):  #also root.destroy() ?
            root.quit()
            
        def handle_close(evt):
            """Useful maybe to save log? At the moment is saved in quit."""
            print('Closed Figure!')
                
        if figure is None: 
            figure=plt.gcf()
        else:
            plt.figure(figure.number)
                               
        ax=plt.gca()            
        if object is None:
            object==ax
            
        self.connect(object,figure)
        return object  #return the same object. It was figure,ax in SampleImageList, where nearly all code used plt.gca()
        
    
"""
http://stackoverflow.com/questions/7969352/matplotlib-interactively-select-points-or-locations
from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand

if 1:
    fig, ax = plt.subplots()
    ax.set_title('click on points', picker=True)
    ax.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
    line, = ax.plot(rand(100), 'o', picker=5)

    def onpick1(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            print 'X='+str(np.take(xdata, ind)[0]) # Print X point
            print 'Y='+str(np.take(ydata, ind)[0]) # Print Y point

    fig.canvas.mpl_connect('pick_event', onpick1)
"""

if __name__=="__main__":

    def pltrnd():
        """a test function that plots a random array on current axis."""
        plt.plot(np.arange(100),np.random.rand(100))
        plt.draw()
        print('pltrnd!')
        
    def display_help(self):
        """builds a help string from a list of """
        #helpstrings=["-KEY COMMANDS-"]+['%s: %s'%(key,self.helpstrings[key]) for key in list(self.cmds.keys())]
        #helptext= "\n".join(helpstrings)
        helptext=self.help_str()
        
        print (helptext)
        root = tk.Tk()
        root.lift()
        root.focus_force()
        #root.withdraw()
        w = tk.Label(root, text=helptext)
        w.pack()
        root.mainloop()
        
    switch='010'
        
    plt.close('all')    
    
    if switch[0] != '0':
        # example on using a AbstractImageKey (completely blank class)
        # and add some functions
        pltrnd() #make a plot 
        # create the key commands and 
        a=AbstractImageKey(['c','r'],[plt.clf,pltrnd],['clear figure','plot random image'])
        #a()    
        a.add_key('h',display_help,'display this help')  ##here I want to add a function that
        a()
        plt.title('AbstractImageKey with added c, r, h')
        #plt.ion()
        plt.show()

    if switch[1] != '0':    
        #same with BaseImageKey, here uses standard keys
        plt.figure()
        pltrnd() #make a plot 
        b=BaseImageKey()
        b.add_key('r',pltrnd,'plot random line') 
        b()
        plt.title('BaseImageKey with standard h, added r')
        plt.show()
        
    if switch[2] != '0':
        plt.figure()
        pltrnd() #make a plot 
        c=BaseImageKey()
        c()
        plt.title('BaseImageKey with standard h, r')
        plt.show()        
