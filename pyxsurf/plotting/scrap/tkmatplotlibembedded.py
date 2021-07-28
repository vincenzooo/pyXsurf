"""
Created on Wed Jun 04 17:56:03 2014
 
@author: Sukhbinder Singh
 
 
 
"""
 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import numpy as np
 
 
import Tkinter as Tk
 
  
class Window():
    def __init__(self, master):
        self.frame = Tk.Frame(master)
        self.f = Figure( figsize=(10, 9), dpi=80 )
        self.ax0 = self.f.add_axes( (0.05, .05, .50, .50), axisbg=(.75,.75,.75), frameon=False)
        self.ax1 = self.f.add_axes( (0.05, .55, .90, .45), axisbg=(.75,.75,.75), frameon=False)
        self.ax2 = self.f.add_axes( (0.55, .05, .50, .50), axisbg=(.75,.75,.75), frameon=False)
 
     
        self.ax0.set_xlabel( 'Time (s)' )
        self.ax0.set_ylabel( 'Frequency (Hz)' )
        self.ax0.plot(np.max(np.random.rand(100,10)*10,axis=1),"r-")
        self.ax1.plot(np.max(np.random.rand(100,10)*10,axis=1),"g-")
        self.ax2.pie(np.random.randn(10)*100)
 
          
        self.frame = Tk.Frame( root )
        self.frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
 
        self.canvas = FigureCanvasTkAgg(self.f, master=self.frame)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.show()
     
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.frame )
        self.toolbar.pack()
        self.toolbar.update()
 
 
if __name__ == '__main__':
    root = Tk.Tk()
    app = Window(root)
    root.title( "MatplotLib with Tkinter" )
    root.update()
    root.deiconify()
    root.mainloop()