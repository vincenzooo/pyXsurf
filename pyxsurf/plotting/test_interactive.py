"""experiments how it is possible to advance from a plotting function with 
interactive control as discussed in several sources on the internet."""


from __future__ import print_function
#import matplotlib
#matplotlib.use('GTK3Cairo')
#matplotlib.rcParams['toolbar'] = 'toolmanager'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_tools import ToolBase, ToolToggleBase

class ViewMarkers(ToolToggleBase):
    '''Hide lines with a given gid'''
    default_keymap = 'M'
    description = 'Hide or Show Markers'

    def __init__(self, *args, **kwargs):
        self.gid = kwargs.pop('gid')
        ToolToggleBase.__init__(self, *args, **kwargs)

    def enable(self, *args):
        self.set_lines_visibility(False)

    def disable(self, *args):
        self.set_lines_visibility(True)

    def set_lines_visibility(self, state):
        gr_lines = []
        for ax in self.figure.get_axes():
            for line in ax.markers():
                if line.get_gid() == self.gid:
                    line.set_visible(state)
        self.figure.canvas.draw()

def test_toolbar():

    #plot image and markers

    fig.canvas.manager.toolmanager.add_tool('Hide', GroupHideTool, gid='mygroup')

    # To add a custom tool to the toolbar at specific location inside
    # the navigation group
    fig.canvas.manager.toolbar.add_tool('Hide', 'navigation', 1)        


def main():
    """this works. input from keyboard, can be replaced by?
    infinite loop with check on figure property, generator.
    see e.g. https://github.com/matplotlib/matplotlib/issues/1942/"""
    plt.axis([-50,50,0,10000])
    plt.ion()
    plt.show()

    x = np.arange(-50, 51)
    for pow in range(1,5):   # plot x^1, x^2, ..., x^4
        y = [Xi**pow for Xi in x]
        plt.plot(x, y)
        plt.draw()
        plt.pause(0.01)  #this is to allow drawing. needed?
        input("Press [enter] to continue.")

def main2():

    plt.ion() # enables interactive mode
    plt.plot([1,2,3]) # result shows immediatelly (implicit draw())

    print ('continue computation')

    # at the end call show to ensure window won't close.
    plt.show()

if __name__ == '__main__':
    main2()

    
