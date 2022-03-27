"""functions to adjust linestyles."""

import matplotlib.pyplot as plt
import numpy as np


def colors_20(plot=False):
    """
    Build a 
    
    Lazily copied from:    https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
    """
    
    NUM_COLORS = 20
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    colors = []
    styles = []
    cm = plt.get_cmap('gist_rainbow')


    for i in range(NUM_COLORS):
        lc = cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS)
        ls = LINE_STYLES[i%NUM_STYLES]
        colors.append(lc)
        styles.append(ls)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = [np.arange(10)*(i+1) for i in range(NUM_COLORS)]
        for i,(lc,ls) in enumerate(zip(colors,styles)):
            lines = ax.plot(data[i])
            lines[0].set_color(lc)
            lines[0].set_linestyle(ls)
        plt.show()
    
    return colors, styles