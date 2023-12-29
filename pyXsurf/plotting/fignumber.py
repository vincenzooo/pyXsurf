import matplotlib.pyplot as plt

def fignumber(fignum=None,*args,**kwargs):
    """ Set and return a figure given a figure number (0 for current, None for new).
    
    If `fignum` is None create a new figure, if 0 uses current figure after cleaning it, if other number, use as number of the figure. Return a `plt.figure` object.
    
    ex. fig=fignumber(0) # clear and use current figure. 

    Offers a way to use a figure number with consistent rules including opening a new window or clearing the old one (?)."""
    
    if fignum == 0:
        fig=plt.gcf()
    else:
        fig=plt.figure(fignum,*args,**kwargs)
    plt.clf()
    
    return fig
    