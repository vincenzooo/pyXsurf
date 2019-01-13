import matplotlib.pyplot as plt

def fignumber(fignum=None,*args,**kwargs):
    """ passed a figure number, if None create a new figure, if 0 uses current figure
    after cleaning, if other number, use as number of the figure.
    ex. fig=fignumber(0)
    
    offers a way to get a figure number with consistent rules including opening a new window or clearing the old one."""
    
    if fignum == 0:
        fig=plt.gcf()
    else:
        fig=plt.figure(fignum,*args,**kwargs)
    plt.clf()
    
    return fig
    