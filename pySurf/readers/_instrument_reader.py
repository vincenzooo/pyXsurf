"""

experimental instrument reader module to provide common interface to several instrument reader
    routines. Functions are made to be imported in intrumentReader (from which functions were originally copied) and not to be
    directly called.
    Routines are written here until a stable interface is reached, after which functions will be moved to instrumentReader.

    2018/09/26 note that these routines shouldn't take *args,**kwargs arguments, unless they need to pass it to another
        function or want to be forgiving of wrong argument passed to them."""


import numpy as np
import os

'''
def fits_reader(wfile,ypix=1.,ytox=1.,header=False):
    """reads a generic matrix from a fits file (no x and y or header information are extracted)."""

    aa=fits.open(wfile)
    head=aa[0].header
    if header: return head

    data=-aa[0].data
    aa.close()

    ny,nx=data.shape
    x=np.arange(nx)*ypix*ytox*nx/(nx-1)
    y=np.arange(ny)*ypix*ny/(ny-1)
    #data.header=head
    return data,x,y
'''

def test_zygo(wfile=None):
    import os
    import matplotlib.pyplot as plt
    from  pySurf.data2D import plot_data
    if wfile is  None:
        relpath=os.path.join(testfolder,r'input_data\zygo_data\171212_PCO2_Zygo_data.asc')
        wfile= os.path.join(os.path.dirname(__file__),relpath)
    (d1,x1,y1)=csvZygo_reader(wfile,ytox=220/1000.,center=(0,0))
    (d2,x2,y2)=csvZygo_reader(wfile,ytox=220/1000.,center=(0,0),intensity=True)
    plt.figure()
    plt.suptitle(relpath)
    plt.subplot(121)
    plt.title('height map')
    plot_data(d1,x1,y1,aspect='equal')
    plt.subplot(122)
    plt.title('continuity map ')
    plot_data(d2,x2,y2,aspect='equal')
    return (d1,x1,y1),(d2,x2,y2)

def test_zygo_binary (wfile=None):
    
    import os
    import matplotlib.pyplot as plt
    from  pySurf.data2D import plot_data
    if wfile is  None:
        relpath=r'/Volumes/GoogleDrive/Il mio Drive/progetti/sandwich/dati/09_02_27/botta di culo 1a.dat'
        wfile= os.path.join(os.path.dirname(__file__),relpath)
        
    d1,x1,y1=datzygo_reader(f)
    
    plt.figure()
    plt.suptitle(relpath)
    plt.title('map')
    plot_data(d1,x1,y1,aspect='equal')
    return (d1,x1,y1)


