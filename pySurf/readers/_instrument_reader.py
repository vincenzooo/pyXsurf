"""

experimental instrument reader module to provide common interface to several instrument reader
    routines. Functions are made to be imported in intrumentReader (from which functions were originally copied) and not to be
    directly called.
    Routines are written here until a stable interface is reached, after which functions will be moved to instrumentReader.

    2018/09/26 note that these routines shouldn't take *args,**kwargs arguments, unless they need to pass it to another
        function or want to be forgiving of wrong argument passed to them."""

from dataIO.read_pars_from_namelist import read_pars_from_namelist
import numpy as np

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

def csv4D_reader(wfile,ypix=None,ytox=None,header=False,delimiter=',',endline=True,skip_header=12,*args,**kwargs):
    """read csv data in 4sight 4D format.
    12 lines header with info in namelist format, uses `xpix`, `aspect` and `wavelength` if available.
    Note that standard csv format ends line with `,` which adds an extra column.
    This is automatically removed if `endline` is set to True."""

    head=read_pars_from_namelist(wfile,': ') #this returns a dictionary, order is lost if header is returned.
    if header:
        return '\n'.join([": ".join((k,v)) for (k,v) in head.items()])+'\n'

    if ypix == None:
        try:
            ypix=np.float(head['xpix'])
        except KeyError:
            ypix=1.

    if ytox == None:
        try:
            ytox=np.float(head['aspect'])
        except KeyError:
            ytox=1.

    try:
        zscale=np.float(head['wavelength'])
    except KeyError:
        zscale=1.
    from pySurf.data2D import data_from_txt
    #data=np.genfromtxt(wfile,delimiter=delimiter,skip_header=12)
    data=data_from_txt(wfile,delimiter=delimiter,skip_header=skip_header)[0]

    #this defines the position of row/columns, starting from
    # commented 2018/08/28 x and y read directly
    ny,nx=data.shape
    x=np.arange(nx)*ypix*ytox*nx/(nx-1)
    y=np.arange(ny)*ypix*ny/(ny-1)
    data=data*zscale
    #data.header=head
    if endline:
        data=data[:,:-1]

    return data,x,y

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

#used by auto_reader to open according to extension
reader_dic={#'.asc':csvZygo_reader,
            '.csv':csv4D_reader} #,
            #'.fits':fitsWFS_reader,
            #'.txt':points_reader,
            #'.sur':sur_reader,
            #'.dat':datzygo_reader}

def auto_reader(wfile):
    """guess a reader for wfile. Return reader routine."""
    ext=os.path.splitext(wfile)[-1]
    try:
        reader=reader_dic[ext]
    except KeyError:
        print ('fileformat ``%s``not recognized for file %s'%(ext,file))
        print ('Use generic text reader')
        reader=points_reader  #generic test reader, replace with asciitables

    return reader


