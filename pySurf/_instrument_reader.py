"""experimental instrument reader module to provide common interface to several instrument reader
    routines. It is imported from intrumentReader from which functions were originally copied.
    Do not call directly, rather import from instrumentReader.
    Routines are written here until a stable interface is reached, after which functions will be moved to instrumentReader.

    2018/09/26 note that these routines shouldn't take *args,**kwargs arguments, unless they need to pass it to another
        function or want to be forgiving of wrong argument passed to them."""

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from pySurf.points import get_points
from pySurf.points import crop_points
from pySurf.points import points_find_grid
from pySurf.points import resample_grid
from pySurf.read_sur_files import readsur
from pySurf.read_metropro_files import readMetroProData
#from utilities.imaging.man import stripnans
from dataIO.read_pars_from_namelist import read_pars_from_namelist

import pdb
import os


def csv4D_reader(wfile,ypix=None,ytox=1,header=False,delimiter=',',skip_header=12,*args,**kwargs):
    head=read_pars_from_namelist(wfile,': ') #this returns a dictionary, order is lost if header is returned.
    if header:
        return '\n'.join([": ".join((k,v)) for (k,v) in head.items()])+'\n'

    if ypix == None:
        try:
            ypix=np.float(head['xpix'])
        except KeyError:
            ypix=1.

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

    return data,x,y

def points_reader(wfile,*args,**kwargs):
    """Read a processed points file in format x,y,z as csv output of analysis routines."""
    w0=get_points(wfile,*args,**kwargs)
    w=w0.copy()
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return pdata,x,y

def sur_reader(wfile,header=False,*args,**kwargs):
    """read .sur binary files."""
    head=readsur(wfile,*args,**kwargs)
    if header: return head

    data,x,y=head.points,head.xAxis,head.yAxis
    del(head.points,head.xAxis,head.yAxis) #remove data after they are extracted from header to save memory.
    return data,x,y

def datzygo_reader(wfile,header=False,*args,**kwargs):
    """read .dat binary files (MetroPro/Zygo)."""
    d1,head,d3,d4=readMetroProData(wfile,*args,**kwargs)
    if header: return head
    try:
        x0=head['X0']
    except KeyError:
        x0=0
    try:
        y0=head['Y0']
    except KeyError:
        y0=0
    data,x,y=d1,x0+np.arange(np.shape(d1)[1]),y0+np.arange(np.shape(d1)[0])

    return data,x,y

def csvZygo_reader(wfile,intensity=False,header=False,*args,**kwargs):
    """read .csv zygo files.

    Height map is returned unless intensity is set to True to return Intensity map in same format.
    """

    with open(wfile) as myfile:
        raw = myfile.readlines()

    head,d=raw[:15],raw[15:]
    if header: return head

    nx,ny = list(map( int , head[8].split()[:2] ))
    # coordinates of the origin of the
    #connected phase data matrix. They refer to positions in the camera coordinate system.
    #The origin of the camera coordinate system (0,0) is located in the upper left corner of the
    #video monitor.
    origin = list(map(int,head[3].split()[:2]))

    #These integers are the width (columns) and height (rows)
    #of the connected phase data matrix. If no phase data is present, these values are zero.
    connected_size=list(map(int,head[3].split()[2:4]))

    #This real number is the interferometric scale factor. It is the number of
    # waves per fringe as specified by the user
    IntfScaleFactor=float(head[7].split()[1])

    #This real number is the wavelength, in meters, at which the interferogram was measured.
    WavelengthIn=float(head[7].split()[2])

    #This integer indicates the sign of the data. The data sign may be normal (0) or
    #inverted (1).
    DataSign = int(head[10].split()[7])
    #This real number is a phase correction factor required when using a
    #Mirau objective on a microscope. A value of 1.0 indicates no correction factor was
    #required.
    Obliquity = float(head[7].split()[4])

    #This integer indicates the resolution of the phase data points. A value of 0
    #indicates normal resolution, with each fringe represented by 4096 counts. A value of 1
    #indicates high resolution, with each fringe represented by 32768 counts.
    phaseres = int(head[10].split()[0])
    if phaseres==0:
        R = 4096
    elif phaseres==1:
        R = 32768
    else:
        raise ValueError

    #This real number is the lateral resolving power of a camera pixel in
    #meters/pixel. A value of 0 means that the value is unknown
    CameraRes = float(head[7].split()[6])
    if CameraRes==0:
        CameraRes=1.

    #pdb.set_trace()
    ypix=kwargs.pop('ypix',CameraRes)
    ytox=kwargs.pop('ytox',1.)
    zscale=kwargs.pop('zscale',WavelengthIn*1000000.)#original unit is m, convert to um

    datasets=[np.array(aa.split(),dtype=int)  for aa in ' '.join(map(str.strip,d)).split('#')[:-1]]
    #here rough test to plot things
    d1,d2=datasets  #d1 intensity, d2 phase as 1-d arrays
    
    d1,d2=d1.astype(float),d2.astype(float)
    if np.size(d1) > 0: d1 = d1.reshape(ny,nx)
    if np.size(d2) > 0: d2 = d2.reshape(*connected_size[::-1])
    
    d1[d1>65535]=np.nan
    d2[d2>=2147483640]=np.nan
    d2=d2*IntfScaleFactor*Obliquity/R*zscale #in um

    #pdb.set_trace()
    
    dd2=np.zeros([ny,nx])*np.nan #d1 is same size as sensor
    dd2[origin[1]:origin[1]+connected_size[1],origin[0]:origin[0]+connected_size[0]]=d2
    d2=dd2

    #this defines the position of row/columns, starting from
    x=np.arange(nx)*ypix*ytox*nx/(nx-1)
    y=np.arange(ny)*ypix*ny/(ny-1)

    return  (d1,x,y) if intensity else (d2,x,y)

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

def fits_reader(fitsfile,header=False):
    """ Generic fits reader, returns data,x,y.

    header is ignored. If `header` is set to True is returned as dictionary."""

    a=fits.open(fitsfile)
    head=a[0].header
    if header: return head

    data=a[0].data
    a.close()

    x=np.arange(data.shape[1])
    y=np.arange(data.shape[0])

    return data,x,y

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

def test_zygo(wfile=None):
    import os
    import matplotlib.pyplot as plt
    from  pySurf.data2D import plot_data
    if wfile is  None:
        relpath=r'test\input_data\zygo_data\171212_PCO2_Zygo_data.asc'
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
reader_dic={'.asc':csvZygo_reader,
            '.csv':csv4D_reader,
            #'.fits':fitsWFS_reader,
            '.txt':points_reader,
            '.sur':sur_reader,
            '.dat':datzygo_reader}

if __name__=='__main__':
    """It is based on a tentative generic function read_data accepting among arguments a specific reader.
        The function first calls the data reader, then applies the register_data function to address changes of scale etc.
        This works well, however read_data must filter the keywords for the reader and for the register and
        this is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is
        possible to call the read_data procedure with specific parameters, for example in example below, the reader for
        Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers,
        while this can be done using read_data. """

    from pySurf.data2D import plot_data
    tests=[[sur_reader,
    r'test\input_data\profilometer\04_test_directions\05_xysurf_pp_Intensity.sur'
    ,{'center':(10,15)}],[points_reader,
    r'test\input_data\exemplar_data\scratch\110x110_50x250_100Hz_xyscan_Height_transformed_4in_deltaR.dat'
    ,{'center':(10,15)}],
    [csvZygo_reader,
    r'test\input_data\zygo_data\171212_PCO2_Zygo_data.asc'
    ,{'strip':True,'center':(10,15)}],
    [csvZygo_reader,
    r'test\input_data\zygo_data\171212_PCO2_Zygo_data.asc'
    ,{'strip':True,'center':(10,15),'intensity':True}]]

    plt.ion()
    plt.close('all')
    for r,f,o in tests:  #reader,file,options
        print ('reading data %s'%os.path.basename(f))
        plt.figure()
        plt.subplot(121)
        plot_data(*r(f))
        plt.title('raw data')
        plt.subplot(122)
        data,x,y=read_data(f,r,**o)
        plot_data(data,x,y)
        plt.title('registered')
        plt.suptitle(' '.join(["%s=%s"%(k,v) for k,v in o.items()]))
        plt.tight_layout()
        plt.show()
