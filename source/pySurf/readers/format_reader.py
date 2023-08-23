# -*- coding: utf-8 -*-

"""  This module contains functions able to read raw data from different formats. With default arguments returns data,x,y; return header if `header` is set to True. 
    Provides common interface to several instrument reader
    routines. Functions are made to be imported in intrumentReader (from which functions were originally copied) and not to be
    directly called.

2020/05/25 Moved here csv4D_reader

2020/03/18 Transfer here functions from _instrument_reader.

2018/09/26 note that these routines shouldn't take *args,**kwargs arguments, unless they need to pass it to another
    function or want to be forgiving of wrong argument passed to them.
    
 Returns data and header from file, with minimal data
     
2018/09/26 New refactoring of data reader. Vincenzo Cotroneo vcotroneo@cfa.harvard.edu"""

import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.io import fits

from pySurf.points import get_points
from pySurf.points import crop_points
from pySurf.points import points_find_grid
from pySurf.points import resample_grid
from pySurf.readers.read_sur_files import readsur
from pySurf.readers.read_metropro_files import readMetroProData
#from utilities.imaging.man import stripnans
from pySurf.data2D import data_from_txt

from dataIO.read_pars_from_namelist import read_pars_from_namelist
from dataIO.fn_add_subfix import fn_add_subfix
from IPython.display import display

from .nid_reader import make_channel_tags, read_datablock

import pdb


def points_reader(wfile,header=False,*args,**kwargs):
    """Read a processed points file in format x,y,z as csv output of analysis routines."""
    if header: return None
    w0=get_points(wfile,*args,**kwargs)
    w=w0.copy()
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return pdata,x,y

def datzygo_reader(wfile,header=False,*args,**kwargs):
    """read .dat binary files (MetroPro/Zygo)."""
    
    '''from manual: Complete maps of the header formats can be obtained by running the dat_test.exe 
program that is included with MetroPro in folder r"C:\MetroPro\Bin". Type the following at 
a command prompt: 
dat_test  –maps  >  maps.txt 
File maps.txt will contain the output. '''

    d1,head,d3,d4=readMetroProData(wfile,*args,**kwargs)
    #pdb.set_trace()
    if header or kwargs.get('header',False): return head
    try:
        x0=head['X0']
    except KeyError:
        x0=0
    try:
        y0=head['Y0']
    except KeyError:
        y0=0
    try:
        xyscale=head['cameraRes']
    except KeyError:
        xyscale=1.
    # pdb.set_trace()
    #data,x,y=d1,x0+np.arange(np.shape(d1)[1]),y0+np.arange(np.shape(d1)[0])  
    # data, x, y are already in the appropriate units, but xy offsets were not included.
    # units are meters.
    data,x,y=d1,x0*xyscale+d3,y0*xyscale+d4

    return data,x,y


def csv4D_reader(wfile,ypix=None,ytox=None,header=False,delimiter=',',endline=True,skip_header=12,*args,**kwargs):
    """read csv data in 4sight 4D format.
    12 lines header with info in namelist format, uses `xpix`, `aspect` and `wavelength` if available.
    Note that standard csv format ends line with `,` which adds an extra column.
    This is automatically removed if `endline` is set to True.
    Note also that typically instruments invert y axis, but this is not implemented at the moment. 
    TODO: change y scale after code reorganization.
    2020/07/14 read data directly with `np.genfromtxt`,
    rather than uselessely launching the wrapper `data2D.data_from_txt`."""

    head=read_pars_from_namelist(wfile,': ') #this returns a dictionary, order is lost if header is returned.
    if header:
        return '\n'.join([": ".join((k,v)) for (k,v) in head.items()])+'\n'

    if ypix == None:
        try:
            ypix=float(head['xpix'])
        except KeyError:
            ypix=1.

    if ytox == None:
        try:
            ytox=float(head['aspect'])
        except KeyError:
            ytox=1.

    try:
        zscale=float(head['wavelength'])
    except KeyError:
        zscale=1.
    
    #from pySurf.data2D import data_from_txt
    #data=np.genfromtxt(wfile,delimiter=delimiter,skip_header=12)
    #data=data_from_txt(wfile,delimiter=delimiter,skip_header=skip_header)[0]
    data=np.genfromtxt(wfile,skip_header=skip_header,delimiter=delimiter)
    #pdb.set_trace()
    #this defines the position of row/columns, starting from
    # commented 2018/08/28 x and y read directly
    if endline:
        data=data[:,:-1]
    ny,nx=data.shape
    x=np.arange(nx)*ypix*ytox*nx/(nx-1)
    y=np.arange(ny)*ypix*ny/(ny-1)
    data=data*zscale
    #data.header=head


    return data,x,y


def csvZygo_reader(wfile,intensity=False,header=False,xyz=False,*args,**kwargs):

    """read .csv zygo files (and .xyz).

    Height map is returned unless intensity is set to True to return Intensity map in same format.
    Metadata are reaad from header, some can be overridden by
      arguments in kwargs:
      
      ypix defaults to CameraRes
      ytox defaults to 1. 
      zscale defaults to WavelengthIn*1000000. #original unit is m, convert to um
    
    ### XYZ format:
    XYZ Data File Connected Phase Data
    The data in this section is organized by phase origin. Each line contains three pieces of
    data. The first two columns contain the column (y) and row (x) location of the data,
    beginning at the phase origin. The third number on the line can be either the character
    string "No Data" or a floating-point number corresponding to the measurement in
    microns. To convert these measurements to ‘zygos’, use the following formula. (The
    names in parenthesis refer to the Binary Data Format field names)
    The data in the file is in microns. To convert to meters:
    For Low resolution (0) phase data:
    Multiply third column data in by:
     (4096/ (IntfScaleFactor*ObliquityFactor*WavelengthIn*1000000))
    For High resolution (1) phase data:
    Multiply third column data by:
     (32768/ (IntfScaleFactor*ObliquityFactor*WavelengthIn*1000000))
    The PhaseRes is the first value of the eleventh line of the header. Convert the result to an
    integer to be stored in the binary file. This will cause round-off error, but amounts to a
    less than one Angstrom variance from the original data. 

    ### ASCII data format:
    This section describes the format of a MetroPro ASCII data file. The file is made up of
    three parts: header, intensity data, and phase data. Each part is followed by a line
    containing a sharp (#) character. At least one of the data sets must be present. A
    MetroPro ASCII data file is created by using the dat_to_asc conversion utility. 
    
    -----------------
    
    ASCII Data File Intensity Data
    Each data point is an integer. The data is written 10 data points per line in row-major
    order. Acceptable values are from 0 to the value specified in IntensRange. An invalid
    point is indicated by a value >= 65535. A line containing only a sharp character (#) is
    output after the data. The number of intensity data points is:
    IntensWidth * IntensHeight * NBuckets
    ASCII Data File Connected Phase Data
    Each data point is an integer. The data is written 10 data points per line in row-major
    order. Acceptable values are in the range from -2097152 to +2097151. An invalid point
    is indicated by a value >= 2147483640. A line containing only a sharp character (#) is
    output after the data. The number of connected phase data points is:
    PhaseWidth * PhaseHeight
    The phase data points are in internal units representing a scaled number of fringes. To
    convert a value to waves, multiple by (S * O)/R.
    Where: S = IntfScaleFactor, O
    
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
    if xyz:
        zscale=kwargs.pop('zscale',1.) # assume already in um    
    else:
        zscale=kwargs.pop('zscale',WavelengthIn*1000000.)#original unit is m, convert to um

    #pdb.set_trace()
    if xyz:
        tmp = [dd.replace('No Data','nan') for dd in d[:-1]]
        tmp = np.array([l.split() for l in tmp],dtype='float')
        from pySurf.points import resample_grid
        tmp = resample_grid(tmp,matrix=True)
        datasets=[np.array([]),tmp]
        #return datasets[-1]
    else:
        datasets=[np.array(aa.split(),dtype=int)  for aa in ' '.join(map(str.strip,d)).split('#')[:-1]]
    #here rough test to plot things
    d1,d2=datasets  #d1 intensity, d2 phase as 1-d arrays
    
    d1,d2=d1.astype(float),d2.astype(float)
    if np.size(d1) > 0: d1 = d1.reshape(ny,nx)
    if np.size(d2) > 0: d2 = d2.reshape(*connected_size[::-1])
    
        
    # datasets=[np.array(aa.split(),dtype=int)  for aa in ' '.join(map(str.strip,d)).split('#')[:-1]]
    # #here rough test to plot things
    # d1,d2=datasets  #d1 intensity, d2 phase
    # d1,d2=d1.astype(float).reshape(ny,nx),d2.astype(float).reshape(*connected_size[::-1])
    # d1[d1>65535]=np.nan
    # d2[d2>=2147483640]=np.nan
    # d2=d2*IntfScaleFactor*Obliquity/R*zscale #in um
    
    d1[d1>65535]=np.nan
    if not xyz:
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

from .read_sur_files import readsur

def sur_reader(wfile,header=False,*args,**kwargs):
    """read .sur binary files."""
    head=readsur(wfile,*args,**kwargs)
    if header: return head

    data,x,y=head.points,head.xAxis,head.yAxis
    del(head.points,head.xAxis,head.yAxis) #remove data after they are extracted from header to save memory.
    return data,x.flatten(),y.flatten()   #are returned as column vector  
        
def read_nid(file_name,index=0,header=False,read_tags=False):
    """read a file nid. Return a list of `data,x,y` for the scans of index in `index`.
    
    `data,x,y` are extracted from files and adjusted by using 
     info from header. A file nid can contains several "scans" 
     (typically different AFM data: phase, amplitude, etc., having different
     units). This function extracts a single scan.
     
    Metadata have a hyerarchical structure, with first level general settings,
    then subfields for single scans
    if `header` is set, the full file metadata are returned "raw" as list of string,
    it is left to user to extract the subset of metadata corresponding to the 
    extracted data.
    A convenient way to explore the metadata is to convert them to a 
    configparser object. This is already implemented in this routine, where
    values should already include settings and conversions from metadata.
    TODO: consider extracting only relevant metadata (if possible/convenient). 
    
    `read_tags` if set, keys for channels with data are returned (to return also the empty ones call `make_channel_tags`)
    
    Of the old primitive routine `nid_reader.read_nid` and 
    `nid_reader.read_raw_nid`:
    `read_raw_nid` reads `header` as list of strings and `data` as
    single binary block. `read_datablock` is used to extract an image
    from the datablock. `dataIO.config.string_to_config` can be used by converting it to `config`
    object and reading fields."""
    
    from pySurf.readers.nid_reader import read_raw_nid  #,read_nid
    from dataIO.config.make_config import string_to_config
    import logging
     
    
    # all columns of the matrix 
    #ngroups = config.get('DataSet','GroupCount') #number of groups 
    imgdic=[]

    logging.info('reading '+file_name)
    #print('reading '+file_name)
    meta, data = read_raw_nid(file_name)
    if header: return meta
    
    # build a config object `config` by merging the string.
    config = string_to_config(meta)        
    # create itag, a (ordered) list of frame keys, only for frames with data
    itag = make_channel_tags(meta)
    hasdata = [config.get('DataSet',t,fallback=None) is not None for t in itag]
    itag = [i for (i, hd) in zip(itag, hasdata) if hd]
    
    if np.ndim(index) == 0:
        if read_tags:  #return tags of subscans with data.
            if index == 0:
                return itag
        else:
            index=[index]
    if read_tags: return [itag[i] for i in index]
                    
    for i in index:
        #breakpoint()
        cgtag = itag [i]   
        # all raws of the actual column   
        #pdb.set_trace() 
        #print(cgtag)
        logging.info('tag '+cgtag)
        try:
            datatag = config.get('DataSet',cgtag)
            # then read the header, specially �Points� and �Lines� 
            #ReadDataSetInfo(datatag)    
            npoints = int(config.get(datatag,'Points'))
            nlines = int(config.get(datatag,'Lines' ))
            nbits = int(config.get(datatag,'SaveBits'))
            sign = config.get(datatag,'SaveSign')
            if (int(nbits) != 32) or sign !='Signed':
                raise ValueError
            else:
                fmt = '<l'    
            img = read_datablock(data, npoints, nlines, nbits)
            
            #npoints nlines might be inverted
            Dim0Range = float(config.get(datatag,'Dim0Range'))
            Dim0Min = float(config.get(datatag,'Dim0Min'))
            x = np.arange(npoints) / (npoints-1)  * Dim0Range + Dim0Min
            
            Dim1Range = float(config.get(datatag,'Dim1Range'))
            Dim1Min = float(config.get(datatag,'Dim1Min'))
            y = np.arange(nlines) / (nlines-1) * Dim1Range + Dim1Min
            
            Dim2Range = float(config.get(datatag,'Dim2Range'))
            Dim2Min = float(config.get(datatag,'Dim2Min'))
            img = (img + 2**(nbits-1)) / (2**nbits-1) * Dim2Range + Dim2Min
            # z_value = (z_data + 2^(SaveBits-1)) / (2^SaveBits-1)  * Dim2Range + Dim2Min
            imgdic.append([img,x,y])
            units = [config.get(datatag,'Dim0Unit'),
                        config.get(datatag,'Dim1Unit'),
                        config.get(datatag,'Dim2Unit')]
            
            #if units != ['m','m','m']: 
                #raise NotImplementedError('unknown units in read_nid: ',units)
            #print(units)
            logging.info('units '+units)
            #print(points)
            #ReadBinData()
            logging.info(cgtag+' read')
            #print (cgtag+' read')
        except:
            
            #breakpoint()
            logging.info('option '+cgtag+' not found')
            #print('option '+cgtag+' not found')
            pass
        #breakpoint()
    if len (imgdic)==1: imgdic=imgdic[0] 
        
    return imgdic

def nid_reader(file,index=0,header=False,*args,**kwargs):
    """read .sur binary files. Incompleta."""
    
    from pySurf.readers.nid_reader import read_raw_nid  #,read_nid
    from dataIO.config.make_config import string_to_config
    meta = read_raw_nid(file)[0]
    if header: return meta

    config = string_to_config(meta)
            
    ddic = read_nid(file,*args,**kwargs) # head['Gr0-Ch1']
    #data,x,y = 
    
    return data,x,y

   
def nid_reader2(wfile,header=False):
    """read .nid binary files."""
    head=read_nid(wfile)
    if header: return head

    data,x,y=head['Gr0-Ch1']
    return data,x,y

def read_sur(file,header=False):
    """Convert `res` object returned by `readsur` reader to data,x,y"""
    res = readsur(file,raw = False)
    data,x,y = res.points,res.xAxis,res.yAxis
    del res.points
    del res.xAxis
    del res.yAxis
    return None if header else (data,x.flatten(),y.flatten())

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
    
    
def test_read_nid(file_name=None):
    from pySurf.data2D_class import Data2D
    from matplotlib import pyplot as plt
    if file_name is None:
        file_name=r'..\test\input_data\AFM\02_test.nid'
    
    datadic = read_nid(file_name)
    data,x,y = datadic['Gr0-Ch1']
    print("read data of shape",data.shape)
    
    d = Data2D(data,x,y,units=['mm','mm','um'],scale=[1000.,1000.,1000000.]) 

    d.plot()
    plt.show()
    return d

def test_reader(file,reader,outfolder=None,infolder=None,**kwargs):
    """called without `raw` flag, return data,x,y. Infolder is taken from file or can be passed (e.g. to point to local test data)."""
    
    import os
    import matplotlib.pyplot as plt
    from  pySurf.data2D import plot_data
    
    if infolder is None:
        infolder=os.path.dirname(file) 
    
    df=os.path.join(infolder,os.path.basename(file))
    res=reader(df,**kwargs)
    header = reader(df,header=True,**kwargs)
    print("reading file %s with reader %s"%(df,reader))
    print("returned values",[r.shape for r in res],header)
    
    plot_data(res[0],res[1],res[2])
    plt.title(os.path.basename(file)+' '+' '.join(["%s=%s"%(k,v) for k,v in kwargs.items()]))
    if outfolder is not None:
        if outfolder == "" : 
            display(plt.gcf()) 
        else: 
            outname=os.path.join(infolder,outfolder,os.path.basename(df))
            os.makedirs(os.path.dirname(outname),exist_ok=True)
            plt.savefig(fn_add_subfix(outname,'','.png'))
    return res,header
    
#used by auto_reader to open according to extension
reader_dic={#'.asc':csvZygo_reader,
            '.csv':csv4D_reader,
            #'.fits':fitsWFS_reader,
            #'.txt':points_reader,
            '.sur':sur_reader,
            '.dat':datzygo_reader,
            '.nid':read_nid}

def auto_reader(wfile):
    """guess a reader for wfile. Return reader routine."""
    ext=os.path.splitext(wfile)[-1]
    try:
        reader=reader_dic[ext]
    except KeyError:
        print ('fileformat ``%s``not recognized for file %s'%(ext,wfile))
        print ('Use generic text reader')
        reader=data_from_txt
        #points_reader  #generic test reader, replace with asciitables

    return reader


if __name__=='__main__':
    """It is based on a tentative generic function read_data accepting among arguments a specific reader.
        The function first calls the data reader, then applies the register_data function to address changes of scale etc.
        This works well, however read_data must filter the keywords for the reader and for the register and
        this is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is
        possible to call the read_data procedure with specific parameters, for example in example below, the reader for
        Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers,
        while this can be done using read_data. """

    from .test_readers import testfolder, testfiles
    from pySurf.data2D import plot_data, read_data
    
    tests=[[sur_reader,
    os.path.join(testfolder,r'input_data\profilometer\04_test_directions\05_xysurf_pp_Intensity.sur')
    #,{'center':(10,15)}],[points_reader,
    ,{}],[points_reader,
    os.path.join(testfolder,r'input_data\profilometer\04_test_directions\05_xysurf_pp_Intensity.txt')
    # questo fallisce, perche' delimiter e' " "  e non e' possibile passare l'argomento al reader
    #os.path.join(testfolder,r'input_data\exemplar_data\scratch\110x110_50x250_100Hz_xyscan_Height_transformed_4in_deltaR.dat')
    #,{'center':(10,15)}
    ,{}],
    [csvZygo_reader,
    os.path.join(testfolder,r'input_data\zygo_data\171212_PCO2_Zygo_data.asc')
    ,{'strip':True}
    #,'center':(10,15)}
    ],
    [csvZygo_reader,
    os.path.join(testfolder,r'input_data\zygo_data\171212_PCO2_Zygo_data.asc')
    ,{'intensity':True}]]
    #,{'strip':True,'center':(10,15),'intensity':True}]]

    plt.ion()
    plt.close('all')
    for r,f,o in tests:  #reader,file,options
        plt.figure()
        test_reader(f,r,**o)    


    '''
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
        plt.suptitle(os.path.basename(f)+' '+' '.join(["%s=%s"%(k,v) for k,v in o.items()]))
        plt.tight_layout()
        plt.show()
    '''