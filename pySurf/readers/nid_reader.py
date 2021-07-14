"""Set of functions to read .nid AFM data.
This is a first prototype, format is not standard."""

import numpy as npoints
from configparser import ConfigParser
import io
import struct
import numpy as np
import logging
from configparser import NoOptionError
from dataIO.config.make_config import string_to_config


""" dataset info     
Version=2
Points=256
Lines=256
Frame=Scan forward
CurLine=255
Dim0Name=X*
Dim0Unit=m
Dim0Range=1e-05
Dim0Min=0
Dim1Name=Y*
Dim1Unit=m
Dim1Range=1e-05
Dim1Min=0
Dim2Name=Z-Axis
Dim2Unit=m
Dim2Range=1.04e-05
Dim2Min=-5.2e-06
SaveMode=Binary
SaveBits=32
SaveSign=Signed
SaveOrder=Intel
"""

def read_raw_nid (file_name):
    """return header and raw data (all data blocks together) from file_name"""
   
    with open(file_name, 'rb') as binfile:
        lines = binfile.readlines()

    header=[]
    for i,l in enumerate(lines):
        text = l.decode('ansi')
        if '#!' in text:
            break
        header.append(text.strip())

    with open(file_name, 'rb') as binfile:
        a=binfile.read()
    i = a.find(b'#!')

    data = a[i+2:]
    
    return header,data

def read_datablock(data, npoints, nlines, nbits, nim=0, fmt = '<l'):
    """read image of index nim from binary data.
    
    nbits is redundant, can be obtained from fmt.
    fmt is c-style format string,
    see https://docs.python.org/3.5/library/struct.html#struct-format-strings for other formats
    """
    if nbits%8 != 0: raise ValueError
    imsize=npoints*nlines*nbits//8
    buf2 = io.BytesIO(data[nim*imsize:(nim+1)*imsize])
    idata = struct.iter_unpack(fmt,buf2.read())
    a=list(idata)
    img = np.array(a).reshape((nlines,npoints))
    return img

def make_channel_tags(meta):
    """ builds list of fixed format string with channel identifiers.
    results can be used to extract single channel data and metadata, e.g. with:
       
    """
    config = string_to_config(meta)
    # create itag, a (ordered) list of frame keys
    itag=[]
    ngroups = config.get('DataSet','GroupCount')
    for g in range(int(ngroups)):    
        grcount = config.get('DataSet','Gr%i-Count'%g )   
        for c in range(1,int(grcount)+1):     
            cgtag = 'Gr%i-Ch%i'%(g,c)    
            itag.append(cgtag)
    return itag
        

def read_nid(file_name):
    """ 2021/07/02 OBSOLETE: This is function used in ICSO2020.
    It is replaced with standardize version in `format_reader`.
    To convert the old calls to the new:
     
    
    read a file nid. Return a dictionary with keys `'Gr%i-Ch%i'%(g,c)`
    and `data,x,y` as values, one for each image included in nid file.
    Info from header are used to adjust scales of data.
    
    `read_raw_nid` reads `header` as list of strings and `data` as
    single binary block. `read_datablock` is used to extract an image
    from the datablock. `header` can be used by converting it to `config`
    object and reading fields."""
    
    print ("Obsolete, replace read_nid with version in format_reader")
    header, data = read_raw_nid(file_name) 
    
    # build a config object `config` by merging the string.
    from configparser import NoOptionError
    from dataIO.config.make_config import string_to_config
    config = string_to_config(header)
    
    # all columns of the matrix 
    ngroups = config.get('DataSet','GroupCount') #number of groups 
    imgdic={}
    i=0
    logging.info('reading '+file_name)
    #print('reading '+file_name)
    for g in range(int(ngroups)):    
        grcount = config.get('DataSet','Gr%i-Count'%g )
        # all raws of the actual column 
        for c in range(int(grcount)):     
            cgtag = 'Gr%i-Ch%i'%(g,c)
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
                imgdic[cgtag] = [img,x,y]
                i=i+1
                units = [config.get(datatag,'Dim0Unit'),
                         config.get(datatag,'Dim1Unit'),
                         config.get(datatag,'Dim2Unit')]
                
                #if units != ['m','m','m']: 
                    #raise NotImplementedError('unknown units in read_nid: ',units)
                print(units)
                
                #print(points)
                #ReadBinData()
                logging.info(cgtag+' read')
                #print (cgtag+' read')
            except NoOptionError:
                logging.info('option '+cgtag+' not found')
                #print('option '+cgtag+' not found')
                pass
            
    return imgdic

if __name__ == "__main__":
    
    import os 
    from pySurf.data2D_class import Data2D
    
    
    datafolder = r'G:\My Drive\progetti\c_overcoating\esperimenti\20210129_dopamine\20210224_dopamine_clean'
    fn = 'Image00053.nid'
    file_name =  os.path.join(datafolder,fn)

    #file_name=r'C:\Users\kovor\Documents\python\pyXTel\pySurf\test\input_data\AFM\02_test.nid'

    datadic = read_nid(file_name)
    data,x,y = datadic['Gr0-Ch1']
    print("read data of shape",data.shape)
    
    d = Data2D(data,x,y,units=['mm','mm','um'],scale=[1000.,1000.,1000000.]) 

    d.plot()
    plt.show()