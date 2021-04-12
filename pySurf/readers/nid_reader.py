"""Set of functions to read .nid AFM data.
This is a first prototype, format is not standard."""

import numpy as npoints
from configparser import ConfigParser
import io
import struct
import numpy as np

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

def read_nid(file_name):

    header, data = read_raw_nid(file_name) 
    from configparser import NoOptionError
    config = ConfigParser()
    buf = io.StringIO("\n".join(header))
    config.read_file(buf)
    ng = config.get('DataSet','GroupCount')

    # all columns of the matrix 
    ngroups = config.get('DataSet','GroupCount') #number of groups 
    imgdic={}
    i=0
    for g in range(int(ngroups)):    
        grcount = config.get('DataSet','Gr%i-Count'%g )
        # all raws of the actual column 
        for c in range(int(grcount)):     
            cgtag = 'Gr%i-Ch%i'%(g,c)
            try:
                datatag = config.get('DataSet',cgtag)
                # then read the header, specially “Points” and “Lines” 
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

                #print(points)
                #ReadBinData()
            except NoOptionError:
                #print('option '+cgtag+' not found')
                pass
            
    return imgdic