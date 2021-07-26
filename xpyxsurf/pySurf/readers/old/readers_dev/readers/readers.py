import numpy as np
from pySurf.data2D import read_data

"""2019/04/09 readers.py
well formed readers from `instrumentReader` using format reader from `_instrument_reader` and
`read_data` from `instrument_reader` (now moving to `data2D`). """

def matrixZygo_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrixZygo_reader
    with calls to read_data(wfile,reader=csvZygo_reader)"""
    from pySurf._instrument_reader import csvZygo_reader
    return read_data(wfile,csvZygo_reader,*args,**kwargs)

def matrix_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrix4D_reader
    with calls to read_data(wfile,reader=matrix4D_reader)"""
    from pySurf.data2D import data_from_txt
    import pdb
    #pdb.set_trace()
    return read_data(wfile,data_from_txt,matrix=True,*args,**kwargs)
   
def matrix4D_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrix4D_reader
    with calls to read_data(wfile,reader=matrix4D_reader)"""
    from _instrument_reader import csv4D_reader
    
    #import pdb
    #pdb.set_trace()
    return read_data(wfile,csv4D_reader,*args,**kwargs)
    
    
def matrixsur_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to matrixsur_reader
    with calls to read_data(wfile,reader=sur_reader)"""
    from _instrument_reader import sur_reader
    return read_data(wfile,sur_reader,*args,**kwargs)

    
def fits_reader(wfile,*args,**kwargs):
    """temporary wrapper for new readers, replace call to fits_reader
    with calls to read_data(wfile,reader=fits_reader)"""
    from _instrument_reader import fits_reader
    return read_data(wfile,fits_reader,*args,**kwargs)

def points_reader(wfile,*args,**kwargs):
    """temporary wrapper for points readers, read and register
    points and convert to 2D data"""
    from _instrument_reader import points_reader
    return read_data(wfile,points_reader,*args,**kwargs)
    

#used by auto_reader to open according to extension
reader_dic={'.asc':matrixZygo_reader,
            '.csv':matrix4D_reader,
            '.fits':fits_reader,
            '.txt':points_reader,
            '.sur':matrixsur_reader,
            '.dat':points_reader}
            
def auto_reader(wfile):
    """guess a reader for wfile. Return reader routine."""
    #2019/04/09 moved to readers.
    ext=os.path.splitext(wfile)[-1]
    try:
        reader=reader_dic[ext]
    except KeyError:
        print ('fileformat ``%s``not recognized for file %s'%(ext,file))
        print ('Use generic text reader')
        reader=points_reader  #generic test reader, replace with asciitables    
    
    return reader