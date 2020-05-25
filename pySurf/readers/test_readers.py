"""A partire da test_read_sur, trasformato in test_readers 2020/03/18"""


'''
import unittest
from unittest import TestCase

testfolder=r'G:\\My Drive\\libraries\\python\\userKov3\\pySurf\\test' #for economy during development, hard coded path.

import os
from format_reader import read_sur
import numpy as np
from pySurf.data2D import plot_data

class TestRead_sur(TestCase):
    def test_read_sur(self):
        """called without `raw` flag, return data,x,y"""
        df=os.path.join(testfolder,r'input_data\\profilometer\\04_test_directions\\02_xprof_m_Intensity.sur')
        res,header=read_sur(df)
        print("returned values",res[0].shape,header)
        #print('test method `xAxis` of returned object')
        self.assertEqual(len(res), 3)
        plot_data(res[0],res[1],res[2])
        return res

if __name__ == '__main__':
    unittest.main()
'''

#test without unittest, was being inefficient, these run smoother and are more portable.
testfolder = r'..\test'
#testfolder=r'G:\My Drive\libraries\python\userKov3\pySurf\test' #for economy during development, hard coded path.
#testfolder os.path.dirname(__file__) #to set it as relative to this file path.

import os
#from format_reader import read_sur
import numpy as np
import matplotlib.pyplot as plt
from pySurf.data2D import plot_data

#ported to/from notebook

def test_read_fits(file, raw = False):

    """read a sur file using read_sur_files, that is expected to return a structure
     res.points, .xAxis, .yAxis"""

    res = readsur(file,raw = raw) #in readsur the default is False.
    data,x,y = res.points,res.xAxis,res.yAxis
    del res.points
    del res.xAxis
    del res.yAxis
    header=res #stripped of all data information

    return (data,x,y),header
    
'''	
def csv_points_reader(wfile,*args,**kwargs):
    """Read a processed points file in format x,y,z as csv output of analysis routines."""
    w0=get_points(wfile,*args,**kwargs)
    w=w0.copy()
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return pdata,x,y

def csv_zygo_reader(wfile,*args,**kwargs):
    """Read a processed points file in format x,y,z as csv output of analysis routines."""
    w0=get_points(wfile,*args,**kwargs)
    w=w0.copy()
    x,y=points_find_grid(w,'grid')[1]
    pdata=resample_grid(w,matrix=True)
    return pdata,x,y
'''

if __name__=='__main__':

    # da _instrument_reader: li' va rimossa la registrazione.
    
    """It is based on a tentative generic function read_data accepting among arguments a specific reader.
        The function first calls the data reader, then applies the register_data function to address changes of scale etc.
        This works well, however read_data must filter the keywords for the reader and for the register and
        this is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is
        possible to call the read_data procedure with specific parameters, for example in example below, the reader for
        Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers,
        while this can be done using read_data. """

    #from test_readers import testfolder
    from pySurf.data2D import plot_data, read_data
    
    tests=[[sur_reader,
    os.path.join(testfolder,r'input_data\profilometer\04_test_directions\05_xysurf_pp_Intensity.sur')
    ,{'center':(10,15)}],[points_reader,
    os.path.join(testfolder,r'input_data\profilometer\04_test_directions\05_xysurf_pp_Intensity.txt')
    # questo fallisce, perche' delimiter e' " "  e non e' possibile passare l'argomento al reader
    #os.path.join(testfolder,r'input_data\exemplar_data\scratch\110x110_50x250_100Hz_xyscan_Height_transformed_4in_deltaR.dat')
    ,{'center':(10,15)}],
    [csvZygo_reader,
    os.path.join(testfolder,r'input_data\zygo_data\171212_PCO2_Zygo_data.asc')
    ,{'strip':True,'center':(10,15)}],
    [csvZygo_reader,
    os.path.join(testfolder,r'input_data\zygo_data\171212_PCO2_Zygo_data.asc')
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
        plt.suptitle(os.path.basename(f)+' '+' '.join(["%s=%s"%(k,v) for k,v in o.items()]))
        plt.tight_layout()
        plt.show()
