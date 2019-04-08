import unittest
from unittest import TestCase

testfolder=r'G:\My Drive\libraries\python\userKov3\pySurf\test' #for economy during development, hard coded path.

import os
from format_reader import read_sur
import numpy as np
from pySurf.data2D import plot_data

class TestRead_sur(TestCase):
    def test_read_sur(self):
        """called without `raw` flag, return data,x,y"""
        df=os.path.join(testfolder,r'input_data\profilometer\04_test_directions\02_xprof_m_Intensity.sur')
        res,header=read_sur(df)
        print("returned values",res[0].shape,header)
        #print('test method `xAxis` of returned object')
        self.assertEqual(len(res), 3)
        plot_data(res[0],res[1],res[2])
        return res


if __name__ == '__main__':
    unittest.main()