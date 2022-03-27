
import numpy as np

def read_blocks(filename):
    """Read a file containing blocks of numerical data separated by a white line.
    A dictionary is returned in which the first line of each block is used as a key and the following lines
    are assumed to be matrices of data with data on the same line separated by comma.
    e.g.:

    <Block name on this line (data in following lines)>
    351.08931,-165.07514,-691.24115
    352.28775,-125.48324,-691.24096
    353.64744,-78.56968,-691.37016
    355.13786,-30.21684,-691.3701
    """
    
    f = open(filename, 'r')
    content = f.read()
    f.close()
    pDic={}
    for block in content.split('\n\n'):
        if (block.strip() != ''):
            """build a dic with a set of points, first line is used as key"""
            lines=block.split('\n')
            pDic[lines[0]]=np.array([l.split(',') for l in lines[1:]],dtype=float)
    f.close()
    return pDic
