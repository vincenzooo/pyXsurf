"""
Created on Tue Dec 19 14:12:22 2017
https://gist.github.com/g-s-k/ccffb1e84df065a690e554f4b40cfd3a
@author: gkaplan
"""

# %% package dependencies
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

# %% loader function
def datx2py(file_name):
    # unpack an h5 group into a dict
    def _group2dict(obj):
        return {k: _decode_h5(v) for k, v in zip(obj.keys(), obj.values())}

    # unpack a numpy structured array into a dict
    def _struct2dict(obj):
        names = obj.dtype.names
        return [dict(zip(names, _decode_h5(record))) for record in obj]

    # decode h5py.File object and all of its elements recursively
    def _decode_h5(obj):
        # group -> dict
        if isinstance(obj, h5py.Group):
            d = _group2dict(obj)
            if len(obj.attrs):
                d['attrs'] = _decode_h5(obj.attrs)
            return d
        # attributes -> dict
        elif isinstance(obj, h5py.AttributeManager):
            return _group2dict(obj)
        # dataset -> numpy array if not empty
        elif isinstance(obj, h5py.Dataset):
            d = {'attrs': _decode_h5(obj.attrs)}
            try:
                d['vals'] = obj[()]
            except (OSError, TypeError):
                pass
            return d
        # numpy array -> unpack if possible
        elif isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.number) and obj.shape == (1,):
                return obj[0]
            elif obj.dtype == 'object':
                return _decode_h5([_decode_h5(o) for o in obj])
            elif np.issubdtype(obj.dtype, np.void):
                return _decode_h5(_struct2dict(obj))
            else:
                return obj
        # dimension converter -> dict
        elif isinstance(obj, np.void):
            return _decode_h5([_decode_h5(o) for o in obj])
        # bytes -> str
        elif isinstance(obj, bytes):
            return obj.decode()
        # collection -> unpack if length is 1
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 1:
                return obj[0]
            else:
                return obj
        # other stuff
        else:
            return obj

    # open the file and decode it
    with h5py.File(file_name, 'r') as f:
        h5data = _decode_h5(f)

    return h5data


# %% script body
if __name__ == '__main__':
    # use function above
    h5data = datx2py(sys.argv[1])
    # parse out height values
    zdata = h5data['Data']['Surface']
    zdata = list(zdata.values())[0]
    zvals = zdata['vals']
    zvals[zvals == zdata['attrs']['No Data']] = np.nan
    # get units
    zunit = zdata['attrs']['Z Converter']['BaseUnit']
    # get stats
    pv = np.nanmax(zvals) - np.nanmin(zvals)
    rms = np.sqrt(np.nanmean((zvals - np.nanmean(zvals))**2))
    # display
    plt.pcolormesh(zvals)
    plt.title("{0:0.3f} {2} PV, {1:0.3f} {2} RMS".format(pv, rms, zunit))
	