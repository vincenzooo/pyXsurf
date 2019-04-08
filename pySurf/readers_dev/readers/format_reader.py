"""  This module contains functions able to read raw data from different formats.
     Returns data and header as a couple, with minimal data processing, functions are about formats, 
     not the specific instrument or specific header, that are handled in calling functions
     (e.g. from instrument_reader). 
	 

2018/09/26 New refactoring of data reader. Vincenzo Cotroneo vcotroneo@cfa.harvard.edu"""


eegKeys = ["FP3", "FP4"]
gyroKeys = ["X", "Y"]

# 'Foo' is ignored
data = {"FP3": 1, "FP4": 2, "X": 3, "Y": 4, "Foo": 5}

filterByKey = lambda keys: {x: data[x] for x in keys}
eegData = filterByKey(eegKeys)
gyroData = filterByKey(gyroKeys)

print(eegData, gyroData) # ({'FP4': 2, 'FP3': 1}, {'Y': 4, 'X': 3})

filterByKey2 = lambda data,keys : {key: data[key] for key in keys if key in data}

print (filterByKey(eegKeys)) # {'FP4': 2, 'FP3': 1}
	
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
