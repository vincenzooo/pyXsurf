# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
#2021/04/05 Tutorial on basic functions, first attempt of using VScode

get_ipython().run_line_magic('reset', '')
get_ipython().run_line_magic('pylab', '')

import os

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# ## Overview
# 
# `PySurf` library consists in a set of classes and functions reapresenting 2D data and operations on them.
# 
# 

# %%
from pySurf.data2D_class import Data2D


# %%
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# The main class representing 2D data with `x` and `y` axis is `Data2D` object in `pySurf.data2D_class`. `Data2D` can be initialized by providing a matrix of 2-dimensional data and (optionally) `x` and `y` coordinates. Other options can be passed as well.
# 
# The object interface is built on top of a function library in module `pySurf.data2D`: for almnost each method there is a corresponding function that can be called with something like `pySurf.data2D.function(data, x, y, ..)`.
# 
# Similarly, routines operating on profiles (y as a function of x as couples of vector x and y), are contained in class `pyProfile.Profile` and `pyProfile.profile` which have in many points interfaces similar to modules in `pySurf`. 
# 
# Here we will focus on `Data2D` object interface.
# 
# A first way to initialize such an object is by passing directly 2D data, (optionally) coordinates and options.
# 
# 

# %%
nx = 200
ny = 300
data = np.random.random(nx*ny).reshape(ny,nx)
x = np.arange(nx)*25
y = np.arange(ny)*10

D = Data2D(data,x,y,units=['mm','mm','um'])
#D.plot()


# %%
D.plot()

# %%
infolder=r'..\..\test\input_data\4D\180215_C1S06_cut'
fn = '180215_C1S01_RefSub.csv'
file = os.path.join(infolder,fn)

# %%
from pySurf.data2D_class import Data2D
from pySurf.instrumentReader import matrix4D_reader

D = Data2D(file, reader = matrix4D_reader)