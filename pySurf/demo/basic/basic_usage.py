#!/usr/bin/env python
# coding: utf-8


# In[14]:
from pySurf.psd2d import psd2d, plot_psd2d, avgpsd2d
from pySurf.data2D import crop_data
from pySurf.psd2d import calculatePSD
from pySurf.instrumentReader import matrix4D_reader
from pySurf.data2D_class import Data2D
from plotting.multiplots import compare_images
from IPython import get_ipython

# 2021/04/05 Tutorial on basic functions, first attempt of using VScode

get_ipython().run_line_magic('reset', '')
get_ipython().run_line_magic('pylab', '')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Introduction
#
# `PySurf` library consists in a set of classes and functions representing
# 2D data and operations on them.
#

# In[34]:


# The main class representing 2D data with `x` and `y` axis is `Data2D` object in `pySurf.data2D_class`. `Data2D` can be initialized by providing a matrix of 2-dimensional data and (optionally) `x` and `y` coordinates. Other options can be passed as well.
#
#

# In[39]:


nx = 200
ny = 300
data = np.random.random(nx*ny).reshape(ny, nx)
x = np.arange(nx)*10
y = np.arange(ny)*25

D = Data2D(data, x, y, units=['mm', 'mm', 'um'])
D.plot()


# Functions for reading common formats of 2D data are collected in `pySurf.readers` module. The structure and interface of readers is described elsewhere, a reader is essentially a function able to obtain `data, x, y` from a data file, however if the interface is correctly implemented,

# In[ ]:


infolder = r'..\..\test\input_data\4D\180215_C1S06_cut'


# In[19]:


fn = '180215_C1S01_RefSub.csv'


# In[22]:


data, x, y = matrix4D_reader(os.path.join(infolder, fn))


# In[25]:


# In[28]:


D = Data2D(data, x, y, strip=True)


# In[29]:


D.plot()


# In[30]:


D = Data2D(os.path.join(infolder, fn), strip=True, reader=matrix4D_reader)


# In[31]:


D.plot()


# In[33]:


D.level((4, 2)).plot()


# In[ ]:
