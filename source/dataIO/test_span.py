#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2019/04/01 rerun this analysis that was made in days after measurement 19/03/07
#  and  not completed because of issues with python.
# C1S04 was measured in three occasions, this is the first one:
# 2019/03/07  07_ steps 25x1000, 400 Hz  might have moved on support
# 2019/07/11  14_ steps 25x1000, 400 Hz looks good
# 2019/07/11  19_ steps 25x250, 400 Hz  higher sampling

#---
#---
#2018/08/18 C1S05 after compensating layer IrC 
#   this first set of data have some displacement in corners,
#   a second set of data is reacquired on 09/19 with a better correspondance.

#2018/08/07 from C1S05_PZT_0523_analysis

# This is remake with PZT sample after it was sent back to SAO because all previous coating studies
#  were made on the uncoated sample. Since then it was coated with gold and measured today.
# 
from __future__ import print_function
get_ipython().run_line_magic('reset', '')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('pylab', '')


# In[2]:


from pySurf.psd2d import calculatePSD #,plotPSDs,PSDplot
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import span




#gp=get_points(df,scale=gscale,center=(0,0))


pts1=get_points(fn_add_subfix(datafile,'_shape','.dat'),delimiter=' ')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

pts.shape


# In[174]:


a=np.array(np.arange(12).reshape((3,4)))
print('a, shape ',a.shape,':\n',a)

aa=a+np.array([float('.%02i'%i) for i in np.arange(12)]).reshape(a.shape)
print('aa, shape ',aa.shape,':\n',aa)

#this concatenates arrays on an arbitrary axis, defaults to 0
b=np.stack([aa,aa+100])
print('b, shape ',b.shape,':\n',b)


# In[ ]:


#indexing address on first axis
b[0].shape
Out[173]: (3, 4)


# In[172]:


#in numpy min and max, axis that is passed as argument is removed.
b.shape
Out[171]: (2, 3, 4)

np.min(b,axis=0).shape
Out[168]: (3, 4)

np.min(b,axis=1).shape
Out[169]: (2, 4)

np.min(b,axis=2).shape
Out[170]: (2, 3)


# In[187]:


span(b,axis=0).shape


# In[191]:


span(b,axis=0).shape
Out[188]: (3, 4, 2)

span(b,axis=1).shape
Out[189]: (2, 4, 2)

span(b,axis=2).shape
Out[190]: (2, 3, 2)


# In[210]:


span(b,axis=2)


# In[196]:


span(aa,axis=0).shape
Out[194]: (4, 2)

span(aa,axis=1).shape
Out[195]: (3, 2)


# In[205]:


s=span(pts,axis=0)
print('s, shape ',s.shape,':\n',s)
print('s[0], shape ',s[0].shape,':\n',s[0])


# In[ ]:

