from numpy import *
import numpy as np
from matplotlib.pyplot import *
from matplotlib import colors
import pickle

def scatter3d(x,y,z,**args):
    """Make a 3d scatter plot"""
    fig = figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x,y,z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax

#Make iso plot
def isoplot(*args,**kargs):
    fig = gcf()
    #fig.clf()
    ax = fig.add_subplot(111,aspect='equal')
    ax.plot(*args,**kargs)
    fig.show()
    return ax

#Make temp and data double plot
def pltd(x1,y1,x2,y2,xlabel='',ylabel1='',ylabel2='',title='',\
         ystyle1='b-',ystyle2='r-',ylim1='',ylim2='',label1='',label2=''):
    fig = figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x1,y1,ystyle1,label=label1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1,color='b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    if ylim1!='':
        ax1.set_ylim(ylim1)
    ax2 = ax1.twinx()
    ax2.plot(x2,y2,ystyle2,label=label2)
    ax2.set_ylabel(ylabel2,color='r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')
    if ylim2!='':
        ax2.set_ylim(ylim2)
    ax1.set_title(title)

    return ax1,ax2

#Make temp and data double plot
def pltd2(x1,y1,fn,xlabel='',ylabel1='',ylabel2='',title='',\
         ystyle1='b-',ystyle2='r-',ylim1='',ylim2='',label1='',label2=''):
    fig = figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x1,y1,ystyle1,label=label1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    #Get x tick positions
    b = ax1.get_xticks()
    newb = fn(b)
    #newb = newb.astype('str')
    for i in range(size(newb)):
        newb[i] = '%.2f' % newb[i]
    #Create second axis
    ax2 = ax1.twiny()
    ax2.set_xticks(b)
    ax2.set_xticklabels(newb.astype('str'))
    
    return ax1,ax2

#Print a number in scientific notation
def scinot(x):
    print ('%e' % x)

#Find index where vector is closest to a value
def mindiff(x,y):
    diff = abs(x-y)
    diff2 = diff[invert(isnan(diff))]
    return where(diff==min(diff2))[0]

#Make a contour plot of an image
def mycontour(img,x=0\
               ,y=0,log=False,fmt=None,nlev=None,maxl=None,minl=None):
    if nlev is None:
        nlev = 100.
    if size(x) == 1:
        x = arange(shape(img)[1])
        y = arange(shape(img)[0])
    if log==False:
        if maxl==None:
            levels = linspace(nanmin(img),nanmax(img),nlev)
        else:
            levels = linspace(minl,maxl,nlev)
        c = contourf(x,y,img,levels=levels)
    else:
        mn = np.log10(nanmin(img))
        mx = np.log10(nanmax(img))
        levels = linspace(mn,mx,nlev)
        levels = 10**levels
        c = contourf(x,y,img,levels=levels,norm=colors.LogNorm())
    if fmt==None:
        fmt='%.'+str(np.int(-np.floor(log10(levels[1]-levels[0])/nlev)))+'f'
    colorbar(format=fmt)
    return c

#Convert from bin edges to bin centers
def edgestocent(x):
    c = []
    for i in range(size(x)-1):
        c.append((x[i+1]+x[i])/2.)
    return array(c)

#Return center of bins in histogram function
def myhist(arr,bins=10,density=None,range=None):
    val = histogram(arr,bins=bins,density=density,range=range)
    b = np.zeros(size(val[1])-1)
    for i in np.arange(np.size(b)):
        b[i] = val[1][i]+(val[1][1]-val[1][0])/2.
    return (val[0],b)

#Return mean of array with nans
def nanmean(arr):
    nanmask = isnan(arr)
    return arr[invert(nanmask)].mean()

#Load from pickle file
def pload(filename):
    f = open(filename,'r')
    b = pickle.load(f)
    f.close()
    return b

#Save to pickle file
def psave(data,filename):
    f = open(filename,'w')
    pickle.dump(data,f)
    f.close()
