"""here functions transfered from notebook. To be merged with other notebooks ICSO2020."""

import numpy as np
from matplotlib import pyplot as plt
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import span
from pyProfile.profile import crop_profile, save_profile
from pyProfile.psd import plot_psd, psd_units
import os
from scipy.stats import binned_statistic
from pySurf.readers.instrumentReader import matrix4D_reader, matrixdat_reader
#from pySurf.readers.nid_reader import read_nid
from pySurf.readers.format_reader import read_nid
from pySurf.data2D_class import Data2D
from pyProfile.profile_class import Profile, Plist
from plotting.backends import maximize

from time import sleep

'''
def dopsd(files,name,outfolder=None,rmsthr=None,psdrange=None,frange=None):
    plt.close('all')

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    for f in files:
        try:
            print (f)
            datadic = read_nid(f)
            data,x,y = datadic['Gr0-Ch1']
            d = Data2D(data,x,y,units=['um','um','nm'],scale=[1000000.,1000000.,1000000000.],name=name) 
            plt.figure(1)
            if f == 'Image00098.nid':
                d=d.crop(None,[30.,45])
                
            dd = d.level(2,axis=1)    
            dd.plot(stats=[0,1],nsigma=1)
            if outfolder:
                plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'','.png',strip=True)))

            dd=dd.transpose()
            p2 = dd.psd(analysis=True,fignum=2,rmsthr=rmsthr)
            if outfolder:
                plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_psd2d','.png',strip=True)))

            fs,ps = p2.avgpsd()
            plt.figure(3)
            plot_psd(fs,ps,units=['mm','mm','nm'])
            plt.ylim(psdrange)
            plt.xlim(frange)
            plt.title(name)
            if outfolder:
                np.savetxt(os.path.join(outfolder,fn_add_subfix(f,'_psd','.dat',strip=True)),
                    np.vstack([fs,ps]).T,header='# '+','.join(psd_units(['mm','mm','nm'])[1:]))
        except KeyError:
            pass

    if outfolder:
        plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_psd','.png',strip=True)))
'''        


# without hardcoded exceptions edit

def dopsd(files,titles,outfolder=None,rmsthr=None,psdrange=None,frange=None,axis=1):
    """from review. 
    files: full path to input file
    titles: can be a single string or array same length of files, used for plot titles
    Make PSDs from AFM data from a list of .nid files generating some outputs if `outfolder` is provided.
    Return psds as couples f,psd with proper units for AFM images.
    rmsthr is used in calculation of psd for Data2D object.
    psdrange and frange are purely a visual parameters for the output.
    """
    plt.close('all')
    
    os.makedirs(outfolder,exist_ok=True)
    psds = []
    
    from dataIO.arrays import is_iterable
    
    if not is_iterable(titles): titles = [titles for f in files]
    #if np.size(titles) == 1:
    #    titles = [titles for f in files]
    #pdb.set_trace()
        
    for f,n in zip(files,titles):
        try:
            # replace with data,x,y = read_nid
            # units um,um,mm; scae=1e,1e6,1e96
            print (f)
            #a = read_nid(f)
            data,x,y = read_nid(f)  # a['Gr0-Ch1']
            d = Data2D(data,x,y,units=['um','um','nm'],scale=[1000000.,1000000.,1000000000.],
                       center = (0,0), name=n) 
            
            # custom edit for some files, level all deg 2 along x (individual lines) in dd
            # if f == 'Image00098.nid':
                # d=d.crop(None,[30.,45])
            # rmsthr= 0.0007 if f == 'Image00095.nid' else rmsthr
            dd = d.level(2,axis=1)  

            # plot
            plt.figure(1) #,figsize=(10,8))
            plt.clf()  
            dd.plot(stats=1,nsigma=3)
            #plt.title(os.path.basename(f))
            if outfolder:
                plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'','.png',strip=True)))
            
            #psd of transposed
            dd=dd.transpose()
            #pdb.set_trace()
            maximize()
            p2 = dd.psd(analysis=True,fignum=2,rmsthr=rmsthr)
            #plt.title(os.path.basename(f))
            if outfolder:
                plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_psd2d','.png',strip=True)))
            
            # average psd
            try:
                fs,ps = p2.avgpsd()
            except:
                fs,ps = p2.avgpsd()()  # probably changed to return profile, convert to x,y
            plt.figure(3) #,figsize=(10,8))
            plot_psd(fs,ps,units=p2.units) # psd_units(p2.units))
            plt.ylim(psdrange)
            plt.xlim(frange)
            #plt.title(os.path.basename(f))
            
            # make outputs 
            psds.append([fs,ps])
            if outfolder:
                np.savetxt(os.path.join(outfolder,fn_add_subfix(f,'_psd','.dat',strip=True)),
                    np.vstack([fs,ps]).T,header='# '+','.join(psd_units(p2.units)[1:]))
                plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_psd','.png',strip=True)))
            #    np.savetxt(os.path.join(outfolder,fn_add_subfix(f,'_psd','.dat',strip=True)),
            #        np.vstack([fs,ps]).T,header='# '+','.join(psd_units(['mm','mm','nm'])[1:]))
        
        except KeyError:
            pass
        
    return psds
        
'''
def mft_psd(files,outfolder,rmsthr=None,psdrange=None,frange=None):
    
    plt.close('all')
    plt.figure(1)
    ax1 = plt.gca()
    plt.figure(2)
    ax2 = plt.gca()
    plt.figure(3)
    ax3 = plt.gca()
    
    if outfolder:
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
    for f in files:
        print (f)
        plt.figure(1)
        plt.clf()
        if os.path.splitext(f)[-1] == '.csv':
            data, x, y = matrix4D_reader(f,center=(0,0),scale = [1e6,1e6,1])  #note that center works also with `None`
        elif os.path.splitext(f)[-1] == '.dat':
            data, x, y = matrixdat_reader(f,center=(0,0), 
                     scale = [1.e6,1.e6,1.e9])  #note that center works also with `None`, data are in m
        else:
            raise TypeError ("Unrecognized data format for MFT")
        D = Data2D(data, x, y , units = ['um','um','nm'],name = os.path.basename(f))
        D=D.remove_nan_frame()

        D.plot(stats = 2)
        if outfolder:
            plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_MFT','.png',strip=True)))

        plt.figure(2)
        plt.clf()
        p2 = D.transpose().level(1,axis=0).psd(analysis=True)
        if outfolder:
            plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_MFT_psd2d','.png',strip=True)))        

        fs,ps = p2.avgpsd()
        plt.figure(3)
        plt.clf()
        plot_psd(fs,ps,units=['um','um','nm'])
        plt.title(name)
        if outfolder:
            np.savetxt(os.path.join(outfolder,fn_add_subfix(f,'_MFT_psd','.dat',strip=True)),
                np.vstack([fs,ps]).T,header='# '+','.join(psd_units(['um','um','nm'])[1:]))           
    if outfolder:
        plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_MFT_psd','.png',strip=True)))
'''

#DA REVIEW, EDITED
def mft_psd(files,outfolder=None,rmsthr=None,psdrange=None,frange=None):
    """read MFT file. Remove nan frame.
    PSD of data after individual line leveling along y.
    return list of Data2D objects.
    If outfolder is given, output files are genearted, adding _MFT subfix to each file name.
        This is needed to avoid name conflicts, data related to same sample are saved in same folder,
        even if instruments differ."""
    # TODO plot all PSD together at the end.
    
    res = []
    if outfolder:  # save files only if defined
        os.makedirs(outfolder,exist_ok=True)
    for f in files:
        print (f)
        plt.figure(1)
        plt.clf()
        if os.path.splitext(f)[-1] == '.csv':
            data, x, y = matrix4D_reader(f,center=(0,0),scale = [1e6,1e6,1])  #note that center works also with `None`
        elif os.path.splitext(f)[-1] == '.dat':
            data, x, y = matrixdat_reader(f,center=(0,0), 
                     scale = [1.e6,1.e6,1.e9])  #note that center works also with `None`, data are in m
        else:
            raise TypeError ("Unrecognized data format for MFT")
        name = os.path.basename(f)
        D = Data2D(data, x, y , units = ['um','um','nm'],name = name)
        D=D.remove_nan_frame().level()

        res.append(D)
        D.plot(stats=1)
        if outfolder:
            plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_MFT','.png',strip=True)))  
        # 2D PSD ANALYSIS, ORIZONTAL LINES INDIVIDUALLY LEVELED
        plt.figure(2)
        plt.clf()
        p2 = D.transpose().level(1,axis=0).psd(analysis=True)
        if outfolder:
            plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_MFT_psd2d','.png',strip=True)))        

        # 1D PSD
        fs,ps = p2.avgpsd()
        plt.figure(3)
        plt.clf()
        plot_psd(fs,ps,units=p2.units)
        plt.title(name)
        if outfolder:
            np.savetxt(os.path.join(outfolder,fn_add_subfix(f,'_MFT_psd','.dat',strip=True)),
                np.vstack([fs,ps]).T,header='# '+','.join(psd_units(['um','um','nm'])[1:]))
            plt.savefig(os.path.join(outfolder,fn_add_subfix(f,'_MFT_psd','.png',strip=True)))
    
    return res # 2021/06/30

        
    
def trim_psds_group(ranges, datafiles, labels, xrange=None,yrange=None,
                    outname=None):
    """ Makes a single psd trimming and averaging multiple ones according to selection. 
        An equivalent updated version (or an attempt to it) in trim_psds_group2
    
        this is lauched to trim a single set of psds related to same sample (or same context).
        ranges, datafiles, labels are lists with same number of elements,
        describing respectively: 
        ranges for each psd (if None, the psd is not included in the output).
            If set to None, include all data.
        datafiles full paths from which to read psds  
            after adding '_psd' subfix and .dat extension to filename.
            One line header is assumed, zero freq
            element is removed from starting data if present.
        labels to be used in plot.
        outname: if provided generate plot of trim and txt with resulting psd.
        
        """
    if ranges is None:
        ranges = np.repeat([[None],[None]],
                           len(datafiles),axis=1).T
    xtot,ytot,bins,xvals,yvals = [] , [], [], [], []
    plt.figure(figsize=(12,6))
    for ran,fn,lab in zip(ranges,datafiles,labels):
        if ran is not None:
            x,y = np.genfromtxt(fn_add_subfix(fn,'_psd','.dat'),unpack=True,skip_header=1, comments='#')  
            if len(x)==0:
                raise ValueError('No data read from file '+fn_add_subfix(fn,'_psd','.dat'))      
            if x[0] == 0:
                x=x[1:]
                y=y[1:]  
            try:     
                xx,yy = crop_profile(x,y,ran)         
            except(IndexError):
                Warning('possible conflict between range and PSD values for input file\n'+
                        fn_add_subfix(fn,'_psd','.dat')+
                        'and range'+str(ran))
                _ = psds_table(tdic) # in case of problems with range, run some diagnostics        
            
            #print(x,y)
            plot_psd(xx,yy,label=lab,units=['um','um','nm'])
            xtot.append(xx)
            ytot.append(yy)
        plt.legend( prop={'size': 12},loc = 1) #bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(xrange)
        plt.ylim(yrange)
        if outname:
            plt.title(os.path.basename(outname))
        plt.grid( which='minor', linestyle='--')
        plt.tight_layout()
    if outname:
        plt.savefig(fn_add_subfix(outname,'_trimpsds','.png'))
    
    ### The following part is of interest only for rebinning,
    ###   not included in this version's output
    ###   because result will in general not be smooth in transitions between
    ###   different intervals.
    ###   if intervals are not overlapping, it is irrelevant.
    # makes rebinning and averaging.
    # we want to start from lower x (assuming inputs are passed not in order),
    # keep x in non overlapping regions, while averaging on common regions.
    # psd frequencies (x) are not equally spacing. If I do average, I get
    #   spikes when typically lower freqs of each psd are more spaced than higher freq
    #   of same psds. Then overlapping intervals of two psds typically have one with
    #   broader spacing. Most of intervals have points only from psds with tighter spacing
    #      when points from the other enter, you get spike. This is why interpolation is needed.
    #sort groups in xtot in ascending order
    pmin=[a.min() for a in xtot]
    igroup =np.argsort(pmin)
    for i in igroup:
        x = xtot[i]
        y = ytot[i]
        if len(bins) == 0:
            bins.append(x)
            xvals.append(x)
            yvals.append(y)
        else:
            sel = x>max(bins[-1])
            xint = np.hstack([ bins[-1][bins[-1]>=min(x)] , x[sel] ])
            #pdb.set_trace()
            xvals.append(xint)
            yvals.append(np.interp(xint,x,y))
            if any(sel):
                #pdb.set_trace()
                #resample second vector on common range
                bins.append(x[sel])
    xtot=np.hstack(xtot)
    ytot=np.hstack(ytot)
    xvals=np.hstack(xvals)
    yvals=np.hstack(yvals)
    xbins = np.hstack(bins)
    ybins = binned_statistic(xvals,yvals,bins=xbins,statistic='mean') [0]
    ###
    
    plot_psd(xbins[:-1],ybins,label='binned',units=['um','um','nm'],
             linestyle='--')
    #plot_psd(xtot,ytot,label='total',units=['um','um','nm'])
    if outname:
        save_profile(fn_add_subfix(outname,'_binpsd','.dat'),xbins[:-1],ybins)
    
    if outname:
        plt.savefig(fn_add_subfix(outname,'_trimpsds','.png'))
    
    return xtot, ytot


def trim_psds_group2(ranges, datafiles, labels,xrange=None,yrange=None,
                    outname=None):
    """ Makes a single psd trimming and averaging multiple ones according to selection.
    
    2022/11/25 attempt to create from previous version above, a function which uses profile functions 
    and avoid duplication from pyProfile.merge_profile, where this was cloned.
    prepare to accept Profiles instead of datafiles.
    The version in profile merge is basically the same with output removed,
    can be handled with Plists.
    
    this is lauched to trim a single set of psds related to same sample (or same context).
    ranges, datafiles, labels are lists with same number of elements,
    describing respectively: 
    ranges for each psd (if None, the psd is not included in the output).
        If set to None, include all data.
    datafiles full paths from which to read psds  
        after adding '_psd' subfix and .dat extension to filename.
        One line header is assumed, zero freq
        element is removed from starting data if present.
    labels to be used in plot.
    outname: if provided generate plot of trim and txt with resulting psd.
        
        """
    def make_Plist(datafiles):
        """create Plist, this will be the data passed when finished."""
        
        Pl = []
        for fn in datafiles:
            x,y = np.genfromtxt(fn_add_subfix(fn,'_psd','.dat'),unpack=True,skip_header=1)
            if x[0] == 0:
                x=x[1:]
                y=y[1:]
            Pl.append(Profile(x,y,name=fn))
        return Plist(Pl)
    
    if ranges is None:
        ranges = np.repeat([[None],[None]],
                           len(datafiles),axis=1).T
    
    Profiles = make_Plist(datafiles)
    # Profiles.crop(ran) # iterate over Plist
    Profiles = [P.crop(ran) for ran,P in zip(ranges,Profiles)]
    for P,lab in zip(Profiles,labels):
        P.name = lab
    
    # xtot,ytot,bins,xvals,yvals = [] , [], [], [], []
            
    plt.figure(figsize=(12,6))  #excluded in merge
    Profiles.psd(labels,units = psd_units(['um','um','nm']))
    plt.legend( prop={'size': 12},loc = 1) #bbox_to_anchor=(1.05, 1), loc='upper left')
        
    #excluded in merge, da qui non ho continuato con Profile.
    plt.xlim(xrange)
    plt.ylim(yrange)
    if outname:
        plt.title(os.path.basename(outname))
    plt.grid( which='minor', linestyle='--')
    plt.tight_layout()
    if outname:  #excluded in merge
        plt.savefig(fn_add_subfix(outname,'_trimpsds','.png'))
    
    return Plist(Profiles)



    ### The following part is of interest only for rebinning,
    ###   not included in this version's output
    ###   because result will in general not be smooth in transitions between
    ###   different intervals.
    ###   if intervals are not overlapping, it is irrelevant.
    # makes rebinning and averaging.
    # we want to start from lower x (assuming inputs are passed not in order),
    # keep x in non overlapping regions, while averaging on common regions.
    # psd frequencies (x) are not equally spacing. If I do average, I get
    #   spikes when typically lower freqs of each psd are more spaced than higher freq
    #   of same psds. Then overlapping intervals of two psds typically have one with
    #   broader spacing. Most of intervals have points only from psds with tighter spacing
    #      when points from the other enter, you get spike. This is why interpolation is needed.
    #sort groups in xtot in ascending order
    pmin=[a.min() for a in xtot]
    igroup =np.argsort(pmin)
    for i in igroup:
        x = xtot[i]
        y = ytot[i]
        if len(bins) == 0:
            bins.append(x)
            xvals.append(x)
            yvals.append(y)
        else:
            sel = x>max(bins[-1])
            xint = np.hstack([ bins[-1][bins[-1]>=min(x)] , x[sel] ])
            #pdb.set_trace()
            xvals.append(xint)
            yvals.append(np.interp(xint,x,y))
            if any(sel):
                #pdb.set_trace()
                #resample second vector on common range
                bins.append(x[sel])
    xtot=np.hstack(xtot)
    ytot=np.hstack(ytot)
    xvals=np.hstack(xvals)
    yvals=np.hstack(yvals)
    xbins = np.hstack(bins)
    ybins = binned_statistic(xvals,yvals,bins=xbins,statistic='mean') [0]
    ###
    
    #excluded in merge:
    plot_psd(xbins[:-1],ybins,label='binned',units=['um','um','nm'],
             linestyle='--')
    #plot_psd(xtot,ytot,label='total',units=['um','um','nm'])
    if outname:
        save_profile(fn_add_subfix(outname,'_binpsd','.dat'),xbins[:-1],ybins)
    if outname:
        plt.savefig(fn_add_subfix(outname,'_trimpsds','.png'))
    
    return xtot, ytot

'''
def trim_datadic(datadic,to_trim,xrange=None,yrange=None,
                 tdic=None,outname=None):
    """"""
    if tdic is None:
        tdic = {}  #trimmed dic
    for name in to_trim:
        r,(d,l) = to_trim[name],datadic[name]   #range, path and label
        xtot, ytot = trim_psds_group(r,d,l,outname=outname)
        tdic[name] = [xtot,ytot]
    return tdic

def trim_datadic(ranges,datadic_val,xrange=None,yrange=None,
                 outname=None):
    """inutile esempio di come chiamare la funzione."""
    return trim_psds_group(ranges,*datadic_val,outname=outname,
                           xrange=xrange,yrange=yrange)
'''

def plot_all_psds(pathlabels,outfolder=None,subfix='_psd',xrange=None,yrange=None):
    """passing a datadic plots all psds for each key. WARNING: units are hardcoded. 
    
    Generate a formatted plot (N.B.: units are fixed) of all PSDs for zipping pathlabel in a single graph 
    and prints their rms.
    Return a list of psds corresponding to the ones read from `pathlabels`, list of couples (file,label)
    No output files are created or returned, must be saved externally.
    
    Note that this was initially accepting a dictionary, of pathlabel values, generating one plot
        per each group of psds, but keys were not used, so this was deprecated.
        If this is the desired behavior, loop can be handled externally.
    
    `pathlabels` is a 2-el list in format [datafile_full_paths,labels] as the values of datadic,
    (modified 2023/08/20 with deprecation warning, it was {label:[(datafile1_full_path,label1),..]} i.e. datadic). 
    `subfix` is the subfix of input files, whose names are in as first item. They must contains freq. e PSD on two cols.
    
    
    `outfolder` is deprecated 2023/08/20 it was only saving plots overwriting them, it can be done externally.
    
    e.g.:
    
        psds = plot_all_psds(pathlabels,outfolder=None,subfix='_psd',xrange=None,yrange=None)
        # plt.title('test')   #modify plot as you like
        plt.savefig('plot_all_psds_demo.png') 
    
    old version: Output files are created in outfolder and sequentially numbered as ##_fullpsd.png using a sequence of numbers
    (can be improved, but keys shouldn't be used as they can contain invalid characters, 
    while using info from datafiles can give raise to repetitions."""
    
    if type(pathlabels) == dict:
        DeprecationWarning("WARNING: deprecated, modify your code to accept a list [fullpaths_beforesubfix, labels]")
        for k in pathlabels.keys():
            plot_all_psds(pathlabels[k], 
                            outfolder = outfolder, subfix = subfix, xrange = xrange, yrange = yrange)
        sleep (10)
        return
    
    filelist, labels = pathlabels
    psds = []
    
    plt.figure(figsize=(12,6))
    for d,l in zip(filelist,labels):
        #read psds from files, add to plot and tdic
        fs,ps = np.genfromtxt(fn_add_subfix(d,subfix,'.dat'), unpack=True,skip_header=1)
        if fs[0] == 0:
            fs = fs[1:]
            ps = ps[1:]
        rms = p[1:].sum()/span(x,1) # this is the correct way to obtain rms, see test_psd_normalization (note that none of the formulas is suitable for non-equally spaced values).
        
        print (os.path.basename(d),l,' : ',rms)
        plot_psd(fs,ps,label=l+', rms = %6.2f nm'%rms,units=['$\mu$m','$\mu$m','nm'])
        psds.append((fs,ps))
    plt.xlabel('Spatial frequency ($\mu$m$^{-1}$)')
    plt.ylabel('PSD (nm$^{2}$ $\mu$m)')
    plt.legend( prop={'size': 12},loc = 1) #bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.gca().xaxis.label.set_size(12)
    plt.gca().yaxis.label.set_size(12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title(name)  #it doesn't make sense, as it ends up using the last key
    plt.grid( which='minor', axis = 'x', linestyle='--')
    plt.tight_layout()
    # if outfolder:
    #     fn = '%02i'%i #os.path.basename(os.path.dirname(d))  #last file containing folder
    #     plt.savefig(fn_add_subfix(fn,'_fullpsd','.png',pre=outfolder+os.path.sep))
    
    return psds


# MAKE PSD TABLE
from pyProfile.profile import crop_profile
def psds_table(tdic,outfolder=None,subfix='_psdtable',ranges = None):
    """ make a table of rms from trimmed PSD in tdic as sqrt of psd integral over range.
    
    This is mostly useless, can be incorporated in plot_all_psds or similer, as psd 
    outfolder not used, kept only for format consinstency. """
    rdic = {}
    for i,name in enumerate(tdic.keys()):
        rmss=[]
        fs,ps = tdic[name]
        isort = np.argsort(fs)
        fs = fs[isort]
        ps = ps[isort]
        if min(fs) == 0:
            print('attenzione')
            fs=fs[1:]
            ps=ps[1:]
        rms_full = np.sqrt(p.sum()/span(x,1))
        
        print (os.path.basename(name),' : ','full freq. range [%6.3f:%6.3f], rms %6.3f'%(*(span(fs)),rms_full))
        rmss.append([span(fs),rms_full])
        #pdb.set_trace()
        if ranges is not None:   #replace with a more sofisticated filter, from other functions
            for r in ranges:  
                if r[0] is None:    #None set auto
                    r[0] = np.min(fs)
                if r[1] is None:
                    r[1] = np.max(fs)
                    
                if r[0]<fs[0] or r[1]>fs[-1]:       # if the range is larger than data, return nan 
                    rms = np.nan
                else:
                    f,p = crop_profile(fs,ps,r)
                    rms = np.sqrt(p.sum()/span(x,1))  #changed from trapz
                print (os.path.basename(name),' : ','freq. range [%6.3f:%6.3f], rms %6.3f'%(r[0],r[1],rms))
                rmss.append([r, rms])
        rdic[name] = rmss
    return rdic
            
#(fs,ps,label=l+', rms = %6.2f nm'%rms,units=['um','um','nm'])


#if outfolder:
#    plt.savefig(fn_add_subfix(fn,'_fullpsd','.png',pre=outfolder+os.path.sep))

'''
# From MIRO_analysis
def psd_compare(plotnames,psdfolder,labels = None, outname=None,
                subfix='_binpsd',xrange=None,yrange=None):
    """Uses `plot_all_psds` to plot comparison between different (binned) psds.
    
    plotnames is a list of keys used to generate input psd files (the only requirement is that 
        `"psdfolder\plotnames+subfix.dat` is a valid psd file). 
        Here they can be `tdic` keys.
        
    outname is a full path whose last part is used both as title for plot and as filename
      for generated plot after adding '_psdcomp' subfix.
      
    labels if provided are used for each plot line
        
    """
    #   tag is used passed to plot_all_psd where is used for print and title 
    #   (no output files are generated). 
    #   output file is created at savefig.

    tag = os.path.basename(outname)
    if labels is None:
        labels = plotnames
    
    toplot = {tag:[[fn_add_subfix(name,subfix,'.dat',pre=psdfolder+os.path.sep) 
                    for name in plotnames],labels]}

    plot_all_psds(toplot,outfolder=None,subfix='',xrange=xrange,yrange=yrange)
    if outname:
        plt.savefig(fn_add_subfix(outname,'_psdcomp','.png'))  

'''
def psd_compare(filelist,psdfolder,labels = None,
                subfix='_binpsd',xrange=None,yrange=None):
    """Uses `plot_all_psds` to plot comparison between different (binned) psds.
    
    filelist is a list of files in psdfolder to use input psd files 
    as `psdfolder\plotnames+subfix.dat'.
    labels if provided are used for each plot line
        
    2023/08/20 modified for a more consistent interface, it is mostly a different
    interface to plot_all_psds. Can be removed in future versions,
    see e.g. MUSE_analysis notebooks.
    """
    #   

    if labels is None:
        labels = filelist
    
    filelist = [fn_add_subfix(name,subfix,'.dat',pre=psdfolder+os.path.sep) 
                    for name in plotnames]

    plot_all_psds(filelist,labels,outfolder=None,subfix='',xrange=xrange,yrange=yrange) 



'''

# From review 
def psd_compare(plotnames,psdfolder,labels = None, outname=None,
                subfix='_binpsd',xrange=None,yrange=None):
    """Uses `plot_all_psds` to plot comparison between different (binned) psds.
    
    plotnames is a list of keys used to generate input psd files (the only requirement is that 
        `"psdfolder\plotnames+subfix.dat` is a valid psd file). 
        Here they can be `tdic` keys.
        
    outname is a full path whose last part is used both as title for plot and as filename
      for generated plot after adding '_psdcomp' subfix.
      
    labels if provided are used for each plot line
        
    """
    #   tag is used passed to plot_all_psd where is used for print and title 
    #   (no output files are generated). 
    #   output file is created at savefig.

    tag = os.path.basename(outname)
    if labels is None:
        labels = plotnames
    
    toplot = {tag:[[fn_add_subfix(name,subfix,'.dat',pre=psdfolder+os.path.sep) 
                    for name in plotnames],labels]}

    plot_all_psds(toplot,outfolder=None,subfix='',xrange=xrange,yrange=yrange)
    if outname:
        plt.savefig(fn_add_subfix(outname,'_psdcomp','.png'))  
        '''