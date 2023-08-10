"""functions to adjust linestyles."""

import matplotlib.pyplot as plt
import numpy as np


def colors_20(plot=False):
    """
    Build a 
    
    Lazily copied from:    https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
    """
    
    NUM_COLORS = 20
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    colors = []
    styles = []
    cm = plt.get_cmap('gist_rainbow')


    for i in range(NUM_COLORS):
        lc = cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS)
        ls = LINE_STYLES[i%NUM_STYLES]
        colors.append(lc)
        styles.append(ls)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = [np.arange(10)*(i+1) for i in range(NUM_COLORS)]
        for i,(lc,ls) in enumerate(zip(colors,styles)):
            lines = ax.plot(data[i])
            lines[0].set_color(lc)
            lines[0].set_linestyle(ls)
        plt.show()
    
    return colors, styles
    

def make_styles(s1,cf,argsdic,legshow=None):
    """given a dataframe s1, associate to each element in s1 a plotting style in form of dictionary
        according to values of columns for s1 indices in cf.
        argsdic has the form {'graphic_property':{'colkey':[style1,style2]}},
        where col_to is the column index that is used to determine the graphic style.
        it is done in this format {gp:{ck:stylst}} rather than the more intuitive {ck:{gp:stylst}}
        to conform to the standard {gp:values}.
        Note that for each graphic_property, there should be a single key for the associated dictionary 
        {kc:stylst}
        Also, the entire dataframe s1 is passed instead of its index, so that a function on s1
        columns can be added subsequently (maybe this is useless and all processing should be
        handled on cf, that can also be =s1 if this contains already all information).
        Legshow is a list with indices (in form of `colkey`) that tells
        which legend to plots (locations are determined in increasing values of loc starting from 1, with legend plotted in order as 
        in argsdic). Set to empty string to disable plotting, legends can
        be plotted at later time using legenddic.
        
        2022/11/23 moved here from pySurf.scripts.repeatability_dev.
        TODO See also seaborn
        """
    # TODO:
    # - come posso dare un doppio stile (e.g. red circles vs blue triangles
    # - aggiustare legende per plottare solo simboli o solo linee con stili comuni inclusi 
    # - how to apply a legend different than value (e.g. chiller on/off instead of 0/1)
    
    stylelist=[]
    #stylelist=[{} for i in range(len(s1.index))]  #[{}]*len doesn't work, it creates n copies of same dictionary, so any inplace change to an element is reflected to other elements
    if legshow is None:
        legshow=[list(argsdic[k].keys())[0] for k in list(argsdic.keys()) if isinstance(argsdic[k], collections.Mapping)]
    legenddic={}
    pos=1
    j=0  #ugly way to check when it's first iteration and creating item    
    for k,v in argsdic.items():   #iterate over column tags associated with the property
        if isinstance(v, collections.Mapping): #dictionary
            assert len(v.keys())==1            
            import pdb
            for p,c in v.items(): #iterate over properties, anyway this will always be a single key,
                newkeys=np.unique(cf[p].values)
                if isinstance(c, collections.Mapping): #dictionary
                    sd={kk:{k:vv} for kk,vv in c.items()}
                elif isinstance(c, list):            
                    #so there might be neater ways to unpack
                    p_cycle=cycler(k,c)

                    #builds a dictionary with unique values as keys and property value as value
                    sd={}
                    for nk,sty in zip(newkeys,cycle(p_cycle)):
                        sd[nk]=sty
                    #{'LSKG': {'color': 'r'}, 'VC': {'color': 'g'}}
                    #{'CylRef': {'marker': 'x'}, 'PCO1.2S01': {'marker': 'o'}, 'PCO1S23': {'marker': '+'}}
                    #print(sd)
                    
            # builds the return value stylelist (list of graphic properties
            # associate a style to each row of stats
            for i,cc in enumerate(cf[p].values):
                #stylelist,legenddic=test_make_styles(sc,cc,   #doesn't plot others
            #{"color":{"operator":{'KG':'r','LS':'b'}}})
                #stylelist[i].update(sd.get(None,sd.get(cc,{'marker':'','linestyle':''})))
                #pdb.set_trace()
                #stylelist[i]= 
                if isinstance(v, collections.Mapping): #dictionary
                    s = sd.get(cc,sd.get(None,None)) #set to key if in sd, otherwise to the graph prop dictionary for None if it was
                    #set, or set the style to None (exclude in plot_historical) if not.   
                else:
                    s = {k:v}
                
                if j == 0:
                    stylelist.append(s)
                else:
                    stylelist[i] = None if (stylelist[i] is None or s is None) else {**stylelist[i],**s}        
            j=1   

            #pdb.set_trace()
            #make a dictionary of legends for each of the keys in stylelist
            legenddic[p]=[[sd[t],t] if t is not None else [sd[t],'Other'] for t in sd.keys()]
            #handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
            #empty_patch = mpatches.Patch(color='none', label='Extra label') # create a patch with no color
            #handles.append(empty_patch)  # add new patches and labels to list
            #labels.append("Extra label")

            #plt.legend(handles, labels) # apply new handles and labels to plot
            handles=[Line2D([], [], label= vv[1], **(vv[0])) for vv in legenddic[p]]
            labels=[vv[1] for vv in legenddic[p]]
            #handles=[v[0] for v in ]
            
            if p in legshow:
                plt.gca().add_artist(plt.legend(handles,labels,title=p,loc=pos))
            pos=pos+1
            
        else:
            for i in range(len(cf.index)):
                s = {k:v}
                if j == 0:
                    stylelist.append(s)
                else:
                    stylelist[i] = None if (stylelist[i] is None or s is None) else {**stylelist[i],**s}        
            j=1   
        
    return stylelist,legenddic
        
def test_make_styles(s1,cf,kwargs):
    plt.clf()
    stylelist,legenddic=make_styles(s1,cf,kwargs)
    print("styledic(styles for each line):\n%s \nlegenddic(dictionary of legends):\n%s\n"%
          (stylelist,legenddic))
    display(plt.gcf())
    print ("\n")
    return stylelist,legenddic