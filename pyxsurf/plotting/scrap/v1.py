def add_clickable_markers(fig=None,ax=None,key='enter',propertyname='markers',modkey='control',hold=False):
    """Makes pickable the image (image only) in the current figure.
    add a list `propertyname` to the figure. if axis is passed, markers are added only to
        specific axis. This way it is possible to associate differently named markers to 
        different axis in same figure."""
    
    #useful references  
    #http://astronomy.nmsu.edu/holtz/a575/ay575notes/node6.html
    #https://matplotlib.org/examples/event_handling/keypress_demo.html
    def press(event):
        #print('press', event.key)
        sys.stdout.flush()
        if event.key == key:
            fig.canvas.mpl_disconnect(cid1)
            fig.canvas.mpl_disconnect(cid2)
            #fig.canvas.stop_event_loop()
            print (plt.isinteractive())
            plt.interactive(interact_store)  #this sets interactive to original status, but the behaviour doesn't update (if set to ON you still need to close the figure)
            #plt.show(block=False)
            print (plt.isinteractive())
            fig.exit=True  #experimental for loop
        
    def onclick(event):
        button=event.button
        x=event.xdata
        y=event.ydata
        fig=event.canvas.figure
        cax=event.inaxes
        
        if ax is None or cax==ax:         
            if button==1: 
                if event.key is not None:
                    if event.key == modkey:
                        #print (id(fig),propertyname)
                        l=getattr(fig,propertyname)
                        l.append([event.xdata, event.ydata])
                        #add point
                        ax.set_autoscale_on(False)
                        ax.plot(x,y,'or',picker=5,markerfacecolor=None)
            if button==3:
                if event.key is not None:
                    if event.key == modkey:
                        l=getattr(fig,propertyname)
                        if len(l)>0:
                            dist=((np.array(l)-(x,y))**2).sum(axis=1)
                            pop=l.pop(np.argmin(dist))
                            dist2=[(np.array((pop[0]-a.get_xdata(),pop[1]-a.get_ydata()))**2).sum()  for a in ax.lines]
                            ax.lines.pop(np.argmin(dist2))
        
        plt.draw()
        #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        #event.button, event.x, event.y, event.xdata, event.ydata)
    
    def handle_close(evt):
        print('Closed Figure!')
        fig.canvas.mpl_disconnect(cid3)

    #to disconnect:
    #fig.canvas.mpl_disconnect(cid)
    """
    def on_pick(event):
        # not used. this is an alternative way, it automatically manage the selection of the plotted point.
        import matplotlib
        thisline = event.artist
        if isinstance(thisline,matplotlib.lines.Line2D):
            #code to remove point from markers list
            thisline.remove()
            plt.show()
    """
    
    if fig is None:
        fig=plt.gcf()
    interact_store=plt.isinteractive()
    #plt.show(block=True) #This makes impossible to select the points because it blocks events handling.
    #plt.show(block=False) #This works to select the points, but it doesn't release afterwards.
    if hold:
        print ('hold')
        plt.ioff()
    else:
        plt.ion()  #this works in setting interactive mode off independently on starting status. However it sets the status forever, setting ioff here and ion at the end alters
    # the status variable, but it is not applied to the window.
    # so it must be set off here if you want to proceed at the end, but it is necessary to 
    # close the windows to continue.
    
    #fig.canvas.start_event_loop(20)  #this is useful only for timed events (ginput?) documentation is
        #not  very clear to this regar, but it's clearly useless here.
    #add a list property (empty) with the chosen name for markers,
    #if property already exist it will automatically append.    
    if not hasattr(fig,propertyname):
        setattr(fig,propertyname,[])
        #setattr(fig.getattr(propertyname),'ax')
    fig.exit=False #experimental for loop
    #ax.imshow(im)
    #ax.autoscale(False)
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', press)
    cid3 = fig.canvas.mpl_connect('close_event', handle_close)
    return fig
