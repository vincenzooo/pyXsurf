    partial=False  #not implemented, if a roi is defined, adds a legend with partial statistics
    if roi is not None:
        rect=np.array([(roi[0][0],roi[1][0]),
                       (roi[0][0],roi[1][1]),
                       (roi[0][1],roi[1][1]),
                       (roi[0][1],roi[1][0]),
                       (roi[0][0],roi[1][0])])   
    
    plt.figure(6)
    plt.clf()
    maximize()
    
    import pdb
    #pdb.set_trace()
    diff=d1-d2
    ax1=plt.subplot(141)
    d1.level().plot(title='Data')
    plt.clim(*remove_outliers(d1.level().data,nsigma=2,itmax=3,span=True))
    if roi is not None:
        plt.plot(rect[:,0], rect[:,1], '--',c='black', lw=2)