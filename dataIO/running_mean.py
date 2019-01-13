def running_mean(y, N):
    #if N is even window is not centered on points but at half between. One point less is returned.  
    y2=np.insert(y, 0, 0)
    cumsum = y2*0+np.nan_to_num(y2).cumsum()
    cumsum= (cumsum[N:] - cumsum[:-N]) / N
    start=np.array([np.nanmean(y[:i*2+1]) for i in range(N)][:int(N/2)])  #first elementh of cumsum is the average 
    end=np.array([np.nanmean(y[-(i*2+1):]) for i in range(N)][:int(N/2)][::-1])
    cumsum= np.hstack([start,cumsum,end]) #add ends
    return cumsum