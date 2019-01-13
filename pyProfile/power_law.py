def power_law(x,k_n,N):
    """return a power law for points at x, 
    given scale factor k_n and exponent N as given by k_n*x**N """
    return k_n*x**N    
    
def fit_power_law(x,y,range=None):
    """fit a PSD according to a model y=K_N*x**N
    where x is in mm-1 and y in mm*3"""
    logx = np.log10(x)
    logy = np.log10(y)
    
    if not(range is None):
        i = np.where((logx > range[0]) & (logx < range[1]))
        logx = logx[i]
        logy = logy[i]
    out = np.polyfit(logx, logy, 1)
      
    return (10**out[1],out[0])  #offset, slope -> ampl., index


