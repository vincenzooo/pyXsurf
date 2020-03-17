def head(fn,N=10):
    """return first n lines of file `fn`, without reading the others.
    
    from https://stackoverflow.com/questions/1767513/read-first-n-lines-of-a-file-in-python"""
    with open(fn) as myfile:
        return [next(myfile) for x in range(N)]
        
        