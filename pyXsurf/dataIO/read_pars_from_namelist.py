def namelist_from_string(l,separator="="):
    """extract namelist parameters from a list of strings (e.g. file lines). """
    l=[ll.strip() for ll in l if ll.count(separator) == 1 ]  #consider only valid line (one and only one = sign in the line).
    dict={}
    for ll in l:
        s=ll.split(separator)
        dict[s[0].strip()]=s[1].strip()
    return dict    


def read_pars_from_namelist(filename,separator="="):
    '''Read a set of parameters from a (fortran-like) namelist in file FILENAME.
    Return a dictionary. All values are read as strings.
    A file-object like io.StringIO(filetext) can be used in place of a real file.'''
    
    if hasattr(filename,'readlines'):
        l=filename.readlines()
    else:
        l=open(filename,'r').readlines()

    return namelist_from_string(l,separator=separator)

if __name__=="__main__":
    file=r'E:\work\WTDf\geoSettings.txt'
    d=read_pars_from_namelist(file)
    print(d)