def read_pars_from_namelist(filename,separator="="):
    '''Read a set of parameters from a (fortran-like) namelist in file FILENAME.
    Return a dictionary. All values are read as strings.'''

    l=open(filename,'r').readlines()
    l=[ll.strip() for ll in l if ll.count(separator) == 1 ]  #consider only valid line (one and only one = sign in the line).
    dict={}
    for ll in l:
        s=ll.split(separator)
        dict[s[0]]=s[1]
    return dict


if __name__=="__main__":
    file=r'E:\work\WTDf\geoSettings.txt'
    d=read_pars_from_namelist(file)
    print(d)