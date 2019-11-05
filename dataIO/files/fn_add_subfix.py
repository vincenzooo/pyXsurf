import os

def fn_add_subfix(fileName,subfix="",newext=None,strip=False,pre=""):
    """Add a subfix to a filename. A new extension can be defined (dot must be included).
    if newext is None, leaves extension unchanged (use empty string to remove extension). 
    If strip is set, only file name (not including path) is returned.
    If pre is passed, it is added as prefix (before the file name, excluding the leading directory).
    datafile='file_with_data.txt'
    outfolder='output_directory'
    print fn_add_subfix(datafile,'_output')
    print fn_add_subfix(datafile,'_output','.dat')
    import os
    fn_add_subfix(os.path.join(outfolder,datafile),'_output','.dat')"""
    
    #2014/07/28 exclude dot from extension (it must be explicitily included in the extension string).
    #   This is useful to remove extension by calling fn_add_subfix(fileName,"","") and also more
    #   consistent with os.path.splitext that returns the extension string with dot included.
    import os
    fn,ext=os.path.splitext(fileName)
    if newext!=None:
        ext=newext
    dir,base=os.path.split(fn)
    if strip:
        dir=""
    return os.path.join(dir,pre+base+subfix+ext)
    """
    if pre:
        return os.path.join(dir,subfix+base+ext)
    else:    
        return os.path.join(dir,base+subfix+ext)"""