"""Independently from underlying mechanism or preprocessing, provides an interface to access configurations
on files or variables and manipulate them. A configuration is a dictionary
"""

import json
#from pyhocon import ConfigFactory
#https://stackoverflow.com/questions/14050281/how-to-check-if-a-python-module-exists-without-importing-it

import importlib

#if pyhocon is installed, use its method ConfigFactory for `parse`, otherwise use json.load.
if importlib.util.find_spec("pyhocon") is not None:
    from pyhocon import ConfigFactory as parse
    # use configuration files ono HOCON format: The parsed config can be seen as a nested dictionary (with types automatically inferred) where values can be accessed using normal dictionary getter (e.g., conf['a']['b'] or using paths like conf['a.b']) or via the methods get, get_int (throws an exception if it is not an int), get_string, get_list, get_float, get_bool, get_config.
else:
    parse = lambda jfile : json.load(open(jfile,'r'))

    
def dict_from_json(file):
    """just a wrapper around parse, reads a neste dictionary from json file. Return a dictionary."""
    print ('dict_from_json is a wrapper for configuration: please use a more direct reader if you are'+
        ' interested in a specific format (e.g. json.load().')
    return parse(file)

def add_config(confdic,tag,**kwargs):
    """add configuration from variable to confdic.
    Used to migrate old code to new."""
    confdic[tag]={}
    for k,v in kwargs.items():
        confdic[tag][k]=v

        
        
def read_json_conf(jfile):
    """Read a dictionary from a json file and return a dic. Can include or exclude first level keys.
    
    supercede conf_from_json.
    For `settings` dictionary, each key is a unique name for a configuation,
    The value is a nested dictionary in format {'property_name':<value>}, where property_name 
    is a string and value can be any type (including dictionaries?). 
    
    special keys:
        'default': dictionary of variable names and values that are used for
        every entry in the configuration when these variables are not defined.
        'override': dictionary of variable names and values that override every
            entry in the configuration.
        'includeonly': if set to some list of strings, includes only configurations that are part of the list
        'exclude': list as in includeonly, exclude the listed config, is applied after includeonly.
        """
    
    
    d=parse(jfile)
    #d=json.load(open(jfile,'r'))

    config_settings=d.pop('config_settings',None)    
    default=d.pop('default',None)            
    override=d.pop('override',None)    
    
    if config_settings is not None:
        if 'includeonly' in config_settings:
            if len(config_settings['includeonly'])>0:
                d={k:d[k] for k in config_settings['includeonly'] if k in d}        
        
        if 'exclude' in config_settings:
            #import pdb
            #pdb.set_trace()
            for k in config_settings['exclude']:
                res=d.pop(k,None)
                if res is None:
                    print ("WARNING: json config exclude file not found %s"%k)     
                    
    if default is not None:
        for tt,cc in d.items(): #tag and conf
            for k,dv in default.items():
                cc[k]=cc.get(k,dv)  #or also d[tt][k]=cc.get('k',dv)
       
    if override is not None:
        #print('override')
        for tt,cc in d.items(): #tag and conf
            for k,dv in override.items():
                cc[k]=dv  
                #print('%s override %s %s'%(tt,k,dv))
                
        
                
    return d
    
    
def write_json_conf(jfile,conf):
    """write a configuration to a json file."""
    json.dump(conf, open(jfile,'w'), sort_keys=True, indent=4)  #according to https://stackoverflow.com/questions/7100125/storing-python-dictionaries
    #json.dump(conf,open(jfile,'w'),indent=1)
    
    
    
def conf_from_json(jfile,include=None,exclude=None):
    
    """ Read a dictionary from a json file and return a dic. Can include or exclude first level keys.
    
    For `settings` dictionary, each key is a unique name for a configuation,
    The value is a nested dictionary in format {'property_name':<value>}, where property_name 
    is a string and value can be any type (including dictionaries?). 
    
    If included is set to a list,
    only the keys of first level (configurations) matching names in this list are included.
    In alternative, exclude can be set, leading to include all but these names.
    
    """ 
    
    PRINT ("`conf_from_json` will be eliminated, use instead `read_json_conf(jfile)` ")
        
    d=json.load(open(jfile,'r'))
    #print("%i keys read from file.."%(len(d)))  
    
    if include is not None:
        d={k:d[k] for k in d.keys() if k in include}
    else:
        if exclude is not None:
            for e in exclude:
                try:
                    d.pop(e)
                except (KeyError,ValueError):
                    print("key not found for ",e)
    
    print("----")
    print("included keys: ",list(d.keys()),'\n\nfirst element:\n',d[list(d.keys())[0]])
    print(len(list(d.keys()))," datasets")
    
    return d
    

