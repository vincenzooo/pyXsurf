from ast import literal_eval
import os
import io

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0


def save_section(sec,outfile):
    
    
    """save a specific section (or list of sections) to a config file."""
    
    
    try:
        sec.name
        sec=[sec]
    except AttributeError:
        assert type(sec)==list
        
    outfolder=os.path.basename(os.path.dirname(outfile))
    with open(outfile,'w') as cfgfile:
        newconf= ConfigParser()
        for s in sec:
            newconf.add_section(s.name)
            for k,v in list(s.items()):
                newconf.set(s.name,k,v)
        newconf.write(cfgfile)
        cfgfile.close()
            
def make_config(settingsFile):
    """create a configuration object from a .ini file, set defaults.
    return config and list of sections to process as specified in 'process' section
    (default to all sections)."""
    # on configparser https://docs.python.org/3/library/configparser.html
    config = ConfigParser()
    """
    these have been included in settings file.
    config.set('DEFAULT' ,'rect1', "")
    config.set('DEFAULT' ,'mfile', "")
    config.set('DEFAULT' ,'mscale', "[1.,1.,1.]")
    config.set('DEFAULT' ,'gscale', "[1.,1.,1.]")
    #config.set('DEFAULT','cane',0)  #set default
    """
    # parse existing file
    config.read(settingsFile)    

    #includelist=json.loads(config.get('process', 'includelist')) #list of settings to process
    if config.has_section('process'):
        includelist=literal_eval(config.get('process', 'includelist')) #list of settings to process
    else:
        includelist=config.sections()
    return config,includelist

def string_to_config(header):
    """ convert a string to a config object. """
    config = ConfigParser()
    buf = io.StringIO("\n".join(header))
    config.read_file(buf)
    return config

