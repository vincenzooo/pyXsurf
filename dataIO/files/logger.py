import logging


def start_logger(cfgfile=None):

    #if logger exists recover without adding handlers, if not load from file
    #  or if config file is not provided, uses print
    try:
        logger
        print("logger existing")
    except:
        
    logger=logging.getLogger()
    if len(logger.handlers) == 0:
        
    if cfgfile is not None:
        logger=logging.config.fileConfig(cfgfile)
    else 
    
    
    return logger

Class Logger(object):
    
    #methods that mimic the behavior of logging.Logger
    def info(msg):
        print (msg)
    def debug(msg):
        print(msg)
        

cfgfile='logging.ini'