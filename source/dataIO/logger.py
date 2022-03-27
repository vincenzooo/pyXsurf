import logging


def start_logger(cfgfile=None):

    #if logger exists recover without adding handlers, if not load from file
    #  or if config file is not provided, uses print
    try:
        logger
        print("logger existing")
    except:
        print("logger non existing.")
    logger=logging.getLogger()
    if len(logger.handlers) == 0:
        """not sure about original indentation here."""   
        if cfgfile is not None:
            logger=logging.config.fileConfig(cfgfile)
        else:
            print('cfgfile missing, not sure what happens now.')
    
    return logger

class Logger():
    
    #methods that mimic the behavior of logging.Logger
    def info(self,msg):
        print (msg)
    def debug(self,msg):
        print(msg)
        


cfgfile='logging.ini'