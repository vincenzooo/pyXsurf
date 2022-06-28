import logging
# questo file dovra' sparire.
from dataIO.logs import logging_function
    
print(__name__)
print("logging: ",logging)

print("logger: ",logging.getLogger(__name__),logging.getLogger())

logging_function()