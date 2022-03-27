##http://stackoverflow.com/questions/8884188/how-to-read-and-write-ini-file-with-python

import ast
import numpy as np

try:
    from configparser import ConfigParser
except ImportError:
    from configparser import ConfigParser  # ver. < 3.0

# instantiate
config = ConfigParser()

# parse existing file
config.read('test2.ini')

# read values from a section
string_val = config.get('section_a', 'string_val')
bool_val = config.getboolean('section_a', 'bool_val')
int_val = config.getint('section_a', 'int_val')
float_val = config.getfloat('section_a', 'pi_val')
m1=np.array(ast.literal_eval(config.get('section_a','markers1')))

# update existing value
config.set('section_a', 'string_val', 'world')

"""
with open('test_update.ini', 'w') as configfile:
    config.write(configfile)
"""