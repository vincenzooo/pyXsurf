from setuptools import find_packages, setup

"""with this install works only if numpy is already installed, otherwise fails on wheel."""

#p = find_packages('.')
#p = find_packages("src", exclude=["test"]),

setup(
  name='pyXsurf',
  version='0.1.40',
  description="Python library for X-Ray Optics, Metrology Data Analysis and Telescopes Design.",
  url='https://github.com/vincenzooo/pyXSurf',
  author='Vincenzo Cotroneo',
  author_email='vincenzo.cotroneo@inaf.it',
  install_requires=['wheel','numpy','matplotlib','scipy','IPython','astropy'],
  #package_dir={'': 'pyxsurf'} #this makes me import as old style e.g. from pySurf import data2D, but can create overlapping, e.g. `test` or `plotting` may be used for other things.
  #package_dir={'pySurf': 'pyxsurf/pySurf',
  #             'dataIO': 'pyxsurf/dataIO'}
  package_dir={'': 'pyxsurf'},
  #packages=p,
  packages = ['pySurf','dataIO','utilities'],
  setup_requires=['numpy','astropy']
)


from setuptools import find_packages

# or
from setuptools import find_namespace_packages

'''
#from https://docs.pytho
n.org/3/distutils/setupscript.html

from distutils.core import setup

setup(    name='pyXsurf',
    version='0.1.1',
    packages=['pyXsurf'],
    description="Python library for X-Ray Optics, Metrology Data Analysis and Telescopes Design.",
    url='https://github.com/vincenzooo/pyXSurf',
    author='Vincenzo Cotroneo',
    author_email='vincenzo.cotroneo@inaf.it',
    #url='https://github.com/vincenzooo/pyXSurf',
    )
'''     
     
'''

from setuptools import setup
setup(
    name='pyXsurf',
    version='0.1.1',
    packages=['pyXsurf'],
    description="Python library for X-Ray Optics, Metrology Data Analysis and Telescopes Design.",
    url='https://github.com/vincenzooo/pyXSurf',
    author='Vincenzo Cotroneo',
    author_email='vincenzo.cotroneo@inaf.it',
    keywords=['surfaces', 'metrology', 'PSD'],
    tests_require=[
        'pytest'
    ],
    package_data={
        # include json and pkl files
        '': ['*.json', 'models/*.pkl', 'models/*.json'],
    },
    include_package_data=False,
    python_requires='>=3'
)

# from 

'''