from setuptools import find_packages

"""with this install works only if numpy is already installed, otherwise fails on wheel."""

p = find_packages('.')
print(p, "\n-----------")
p=find_packages("pyxsurf", exclude=["abandoned"])
print("found:\n%s\n"%p)

"""RESULT

Vedi struttura directory sotto:
solo i files in packages (setup.py) sono inclusi, con le loro sottocartelle.
Il root e' la cartella passata come argomento 


# find_packages('.')
(base) C:\\Users\kovor\Documents\python\pyXTel>python test_find.py
['pyxsurf', 'pyxsurf.dataIO', 'pyxsurf.plotting', 'pyxsurf.pyGeo3D', 'pyxsurf.pyProfile', 'pyxsurf.pySurf', 'pyxsurf.thermal', 'pyxsurf.dataIO.config', 'pyxsurf.pySurf.readers', 'pyxsurf.pySurf.scripts', 'pyxsurf.pySurf.readers.PyMDT', 'pyxsurf.pySurf.readers.PyMDT.pymdt']
-----------
# find_packages("pyxsurf", exclude=["abandoned"])
found:
['dataIO', 'plotting', 'pyGeo3D', 'pyProfile', 'pySurf', 'thermal', 'dataIO.config', 'pySurf.readers', 'pySurf.scripts', 'pySurf.readers.PyMDT', 'pySurf.readers.PyMDT.pymdt']


Folder PATH listing for volume Windows-SSD
Volume serial number is CA80-B408
C:.
+---.hypothesis
|   \---unicode_data
|       \---11.0.0
+---.pytest_cache
|   \---v
|       \---cache
+---.vscode
+---build
|   +---bdist.win-amd64
|   \---lib
|       +---dataIO
|       +---pyProfile
|       +---pySurf
|       +---pyxsurf
|       |   +---dataIO
|       |   +---pySurf
|       |   \---test
|       \---test
+---dist
+---docs
|   \---pyxsurf
+---pyxsurf
|   +---.ipynb_checkpoints
|   +---.spyproject
|   |   \---config
|   |       +---backups
|   |       \---defaults
|   +---.vs
|   |   \---pyxsurf
|   |       \---v16
|   +---.vscode
|   +---abandoned
|   |   \---pythonAnalyzer
|   |       \---__pycache__
|   +---dataIO
|   |   +---config
|   |   |   +---.ipynb_checkpoints
|   |   |   +---config_experiments
|   |   |   +---results
|   |   |   \---__pycache__
|   |   +---test
|   |   |   +---test_data
|   |   |   \---__pycache__
|   |   \---__pycache__
|   +---notebooks
|   |   \---.ipynb_checkpoints
|   +---plotting
|   |   +---scrap
|   |   +---test
|   |   |   +---input_data
|   |   |   \---Scale
|   |   \---__pycache__
|   +---pyGeo3D
|   |   \---__pycache__
|   +---pyProfile
|   |   +---.ipynb_checkpoints
|   |   +---test
|   |   |   +---.ipynb_checkpoints
|   |   |   +---PSDtest
|   |   |   \---spizzichino
|   |   |       \---testHEW_PCO1.3S04_uniformMRF
|   |   \---__pycache__
|   +---pySurf
|   |   +---demo
|   |   |   +---.ipynb_checkpoints
|   |   |   +---analysis_class
|   |   |   +---basic
|   |   |   |   \---.ipynb_checkpoints
|   |   |   +---fitting
|   |   |   |   \---.ipynb_checkpoints
|   |   |   \---sample_scripts
|   |   +---old
|   |   |   \---findrect_attempts
|   |   +---readers
|   |   |   +---.ipynb_checkpoints
|   |   |   +---old
|   |   |   |   \---readers_dev
|   |   |   |       +---old
|   |   |   |       \---readers
|   |   |   |           \---deleteme
|   |   |   +---PyMDT
|   |   |   |   +---.ipynb_checkpoints
|   |   |   |   +---pymdt
|   |   |   |   |   \---__pycache__
|   |   |   |   +---Test Files
|   |   |   |   \---__pycache__
|   |   |   +---read_ibw_files
|   |   |   \---__pycache__
|   |   +---scripts
|   |   |   \---__pycache__
|   |   +---test
|   |   |   +---.ipynb_checkpoints
|   |   |   +---input_data
|   |   |   |   +---4D
|   |   |   |   |   +---180215_C1S06_cut
|   |   |   |   |   \---C1S15
|   |   |   |   +---AFM
|   |   |   |   |   +---gwyddion_converted
|   |   |   |   |   +---NTMDT
|   |   |   |   |   +---POLIMI
|   |   |   |   |   +---PyMDT
|   |   |   |   |   |   \---Test Files
|   |   |   |   |   \---WSxM
|   |   |   |   |       +---cits
|   |   |   |   |       +---fz
|   |   |   |   |       +---iv
|   |   |   |   |       +---iz
|   |   |   |   |       +---mov
|   |   |   |   |       +---spm
|   |   |   |   |       \---zv
|   |   |   |   +---CCI
|   |   |   |   +---csv
|   |   |   |   +---exemplar_data
|   |   |   |   |   \---scratch
|   |   |   |   +---fits
|   |   |   |   |   +---4D
|   |   |   |   |   |   \---180215_C1S06_cut
|   |   |   |   |   +---reproducibility
|   |   |   |   |   \---WFS
|   |   |   |   +---newview
|   |   |   |   +---profilometer
|   |   |   |   |   +---04_test_directions
|   |   |   |   |   \---2018_01_11
|   |   |   |   +---readers
|   |   |   |   +---test_fitCylinder
|   |   |   |   |   +---bonding_data
|   |   |   |   |   +---OP2S04b_cone
|   |   |   |   |   \---PCO1S17_cylinder
|   |   |   |   \---zygo_data
|   |   |   |       +---merate_mx
|   |   |   |       \---merate_mx2
|   |   |   +---pointCloud
|   |   |   |   \---01_initialization
|   |   |   |       \---input
|   |   |   +---PSD
|   |   |   |   \---2dpsd
|   |   |   |       +---output
|   |   |   |       \---output_ok_20180503
|   |   |   +---results
|   |   |   |   +---04_test_directions_output
|   |   |   |   +---data2D_class
|   |   |   |   +---outliers_analysis
|   |   |   |   +---test_Data2D_fitsreader
|   |   |   |   +---test_read_csv
|   |   |   |   +---test_read_fits
|   |   |   |   \---test_read_sur
|   |   |   +---test_fitCylinder
|   |   |   |   +---bonding_data
|   |   |   |   +---OP2S04b_cone
|   |   |   |   \---PCO1S17_cylinder
|   |   |   +---transform_cells
|   |   |   |   \---cell_grid_position
|   |   |   \---_build
|   |   |       +---doctrees
|   |   |       \---html
|   |   |           +---_sources
|   |   |           \---_static
|   |   +---tutorials
|   |   |   \---analysis_class
|   |   |       \---.ipynb_checkpoints
|   |   \---__pycache__
|   +---pyXsurf.egg-info
|   +---thermal
|   |   \---__pycache__
|   \---__pycache__
+---pyXsurf.egg-info
+---test
\---xpyxsurf
    +---dataIO
    |   \---__pycache__
    +---plotting
    |   \---__pycache__
    +---pyProfile
    |   \---__pycache__
    \---pySurf
        +---readers
        |   \---__pycache__
        +---scripts
        |   \---__pycache__
        \---__pycache__

"""