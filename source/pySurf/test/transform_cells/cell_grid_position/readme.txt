2014/07/11 Vincenzo
Program cell_grid_position.py
The program accepts as input the measured position of the reference points on sample and returns the position in the measurement coordinate system of a set of points (e.g. the center of each cell) provided in sample coordinates.

To use this program 
1) set measured reference points position in file reference_points_input.txt in the order NE, SE, SW, NW. Points are listed on 3 columns as x,y,z where 0 is usually set to 0
2) Set the coordinates of the points to be converted in file cell_center_input.txt
3) Go in the folder with the program and launch it with: 
python cell_grid_position.py
output can be redirected to a file, e.g.:
python cell_grid_position.py > output.txt

Additional notes:
Coordinates of reference points in sample coordinates are assumed to be:
[[-26,26., 0.],
[-26.,-26., 0.],
[26., -26., 0.],
[26., 26., 0.]])
This is hardcoded in the python program, they can be modified there.
