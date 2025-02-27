# Toolpath - Filled zone on cube

The goal of this script is to go from a set of point describing a filled polygon to one or multiple paths for the robot to follow. The idea is to draw multiple lines with the pen until the shape is filled.

The method we are using to "fill" the shape is to do a scanning of the shape. We define the boundaries withing which the scanning is going to take place and then check intersection between each segment of the shape and horizontal segments parallel to the y axis.


![image](images/scan_line_algorithm.jpg)

_Scan-line polygon filling algorithm, https://www.tutorialspoint.com/computer_graphics/polygon_filling_algorithm.htm_

We know that after an odd number of intersections we are inside the polygon. We have to be careful about our scan lines. They must not land on an apex, or else the assumption about being inside the shape after an odd number of intersections isn't true anymore.

Some testing in real conditions will be necessary to determine what the optimal interval between scan lines is.

When all the "inside segments" are found, we can write the output file. It looks like this:
```python
0.0598;0.0101
0.0602;0.0101
"release"
0.0558;0.0121
0.0642;0.0121
"release"
0.0518;0.0141
0.0682;0.0141
"release"
0.0478;0.0161
0.0722;0.0161
"release"
```

For each of the segments we have to draw, we have it's start and stop points. The "release" is here to indicate that the pen must be lifted from the surface we're drawing on.

This generated file can then be given as input in the script in repo https://github.com/6figuress/Toolpath---Shape-on-cube