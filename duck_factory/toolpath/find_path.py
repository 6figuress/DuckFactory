# ---------------------------find_path------------------------------------
# This code is used to take a shape that has to be filled and create paths
# for the pen to be able to draw the filled shape.
# For this, we need the points describing the shape as an input and the
# points describing each paths as an output.
#
#
# AUTHOR: Guillaume Bessard
# DATE: 24.02.2025

import sys

filename = sys.argv[1]

points = []
processed_points = []

# Read pos from files
with open(filename) as f:
    points = f.readlines()

# Processing the points to get numerical values
for point in points:
    if point[-1] == "\n":
        point = point[0:-1]

    point = map(float, point.split(";"))
    processed_points.append(list(point))

print(processed_points)


# Function to check intersections between segment AB and CD
def segment_intersection(A, B, C, D):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D

    # Compute determinants
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denominator

    # Check if intersection is within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return (intersection_x, intersection_y)

    return None  # No intersection within the segment bounds


# Get all the segments from points
segs = []
for i in range(len(processed_points)):
    segs.append([tuple(processed_points[i - 1]), tuple(processed_points[i])])

print(segs)

# get the boundaries to know where to scan
xmax = None
ymax = None
xmin = None
ymin = None

for i in processed_points:
    if xmax == None:
        xmax = i[0]
        ymax = i[1]
        xmin = i[0]
        ymin = i[1]

    xmax = i[0] if i[0] > xmax else xmax
    ymax = i[1] if i[1] > ymax else ymax
    xmin = i[0] if i[0] < xmin else xmin
    ymin = i[1] if i[1] < ymin else ymin


print(xmax, ymax, xmin, ymin)

# scan lines should not land on apexes!!!!
SHIFT = 0.01  # added shift to not land on apexes
INTERVAL = 0.2  # interval between scan lines

posy = ymin + SHIFT
paths = []

# Scanning every INTERVAL
while posy < ymax:
    # Creating the "scaning segment"
    scanA = (xmin - SHIFT, posy)
    scanB = (xmax + SHIFT, posy)
    x_inter = []

    for i in segs:
        # Comparing "scanning segment" against every segment of the path, logging results
        inter = segment_intersection(scanA, scanB, i[0], i[1])
        if inter is not None:
            x_inter.append(inter[0])

    # Here we sort the results. This way, all the intersection are arrenged "from left to right"
    # We can then pop pairs of segment out of x_inter and they will represent segments that are
    # inside the shape (true only if the is no point where we land on an apex)
    x_inter.sort()
    for i in range(0, len(x_inter), 2):
        paths.append([(x_inter[i], posy), (x_inter[i + 1], posy)])
    posy += INTERVAL

out = ""

# Preparing to write results in the file
for path in paths:
    for points in path:
        out += str(points[0] / 100) + ";" + str(points[1] / 100) + "\n"
    out += "release\n"

out = out[:-1]

with open("out.txt", "w") as f:
    f.write(out)
