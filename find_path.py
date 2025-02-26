# ---------------------------find_path------------------------------------
# 
# 
# 
# 
# 
# 
# AUTHOR: Guillaume Bessard
# DATE: 24.02.2025

import  sys

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

def segment_intersection(A, B, C, D):
    """
    Determines if two line segments AB and CD intersect.
    Returns the intersection point (x, y) if they do, otherwise None.
    """
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D
    
    # Compute determinants
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Parallel or coincident
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denominator
    
    # Check if intersection is within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return (intersection_x, intersection_y)
    
    return None  # No intersection within the segment bounds

# Example usage:
A = (1, 1)
B = (4, 4)
C = (1, 4)
D = (4, 1)

intersection = segment_intersection(A, B, C, D)
print("Intersection:", intersection)

# Get all the segments from points
segs = []
for i in range(len(processed_points)):
    segs.append([tuple(processed_points[i-1]), tuple(processed_points[i])])

print(segs)

# idea: create a rectangle around the shape, scan across the rectangle at different intervals, profit???
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

#scan lines should not land on apexes!!!!
SHIFT = 0.01 # added shift to not land on apexes
INTERVAL = 0.2 # interval between scan lines

posy = ymin + SHIFT
paths = []

while (posy < ymax):
    scanA = (xmin - SHIFT, posy)
    scanB = (xmax + SHIFT, posy)
    x_inter = []

    for i in segs:
        inter = segment_intersection(scanA, scanB, i[0], i[1])
        if inter is not None:
            x_inter.append(inter[0])
    
    x_inter.sort()
    for i in range(0, len(x_inter), 2):
        paths.append([(x_inter[i],posy), (x_inter[i+1], posy)])
    posy += INTERVAL
    

print(paths)