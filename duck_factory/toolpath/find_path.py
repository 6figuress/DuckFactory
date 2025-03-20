"""
Generate paths to fill a polygon.

This code is used to take a shape that has to be filled and create paths
for the pen to be able to draw the filled shape.
For this, we need the points describing the shape as an input and the
points describing each paths as an output.

AUTHOR: Guillaume Bessard
DATE: 24.02.2025
"""

import sys


def segment_intersection(a: float, b: float, c: float, d: float) -> tuple[float, float]:
    """
    Check intersection between segments AB and CD.

    Args:
        a: First point of segment AB.
        b: Second point of segment AB.
        c: First point of segment CD.
        d: Second point of segment CD.

    Returns:
        The intersection point if it exists, None otherwise
    """
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

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


def find_path(
    points: list[tuple[float, float]], shift: float = 0.01, interval: float = 0.2
) -> list[list[tuple[float, float]]]:
    """
    Find paths to fill a polygon.

    The function takes a list of points describing a polygon and returns the paths
    to fill the polygon. Each path represents a line the pen should follow to fill the polygon.

    Args:
        points: The points describing the polygon, as a list of tuples (x, y).
        shift: The shift to avoid apexes.
        interval: The interval between scan lines.

    Returns:
        The paths to fill the polygon, as a list of lists of tuples (x, y),
        where each inner list represents a path
    """
    # Get all the segments from points
    segs = []
    for i in range(len(points)):
        segs.append([tuple(points[i - 1]), tuple(points[i])])

    # get the boundaries to know where to scan
    xmax = None
    ymax = None
    xmin = None
    ymin = None
    for i in points:
        if xmax == None:
            xmax = i[0]
            ymax = i[1]
            xmin = i[0]
            ymin = i[1]
        xmax = i[0] if i[0] > xmax else xmax
        ymax = i[1] if i[1] > ymax else ymax
        xmin = i[0] if i[0] < xmin else xmin
        ymin = i[1] if i[1] < ymin else ymin

    # scan lines should not land on apexes!!!!
    posy = ymin + shift
    paths = []

    # Scanning every INTERVAL
    while posy < ymax:
        # Creating the "scaning segment"
        scan_a = (xmin - shift, posy)
        scan_b = (xmax + shift, posy)
        x_inter = []
        for i in segs:
            # Comparing "scanning segment" against every segment of the path, logging results
            inter = segment_intersection(scan_a, scan_b, i[0], i[1])
            if inter is not None:
                x_inter.append(inter[0])
        # Here we sort the results. This way, all the intersection are arrenged "from left to right"
        # We can then pop pairs of segment out of x_inter and they will represent segments that are
        # inside the shape (true only if the is no point where we land on an apex)
        x_inter.sort()
        for i in range(0, len(x_inter), 2):
            if i + 1 < len(x_inter):  # Ensure we have a pair
                paths.append([(x_inter[i], posy), (x_inter[i + 1], posy)])
        posy += interval

    return paths


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 3:
        print("Usage: python find_path.py <in_filename> <out_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    out_filename = sys.argv[2]

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

    # Call the find_path function
    paths = find_path(processed_points)

    print([tuple(p) for p in processed_points])

    print(paths)

    out = ""
    # Preparing to write results in the file
    for path in paths:
        for points in path:
            out += str(points[0] / 100) + ";" + str(points[1] / 100) + "\n"
        out += "release\n"
    out = out[:-1]

    with open(out_filename, "w") as f:
        f.write(out)
