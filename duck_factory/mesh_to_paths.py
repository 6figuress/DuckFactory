import numpy as np
from trimesh import Trimesh, load_mesh
from pointsToPaths import PathFinder
from point_sampling import (
    sample_mesh_points,
    cluster_points,
    Point,
    Color,
)
from scipy.spatial.transform import Rotation
import json

type Quaternion = tuple[float, float, float, float]
type PathPosition = tuple[*Point, *Quaternion]
type Path = tuple[Color, list[PathPosition]]

BASE_COLOR = (255, 255, 0, 255)
COLORS = [
    BASE_COLOR,  # Yellow
    (0, 0, 0, 255),  # Black
    (0, 0, 255, 255),  # Blue
    (0, 255, 0, 255),  # Green
    (0, 255, 255, 255),  # Cyan
    (255, 0, 0, 255),  # Red
    (255, 255, 255, 255),  # White
]


def mesh_to_paths(
    mesh: Trimesh, n_samples: int = 50_000, max_dist: float = 0.1
) -> list[Path]:
    """
    Convert a mesh to a list of paths, each path being of a certain color and containing positions easily usable for plotting.

    Args:
        mesh: The mesh to convert to paths
        n_samples: The number of samples to take from the mesh
        max_dist: The maximum distance between two samples for neighborhood

    Returns:
        List of paths, each containing a color and a list of PathPosition (point and quaternion)
    """
    sampled_points = sample_mesh_points(
        mesh, base_color=BASE_COLOR, colors=COLORS, n_samples=n_samples
    )

    clusters = cluster_points(sampled_points)

    for points, _, _ in clusters:
        for point in points:
            # convert from y-up to z-up coordinate system
            point.coordinates = (
                point.coordinates[0],
                point.coordinates[2],
                point.coordinates[1],
            )
            point.normal = (
                point.normal[0],
                point.normal[2],
                point.normal[1],
            )

    # Compute the paths for each cluster
    paths = []
    for points, color, is_noise in clusters:
        if is_noise:
            # The noise clusters contain points that we don't want to connect
            # Create a new path for each point
            for point in points:
                paths.append((color, [point]))
        else:
            # Connect the points in the cluster to form paths
            path_finder = PathFinder(points, max_dist)
            paths_positions = path_finder.find_paths()

            for path in paths_positions:
                paths.append((color, path))

    rpaths = []
    for color, points in paths:
        p = []
        for point in points:
            n = point.normal

            # normal points outwards, we want our robot to point towards the point
            normal = (-n[0], -n[1], -n[2])

            # No rotation around x-axis
            roll = 0
            pitch = np.arctan2(normal[0], normal[2])
            yaw = np.arcsin(-normal[1])

            # Convert all that to a quaternion
            quat = Rotation.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False
            ).as_quat()

            p.append((*point.coordinates, *quat))

        rpaths.append((color, p))

    return rpaths


if __name__ == "__main__":
    mesh = load_mesh("cube.obj")
    paths = mesh_to_paths(mesh, max_dist=0.024)

    for color, path in paths:
        print(f"Color: {color}")
        for point in path:
            print(f"Point: {point}")
        print()

    print(f"Number of paths: {len(paths)}")
    print(f"Number of points: {sum([len(path) for _, path in paths])}")

    with open("paths.json", "w") as f:
        json.dump(paths, f, indent=4)

    print("Paths exported to paths.json")
