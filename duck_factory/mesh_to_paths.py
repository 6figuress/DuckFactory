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

    # Compute the paths for each cluster
    paths = []
    for points, color, is_noise in clusters:
        if is_noise:
            # The noise clusters contain points that we don't want to connect
            # Create a new path for each point
            for point in points:
                pass
        #                paths.append((color, [point]))
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
            normal = point.normal

            # rotation around z-axis
            yaw = np.degrees(np.arctan2(normal[1], normal[0]))

            # rotation around y-axis (assuming positive y-axis is up)
            pitch = np.degrees(np.arcsin(-normal[2]))

            # No rotation around x-axis
            roll = 0

            # Convert all that to a quaternion
            quat = Rotation.from_euler(
                "xyz", [roll, pitch, yaw], degrees=True
            ).as_quat()

            p.append((*point.coordinates, *quat))

        rpaths.append((color, p))

    return rpaths


if __name__ == "__main__":
    mesh = load_mesh("DuckComplete.obj")
    paths = mesh_to_paths(mesh)

    for color, path in paths:
        print(f"Color: {color}")
        for point in path:
            print(f"Point: {point}")
        print()
