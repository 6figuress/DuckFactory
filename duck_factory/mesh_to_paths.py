import numpy as np
from trimesh import Trimesh, load_mesh
from trimesh.sample import sample_surface
from duck_factory.dither_class import Dither
from duck_factory.reachable_points import PathAnalyzer
from duck_factory.points_to_paths import PathFinder
from point_sampling import (
    sample_mesh_points,
    cluster_points,
    Point,
    Color,
)
from scipy.spatial.transform import Rotation
import json

type Vector3 = tuple[float, float, float]
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
    Do the full conversion from a textured mesh to a list of IK-ready paths.

    Convert a mesh to a list of paths, each path being a 3d position and quaternion (x, y, z, w)
    and being of a certain color.

    Args:
        mesh: The mesh to convert to paths
        n_samples: The number of samples to take from the mesh
        max_dist: The maximum distance between two samples for neighborhood

    Returns:
        List of paths, each containing a color and a list of PathPosition (point and quaternion)
    """
    # Dither the mesh's texture
    img = mesh.visual.material.image
    ditherer = Dither(factor=0.1, algorithm="fs", nc=2)
    img = ditherer.apply_dithering(img.convert("RGB"))
    mesh.visual.material.image = img

    # Sample points from the mesh and cluster them by proximity
    sampled_points = sample_mesh_points(
        mesh, base_color=BASE_COLOR, colors=COLORS, n_samples=n_samples
    )

    path_analyzer = PathAnalyzer(
        tube_length=5e1,
        diameter=2e-2,
        cone_height=1e-2,
        step_angle=360 / 10,
        num_vectors=24 / 2,
    )

    # TODO: sample duck + stand
    mesh_points = sample_surface(mesh, n_samples, sample_color=False)

    counter = 0
    valid_points = []
    for point in sampled_points:
        counter += 1

        # Convert from y-up, x-forward, z-right to z-up, x-forward, y-left
        point.coordinates = (
            point.coordinates[0],
            point.coordinates[2],
            -point.coordinates[1],
        )
        point.normal = (
            point.normal[0],
            point.normal[2],
            -point.normal[1],
        )

        # Find a valid orientation for the point
        valid, new_norm = path_analyzer.find_valid_orientation(
            point.coordinates, point.normal, mesh_points[0]
        )

        if valid:
            point.normal = new_norm
        else:
            # Can't find a way to reach this point, skip it
            continue

        valid_points.append(point)

    # Cluster the points by proximity and color
    clusters = cluster_points(valid_points)

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

    # Format the paths
    rpaths = []
    for color, points in paths:
        p = []
        for point in points:
            n = point.normal
            p.append((*point.coordinates, *norm_to_quat(n)))

        rpaths.append((color, p))

    return rpaths


def norm_to_quat(normal: Vector3) -> Quaternion:
    """
    Convert a normal vector to a quaternion, in the robot/simulator coordinate system.

    The normal vector is to be in the same coordinate system as the robot/simulator (z-up, x-forward, y-left),
    and it should be the normal of the surface at which the robot should point (not the direction the robot should point to).

    Args:
        normal: The normal vector to convert

    Returns:
        The quaternion representing the rotation to align the z-axis with the normal

    Raises:
        ValueError: If the resulting quaternion contains NaNs
    """
    # the normal points "away" from the point, we want our robot to point towards it
    normal = (-normal[0], -normal[1], -normal[2])

    # normalize the normal, just to be sure
    normal = normal / np.linalg.norm(normal)

    # handle edge cases where the normal is parallel to the z-axis
    if np.allclose(normal, [0, 0, 1]):
        return (0, 5.06e-4, 0, 9.9e-1)
    elif np.allclose(normal, [0, 0, -1]):
        return (0, 9.9e-1, 0, 5.06e-4)

    # the hand of the robot points towards the positive z-axis, so
    # we want the rotation that aligns the z-axis with our normal

    # find the rotation axis which is perpendicular to both the z-axis and the normal
    # (i.e. the axis perpendicular to the plane they form)
    rotation_axis = np.cross((0, 0, 1), normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # find the rotation angle between the z-axis and the normal
    # this is the angle to rotate around the rotation axis
    # dot(a, b) = |a| * |b| * cos(angle) = cos(angle) because |a| = |b| = 1
    cos_angle = np.dot([0, 0, 1], normal)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # convert the rotation to a quaternion
    r = Rotation.from_rotvec(angle * rotation_axis)
    quat = r.as_quat()

    # fail if the quat contains NaNs
    if np.isnan(quat).any():
        raise ValueError("Quaternion contains NaNs")

    # if one of the 4 components of the quat is too close to +/- 1, there's a risk for it to mess up the IK
    # so set these components to a value close to 1
    close_to_pos_1 = np.isclose(quat, 1, atol=1e-3)
    close_to_min_1 = np.isclose(quat, -1, atol=1e-3)
    quat[close_to_pos_1] = 9.9e-1
    quat[close_to_min_1] = -9.9e-1

    # normalize the quaternion
    quat = quat / np.linalg.norm(quat)

    return quat


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
