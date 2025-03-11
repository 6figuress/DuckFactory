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
    all_points = []
    for points, color, is_noise in clusters:
        all_points.extend(points)
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

            """ # Convert all that to a quaternion
            quat = Rotation.from_euler(
                "xyz", [roll, pitch, yaw], degrees=True
            ).as_quat()

            p.append((*point.coordinates, *quat)) """

            p.append((*point.coordinates, normal))

        rpaths.append((color, p))

    return rpaths, all_points


def is_point_in_tube(
    point: tuple[float, float, float],
    p0: tuple[float, float, float],
    normal: tuple[float, float, float],
    diameter: float,
    start_dist: float = 0,
) -> bool:
    """
    Check if a 3D point is inside an infinite tube defined by a point (p0), a normal (direction), and radius d.

    Parameters:
        point (tuple): The (x, y, z) coordinates of the point.
        p0 (tuple): The (x, y, z) point through which the tube passes.
        normal (tuple): The (dx, dy, dz) normal vector defining the tube's axis.
        d (float): The radius of the tube.

    Returns:
        bool: True if the point is inside the tube, False otherwise.
    """
    p = np.array(point)
    a = np.array(p0)
    n = np.array(normal)

    # Normalize the normal direction
    n = n / np.linalg.norm(n)

    # Compute the projection of (p - a) onto the normal
    projection_length = np.dot(p - a, n)

    # Ensure the point is in the correct direction (one-sided tube)
    if projection_length < start_dist:
        return np.False_  # Point is behind p0, so it's outside the tube

    # Closest point on the tube's axis
    closest_point = a + projection_length * n

    # Compute distance from the point to the axis
    distance = np.linalg.norm(p - closest_point)

    return distance <= (diameter / 2)


def is_point_in_cone(
    point: tuple[float, float, float],
    p0: tuple[float, float, float],
    normal: tuple[float, float, float],
    height: float,
    diameter: float,
) -> bool:
    """
    Check if a 3D point is inside a cone defined by a tip point (p0), axis (normal), height, and base diameter.

    Parameters:
        point (tuple): The (x, y, z) coordinates of the point.
        p0 (tuple): The (x, y, z) coordinates of the cone tip.
        normal (tuple): The (dx, dy, dz) normal vector defining the cone's axis.
        height (float): The height of the cone.
        diameter (float): The diameter of the cone's base.

    Returns:
        bool: True if the point is inside the cone, False otherwise.
    """
    p = np.array(point)
    a = np.array(p0)
    n = np.array(normal)

    # Normalize the normal direction
    n = n / np.linalg.norm(n)

    # Compute the projection of (p - a) onto the normal (height along axis)
    projection_length = np.dot(p - a, n)

    # Ensure the point is in the valid height range (inside the finite cone)
    if projection_length < 0 or projection_length > height:
        return np.False_  # Point is outside the height limits

    # Compute the radius of the cone at this height (linear interpolation)
    max_radius = (diameter / 2) * (projection_length / height)

    # Closest point on the cone axis
    closest_point = a + projection_length * n

    # Compute perpendicular distance from the axis
    distance_from_axis = np.linalg.norm(p - closest_point)

    return distance_from_axis <= max_radius


def is_point_inside_pen(
    point: tuple[float, float, float],
    p0: tuple[float, float, float],
    normal: tuple[float, float, float],
    tube_length: float,
    diameter: float,
    cone_height: float,
) -> bool:
    """
    Check if a 3D point is inside a structure consisting of a cone followed by a tube, representing the pen.

    Parameters:
        point (tuple): The (x, y, z) coordinates of the point.
        p0 (tuple): The (x, y, z) coordinates of the cone tip (start of structure).
        normal (tuple): The (dx, dy, dz) normal vector defining the axis direction.
        tube_length (float): The length of the cylindrical tube.
        diameter (float): The diameter of both the tube and the cone base.
        cone_height (float): The height of the cone.

    Returns:
        bool: True if the point is inside the tube or the cone, False otherwise.
    """
    p = np.array(point)
    a = np.array(p0)
    n = np.array(normal)

    # Exclude the point p0 explicitly
    if np.all(p == a):
        return False

    # Normalize the normal direction
    n = n / np.linalg.norm(n)

    # Compute the projection of (p - a) onto the normal (height along axis)
    projection_length = np.dot(p - a, n)

    radius = diameter / 2

    # Compute the closest point on the axis (for both tube and cone)
    closest_point = a + projection_length * n

    # Compute perpendicular distance from the axis
    distance_from_axis = np.linalg.norm(p - closest_point)

    # Check if point is inside the cone
    if 0 <= projection_length <= cone_height:
        max_radius = radius * (projection_length / cone_height)
        if distance_from_axis <= max_radius:
            return True

    # Check if point is inside the tube
    projection_length_tube = projection_length - cone_height

    if 0 <= projection_length_tube <= tube_length:
        if distance_from_axis <= radius:
            return True

    return False  # Not inside the tube or cone


def generate_cone_vectors(
    normal: tuple[float, float, float], angle: float, num_vectors: int
) -> list[np.ndarray]:
    """
    Generate multiple vectors that form a cone shape around the normal, all at a given angle.

    Parameters:
        normal (tuple): The (dx, dy, dz) normal vector defining the main axis.
        angle (float): The angle in degrees between each generated vector and the normal.
        num_vectors (int): The number of vectors to generate around the normal.

    Returns:
        list[np.ndarray]: A list of unit vectors forming the cone.
    """
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # Ensure the normal is a unit vector

    # Create an arbitrary perpendicular vector
    arbitrary_vector = (
        np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    )
    perp_vector = np.cross(normal, arbitrary_vector)
    perp_vector = perp_vector / np.linalg.norm(perp_vector)

    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Generate multiple vectors
    cone_vectors = []
    for i in range(num_vectors):
        theta = (2 * np.pi * i) / num_vectors  # Evenly space angles around the normal

        # Rotate perp_vector around the normal to distribute vectors evenly
        rotated_perp = np.cos(theta) * perp_vector + np.sin(theta) * np.cross(
            normal, perp_vector
        )

        # Rotate this new vector by the specified angle
        rotated_vector = (
            np.cos(angle_rad) * normal
            + np.sin(angle_rad) * rotated_perp
            + (1 - np.cos(angle_rad)) * np.dot(normal, rotated_perp) * rotated_perp
        )

        cone_vectors.append(rotated_vector)

    return cone_vectors


def is_reachable(
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
    model_points: list[tuple[float, float, float]],
    tube_length: float,
    diameter: float,
    cone_height: float,
) -> bool:
    """Check if a given point with a specific normal is reachable with the pen without collision to the model.

    Parameters:
        point (tuple): The (x, y, z) coordinates of the point.
        normal (tuple): The (dx, dy, dz) normal vector defining the main axis.
        model_points (list): A list of points representing the model to avoid.
        tube_length (float): The length of the cylindrical tube.
        diameter (float): The diameter of both the tube and the cone base.
        cone_height (float): The height of the cone.

    Returns:
        bool: True if the point is reachable, False otherwise.
    """
    return not any(
        is_point_inside_pen(
            model_point, point, normal, tube_length, diameter, cone_height
        )
        for model_point in model_points
    )


def find_valid_orientation(
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
    model_points: list[tuple[float, float, float]],
    tube_length: float,
    diameter: float,
    cone_height: float,
    normal_angle: float,
    num_vectors: int,
) -> tuple[bool, tuple[float, float, float]]:
    """Find a valid orientation if the default one causes a collision.

    Parameters:
        point (tuple): The (x, y, z) coordinates of the point.
        normal (tuple): The (dx, dy, dz) normal vector defining the main axis.
        model_points (list): A list of points representing the model to avoid.
        tube_length (float): The length of the cylindrical tube.
        diameter (float): The diameter of both the tube and the cone base.
        cone_height (float): The height of the cone.
        normal_angle (float): The angle in degrees between each generated vector and the normal.
        num_vectors (int): The number of vectors to generate around the normal.

    Returns:
        tuple[bool, tuple[float, float, float]]: A tuple containing a boolean indicating if a valid orientation was found and the valid normal direction.
    """
    if is_reachable(point, normal, model_points, tube_length, diameter, cone_height):
        return True, normal  # Default normal is valid

    alternative_vectors = generate_cone_vectors(normal, normal_angle, num_vectors)
    for alt_normal in alternative_vectors:
        if is_reachable(
            point, alt_normal, model_points, tube_length, diameter, cone_height
        ):
            print("Found a valid alternative orientation")
            return True, alt_normal  # Found a valid alternative

    print("No valid orientation found")
    return False, normal  # No valid orientation found


def filter_reachable_points(
    data_points: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    model_points: list[tuple[float, float, float]],
    tube_length: float,
    diameter: float,
    cone_height: float,
    normal_angle: float,
    num_vectors: int,
) -> tuple[
    list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    list[tuple[float, float, float]],
]:
    """Filter a list of data points to keep only the reachable ones.

    Parameters:
        data_points (list): A list of (point, normal) tuples to filter.
        model_points (list): A list of points representing the model to avoid.
        tube_length (float): The length of the cylindrical tube.
        diameter (float): The diameter of both the tube and the cone base.
        cone_height (float): The height of the cone.
        normal_angle (float): The angle in degrees between each generated vector and the normal.
        num_vectors (int): The number of vectors to generate around the normal.

    Returns:
        tuple: A tuple containing a list of updated data points with reachable orientation and a list of unreachable points.
    """
    updated_data_points = []
    unreachable_points = []

    for point, normal in data_points:
        valid, new_normal = find_valid_orientation(
            point,
            normal,
            model_points,
            tube_length,
            diameter,
            cone_height,
            normal_angle,
            num_vectors,
        )
        if valid:
            updated_data_points.append((point, new_normal))
        else:
            unreachable_points.append(point)

    return updated_data_points, unreachable_points


if __name__ == "__main__":
    tube_length = 5e1
    diameter = 2e-2
    cone_height = 1e-2
    normal_angle = 30  # degrees
    num_vectors = 15  # Number of alternative directions

    mesh = load_mesh("DuckComplete.obj")
    paths, all_points = mesh_to_paths(mesh)

    all_points = [pt.coordinates for pt in all_points]

    path_points = [
        ((x, y, z), normal) for color, path in paths for x, y, z, normal in path
    ]

    updated_points, unreachable_points = filter_reachable_points(
        path_points,
        all_points,
        tube_length,
        diameter,
        cone_height,
        normal_angle,
        num_vectors,
    )

    print(f"Number of reachable points: {len(updated_points)}")
    print(f"Number of unreachable points: {len(unreachable_points)}")
