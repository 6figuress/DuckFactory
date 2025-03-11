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
        num_vectors (int): The number of vectors to generate around the cone.

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


def check_path_with_pen(
    data_points: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    model_points: list[tuple[float, float, float]],
    tube_length: float,
    diameter: float,
    cone_height: float,
    cone_angle: float,
    num_vectors: int,
) -> tuple[list, list]:
    """
    Check if a list of points is reachable with a pen structure (cone followed by tube) around the points. If not, generate alternative directions with angle to the normal.

    Parameters:
        data_points (list[tuple]): A list of tuples containing points and their normals ((x, y, z), (nx, ny, nz)).
        model_points (list[tuple]): A list of model points to check for collisions.
        tube_length (float): The length of the cylindrical tube.
        diameter (float): The diameter of both the tube and the cone base.
        cone_height (float): The height of the cone.
        cone_angle (float): The angle in degrees between each generated vector and the normal.
        num_vectors (int): The number of vectors to generate around the cone.

    Returns:
        tuple[list, list]: Two lists of reachable and unreachable points.
    """
    reachable_points = []
    unreachable_points = []

    for point, normal in data_points:
        # Check if the default normal workss
        collision = any(
            is_point_inside_pen(
                model_point, point, normal, tube_length, diameter, cone_height
            )
            for model_point in model_points
        )

        if not collision:
            reachable_points.append(point)
            continue  # This point is already good

        # Generate alternative directions using cone vectors
        alternative_vectors = generate_cone_vectors(normal, cone_angle, num_vectors)
        found_valid = False

        # print("Alternative vectors", alternative_vectors)

        for alt_normal in alternative_vectors:
            collision = any(
                is_point_inside_pen(
                    model_point, point, alt_normal, tube_length, diameter, cone_height
                )
                for model_point in model_points
            )

            if not collision:
                reachable_points.append(point)
                found_valid = True
                print("Found valid alternative")
                break

        if not found_valid:
            print("No valid alternative found")
            unreachable_points.append(point)

    return reachable_points, unreachable_points


import matplotlib.pyplot as plt


def plot_results(reachable_points, unreachable_points, model_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if reachable_points:
        reachable_points = np.array(reachable_points)
        ax.scatter(
            reachable_points[:, 0],
            reachable_points[:, 1],
            reachable_points[:, 2],
            c="g",
            label="Reachable",
        )

    if unreachable_points:
        unreachable_points = np.array(unreachable_points)
        ax.scatter(
            unreachable_points[:, 0],
            unreachable_points[:, 1],
            unreachable_points[:, 2],
            c="r",
            label="Unreachable",
        )

    # if model_points:
    #     model_points = np.array(model_points)
    #     ax.scatter(
    #         model_points[:, 0],
    #         model_points[:, 1],
    #         model_points[:, 2],
    #         c="b",
    #         marker="x",
    #         label="Model Points",
    #     )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    mesh = load_mesh("DuckComplete.obj")
    paths, all_points = mesh_to_paths(mesh)

    all_points = [pt.coordinates for pt in all_points]

    # ------------------------------

    tube_length = 5e1
    diameter = 2e-2
    cone_height = 1e-2
    cone_angle = 30  # degrees
    num_vectors = 15  # Number of alternative directions

    path_points = []
    for color, path in paths:
        for point in path:
            x, y, z, normal = point
            path_points.append(((x, y, z), normal))

    reachable, unreachable = check_path_with_pen(
        path_points,
        all_points,
        tube_length,
        diameter,
        cone_height,
        cone_angle,
        num_vectors,
    )

    # print("Reachable points:", reachable)
    # print("Unreachable points:", unreachable)

    plot_results(reachable, unreachable, all_points)

    # --------------------------------

    # for color, path in paths:
    #     print(f"Color: {color}")
    #     for point in path:
    #         x, y, z, normal = point
    #         nx, ny, nz = normal

    #         for pt in points:
    #             if is_point_inside_pen(
    #                 pt.coordinates,
    #                 (x, y, z),
    #                 normal,
    #                 tube_length=5e1,
    #                 diameter=2e-2,
    #                 cone_height=1e-2,
    #             ):
    #                 print("Point inside pen")

    #                 break

    #  -------------------------

    # for color, path in paths:
    #     print(f"Color: {color}")
    #     for point in path:
    #         x, y, z, normal = point
    #         nx, ny, nz = normal

    #         for pt in points:
    #             if is_point_inside_pen(
    #                 pt.coordinates,
    #                 (x, y, z),
    #                 normal,
    #                 tube_length=5e1,
    #                 diameter=2e-2,
    #                 cone_height=1e-2,
    #             ):
    #                 # Generate new vectors from cone
    #                 new_vectors = generate_cone_vectors(normal, angle=30, num_vectors=8)
    #                 found_solution = False
    #                 for new_vector in new_vectors:
    #                     if not is_point_inside_pen(
    #                         pt.coordinates,
    #                         (x, y, z),
    #                         new_vector,
    #                         tube_length=5e1,
    #                         diameter=2e-2,
    #                         cone_height=1e-2,
    #                     ):
    #                         found_solution = True
    #                         break
    #                 print("out of loop")
    #                 break

    # create a new tuple with ((x, y, z), (nx, ny, nz))

    # -------------------------------------------

    # path_points = []
    # for color, path in paths:
    #     for point in path:
    #         x, y, z, normal = point
    #         path_points.append(((x, y, z), normal))

    # check_points = []

    # while len(path_points) > 0:
    #     point = path_points.pop(0)
    #     x, y, z = point[0]
    #     nx, ny, nz = point[1]

    #     for pt in all_points:
    #         if is_point_inside_pen(
    #             pt.coordinates,
    #             (x, y, z),
    #             point[1],
    #             tube_length=5e1,
    #             diameter=2e-2,
    #             cone_height=1e-2,
    #         ):
    #             check_points.append(point)

    # print("Remaining points to not take normal", len(check_points))

    # # for each points in check_points, generate new vectors from cone. And the check if for one there is no point inside the pen

    # for point in check_points:
    #     print("Point", point)
    #     x, y, z = point[0]
    #     nx, ny, nz = point[1]

    #     # Generate new vectors from cone
    #     new_vectors = generate_cone_vectors(point[1], angle=30, num_vectors=8)
    #     for new_vector in new_vectors:
    #         found_no_solution = True
    #         for pt in all_points:
    #             if not is_point_inside_pen(
    #                 pt.coordinates,
    #                 (x, y, z),
    #                 new_vector,
    #                 tube_length=5e1,
    #                 diameter=2e-2,
    #                 cone_height=1e-2,
    #             ):
    #                 found_no_solution = False
    #                 break
    #         if found_no_solution:
    #             print("Found solution")
    #         else:
    #             print("No solution")

    #     print("out of loop")
    #     break

    # ------------------------------
