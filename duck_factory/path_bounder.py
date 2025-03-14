from trimesh import Trimesh, load_mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # Compute the bounding box of the mesh
# box = mesh.bounding_box

from collections import deque


def get_bounding_points(mesh: Trimesh) -> Trimesh:
    """Get the bounding points of a mesh."""
    box = mesh.bounding_box_oriented

    # Get the vertices of the bounding box
    vertices = box.vertices

    # Get the min and max values of the bounding box
    min_values = vertices.min(axis=0)
    max_values = vertices.max(axis=0)

    return min_values, max_values


def is_horizontal_or_above(orientation):
    """
    Determines if a given normal orientation is horizontal or above.

    Parameters:
        orientation (tuple): A tuple (nx, ny, nz) representing the normal vector.

    Returns:
        bool: True if the normal is horizontal or above, False if it is pointing downward.
    """
    nx, ny, nz = orientation
    return nz >= 0


# def get_exit_plane_oriented(mesh: Trimesh, orientation: tuple) -> tuple:
#     """
#     Determines the normal of the bounding box plane that will be exited when moving
#     backward along the given orientation, considering an oriented bounding box.

#     Parameters:
#         mesh (trimesh.Trimesh): The input mesh.
#         orientation (tuple): A normal vector (nx, ny, nz) representing the direction.

#     Returns:
#         tuple: The normal vector of the bounding box plane that will be exited.
#     """
#     # Normalize the orientation vector
#     nx, ny, nz = orientation
#     norm = np.linalg.norm([nx, ny, nz])
#     if norm == 0:
#         raise ValueError("Orientation vector cannot be zero.")

#     direction = np.array([nx, ny, nz]) / norm  # Unit direction vector

#     # Get the oriented bounding box
#     obb = mesh.bounding_box_oriented

#     # The bounding box corners (min and max)
#     box_corners = obb.bounds  # Shape (2,3), min and max points

#     # Rotation matrix of the oriented bounding box
#     rotation_matrix = obb.primitive.transform[:3, :3]  # Extract the 3x3 rotation matrix

#     # Define the six plane normals in local OBB space
#     local_normals = np.array(
#         [
#             [-1, 0, 0],
#             [1, 0, 0],  # X-min, X-max
#             [0, -1, 0],
#             [0, 1, 0],  # Y-min, Y-max
#             [0, 0, -1],
#             [0, 0, 1],  # Z-min, Z-max
#         ]
#     )

#     # Transform the normals into world space using the OBB rotation
#     world_normals = local_normals @ rotation_matrix.T

#     # Determine which face is exited
#     max_exit_value = float("-inf")
#     exit_plane = None

#     for normal in world_normals:
#         # Check which face is most aligned with the movement (backward direction)
#         if np.dot(-direction, normal) > 0:
#             if np.dot(normal, box_corners[1]) > max_exit_value:
#                 max_exit_value = np.dot(normal, box_corners[1])
#                 exit_plane = tuple(normal)

#     return exit_plane  # Returns the world-space normal of the exit plane


# def plot_mesh_with_bounding_box(mesh):
#     """
#     Plots a given mesh along with its bounding box in a 3D matplotlib figure.

#     Parameters:
#         mesh (trimesh.Trimesh): The input 3D mesh.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")

#     # Get the bounding box
#     box = mesh.bounding_box_oriented  # Or use mesh.bounding_box for AABB

#     # Plot the bounding box vertices
#     ax.scatter(
#         box.vertices[:, 0],
#         box.vertices[:, 1],
#         box.vertices[:, 2],
#         c="red",
#         label="Bounding Box",
#     )

#     # Plot the bounding box edges
#     edges = [
#         (0, 1),
#         (1, 3),
#         (3, 2),
#         (2, 0),  # Bottom edges
#         (4, 5),
#         (5, 7),
#         (7, 6),
#         (6, 4),  # Top edges
#         (0, 4),
#         (1, 5),
#         (2, 6),
#         (3, 7),  # Vertical edges
#     ]
#     for edge in edges:
#         points = box.vertices[list(edge)]
#         ax.plot(points[:, 0], points[:, 1], points[:, 2], c="blue")

#     # Plot the mesh faces
#     for face in mesh.faces:
#         coords = mesh.vertices[face]
#         ax.add_collection3d(
#             Poly3DCollection([coords], color="gray", alpha=0.3, edgecolor="black")
#         )

#     # Labels and view
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.legend()

#     plt.show()


def ray_box_intersection(ray_origin, ray_direction, box):
    """
    Compute the intersection point of a ray with an oriented bounding box.

    Parameters:
        ray_origin (np.ndarray): The (x, y, z) origin of the ray.
        ray_direction (np.ndarray): The normalized (dx, dy, dz) direction of the ray.
        box (trimesh.primitives.Box): The oriented bounding box.

    Returns:
        np.ndarray or None: The (x, y, z) coordinates of the intersection point or None if no intersection.
    """
    # Extract box transform and half extent
    box_transform = box.primitive.transform
    box_center = box_transform[:3, 3]  # OBB center
    box_axes = box_transform[:3, :3]  # OBB rotation matrix
    box_extents = box.primitive.extents / 2.0  # Half extents

    # Transform ray into OBB local space
    inv_axes = np.linalg.inv(box_axes)
    local_origin = inv_axes @ (ray_origin - box_center)
    local_direction = inv_axes @ ray_direction

    # Ray-box intersection using slab method
    t_min = (-box_extents - local_origin) / local_direction
    t_max = (box_extents - local_origin) / local_direction

    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)

    t_near = np.max(t1)
    t_far = np.min(t2)

    if t_near > t_far or t_far < 0:
        return None  # No intersection

    # Compute intersection point in world space
    intersection_local = local_origin + t_near * local_direction
    intersection_world = (box_axes @ intersection_local) + box_center

    return intersection_world


def get_intersection_with_obb(mesh, point, orientation, precision=3):
    """
    Given a point on the mesh and an orientation, move backward along the normal
    and find the intersection with the oriented bounding box.

    Parameters:
        mesh (trimesh.Trimesh): The input 3D mesh.
        point (np.ndarray): The (x, y, z) coordinates of the starting point.
        orientation (np.ndarray): The (nx, ny, nz) normal at the starting point.
        precision (int): Decimal places to round the coordinates.

    Returns:
        np.ndarray: The (x, y, z) coordinates of the intersection point with the OBB.
    """
    # Normalize the orientation vector
    direction = np.array(orientation)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Orientation vector cannot be zero.")

    direction = -direction / norm  # Reverse direction for backward movement

    # Get the oriented bounding box
    obb = mesh.bounding_box_oriented

    # Compute ray-box intersection
    exit_point = ray_box_intersection(np.array(point), direction, obb)

    if exit_point is None:
        raise RuntimeError(
            "No intersection found between the backward direction and the bounding box."
        )

    exit_point = np.round(exit_point, precision)
    return exit_point


def plot_intersection_with_obb(mesh, start_point, normal_at_point, exit_point):
    """
    Plots the intersection of a backward ray with an oriented bounding box.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        start_point (np.ndarray): The (x, y, z) coordinates of the starting point.
        normal_at_point (np.ndarray): The (nx, ny, nz) normal at the starting point.
        precision (int): Decimal places to round the coordinates.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Get OBB vertices
    obb_vertices = mesh.bounding_box_oriented.vertices
    ax.scatter(
        obb_vertices[:, 0],
        obb_vertices[:, 1],
        obb_vertices[:, 2],
        c="red",
        label="OBB Vertices",
    )

    # Plot OBB edges
    obb_faces = [
        [obb_vertices[i] for i in face] for face in mesh.bounding_box_oriented.faces
    ]
    ax.add_collection3d(Poly3DCollection(obb_faces, alpha=0.3, edgecolor="black"))

    # Plot start point
    ax.scatter(
        start_point[0],
        start_point[1],
        start_point[2],
        c="green",
        s=100,
        label=f"Start Point {tuple(start_point)}",
    )

    # Plot exit point
    ax.scatter(
        exit_point[0],
        exit_point[1],
        exit_point[2],
        c="blue",
        s=100,
        label=f"Exit Point {tuple(exit_point)}",
    )

    # Plot movement direction (backward along normal)
    ax.quiver(
        start_point[0],
        start_point[1],
        start_point[2],
        normal_at_point[0],
        normal_at_point[1],
        normal_at_point[2],
        color="orange",
        length=1.5,
        normalize=True,
        label="Backward Direction",
    )

    num_faces = min(1000, len(mesh.faces))
    random_indices = np.random.choice(len(mesh.faces), num_faces, replace=False)
    subset_faces = mesh.faces[random_indices]
    for face in subset_faces:
        coords = mesh.vertices[face]
        ax.add_collection3d(
            Poly3DCollection([coords], color="gray", alpha=0.3, edgecolor="black")
        )

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()


def generate_path_on_box(start, end, box):
    """
    Generates the shortest path along the edges of an oriented bounding box (OBB).
    The path does not go through the box, only along its edges.

    :param start: Tuple (x, y, z) representing the start point on the edge of the box.
    :param end: Tuple (x, y, z) representing the end point on the edge of the box.
    :param box: A trimesh Trimesh object representing the oriented bounding box.
    :return: List of tuples representing the path along the edges of the box.
    """
    # Get the 8 corner vertices of the box
    vertices = np.array(box.vertices)

    # Get the edges of the box (each edge is a pair of vertex indices)
    edges = set()
    for face in box.faces:
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            edges.add(edge)

    # Build adjacency list for the box edges
    adjacency = {i: set() for i in range(len(vertices))}
    for v1, v2 in edges:
        adjacency[v1].add(v2)
        adjacency[v2].add(v1)

    # Find closest vertices to start and end using NumPy distance computation
    start_idx = np.argmin(np.linalg.norm(vertices - np.array(start), axis=1))
    end_idx = np.argmin(np.linalg.norm(vertices - np.array(end), axis=1))

    # BFS to find shortest path along edges
    queue = deque([(start_idx, [start_idx])])
    visited = set()

    while queue:
        current, path = queue.popleft()

        if current == end_idx:
            result_path = (
                [tuple(map(float, start))]
                + [tuple(map(float, vertices[i])) for i in path]
                + [tuple(map(float, end))]
            )
            return result_path

        if current in visited:
            continue
        visited.add(current)

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return [tuple(map(float, start)), tuple(map(float, end))]


def plot_path_on_box(mesh, start_point1, normal1, start_point2, normal2, paths):
    """
    Plots the generated paths on an oriented bounding box.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        start_point1 (np.ndarray): First start point.
        normal1 (np.ndarray): First normal.
        start_point2 (np.ndarray): Second start point.
        normal2 (np.ndarray): Second normal.
        paths (list of list of np.ndarray): List of paths to be plotted.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    obb_vertices = mesh.bounding_box_oriented.vertices
    ax.scatter(
        obb_vertices[:, 0],
        obb_vertices[:, 1],
        obb_vertices[:, 2],
        c="red",
        label="OBB Vertices",
    )

    obb_faces = [
        [obb_vertices[i] for i in face] for face in mesh.bounding_box_oriented.faces
    ]
    ax.add_collection3d(Poly3DCollection(obb_faces, alpha=0.3, edgecolor="black"))

    # Plot start points
    ax.scatter(
        start_point1[0],
        start_point1[1],
        start_point1[2],
        c="yellow",
        s=100,
        label="Start Point 1",
    )
    ax.scatter(
        start_point2[0],
        start_point2[1],
        start_point2[2],
        c="blue",
        s=100,
        label="Start Point 2",
    )

    # Plot movement direction
    ax.quiver(
        start_point1[0],
        start_point1[1],
        start_point1[2],
        normal1[0],
        normal1[1],
        normal1[2],
        color="orange",
        length=1.5,
        normalize=True,
        label="Backward Direction 1",
    )
    ax.quiver(
        start_point2[0],
        start_point2[1],
        start_point2[2],
        normal2[0],
        normal2[1],
        normal2[2],
        color="purple",
        length=1.5,
        normalize=True,
        label="Backward Direction 2",
    )

    # Plot paths
    paths = np.array(paths)
    ax.plot(
        paths[:, 0],
        paths[:, 1],
        paths[:, 2],
        marker="o",
        linestyle="-",
        color="green",
        label="Path",
    )

    num_faces = min(1000, len(mesh.faces))
    random_indices = np.random.choice(len(mesh.faces), num_faces, replace=False)
    subset_faces = mesh.faces[random_indices]
    for face in subset_faces:
        coords = mesh.vertices[face]
        ax.add_collection3d(
            Poly3DCollection([coords], color="gray", alpha=0.3, edgecolor="black")
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


mesh = load_mesh("DuckComplete.obj")
# plot_mesh_with_bounding_box(mesh)
min_values, max_values = get_bounding_points(mesh)
print(f"Min values: {min_values}")
print(f"Max values: {max_values}")

point = [0.0321, 0.0268, 0.04844]
orientation = (0.3356, 0.0207, 0.9417)

point_2 = [0.0276, 0.11949, -0.0129]
orientation_2 = (-0.42780, 0.84981, -0.307881)

exit_point = get_intersection_with_obb(mesh, point, orientation)
exit_point_2 = get_intersection_with_obb(mesh, point_2, orientation_2)

# orientation_2 = (-1, 1, 0)

# exit_point = [-0.1, -0.1, -0.1]
# exit_point_2 = [0.1, 0.1, 0.1]
print(f"Exit point: {exit_point}")
print(f"Exit point 2: {exit_point_2}")
# plot_intersection_with_obb(mesh, [0, 0, 0], orientation, exit_point)
# plot_intersection_with_obb(mesh, [0, 0, 0], orientation_2, exit_point_2)

path = generate_path_on_box(exit_point, exit_point_2, mesh.bounding_box_oriented)
print(f"Path points: {path}")


plot_path_on_box(mesh, point, orientation, point_2, orientation_2, path)
