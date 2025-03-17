from trimesh import Trimesh, load_mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from duck_factory.reachable_points import PathAnalyzer

# # Compute the bounding box of the mesh
# box = mesh.bounding_box

from collections import deque


def get_bounding_points(mesh: Trimesh) -> Trimesh:
    """
    Get the bounding points of a mesh.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        tuple: A tuple containing the minimum and maximum values of the bounding box.
    """
    box = mesh.bounding_box_oriented

    # Get the vertices of the bounding box
    vertices = box.vertices

    # Get the min and max values of the bounding box
    min_values = vertices.min(axis=0)
    max_values = vertices.max(axis=0)

    return min_values, max_values


def is_horizontal_or_above(orientation: tuple[float, float, float]) -> bool:
    """
    Determines if a given normal orientation is horizontal or above.

    Parameters:
        orientation (tuple): A tuple (nx, ny, nz) representing the normal vector.

    Returns:
        bool: True if the normal is horizontal or above, False if it is pointing downward.
    """
    _, _, nz = orientation
    return nz >= 0


def get_normal_to_face(
    mesh: Trimesh,
    point1: tuple[float, float, float],
    point2: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Computes the normal of the face of the Oriented Bounding Box (OBB) that contains both given points.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        point1 (tuple): The (x, y, z) coordinates of the first point.
        point2 (tuple): The (x, y, z) coordinates of the second point.

    Returns:
        tuple: The normal vector (nx, ny, nz) of the face containing both points.
    """
    obb = mesh.bounding_box_oriented

    # Extract the transformation matrix of the OBB
    obb_transform = obb.primitive.transform

    # The three face normals of the OBB (columns of rotation part of transform matrix)
    face_normals = obb_transform[:3, :3].T

    # Compute the line direction (unit vector)
    line_direction = np.array(point2) - np.array(point1)
    line_direction /= np.linalg.norm(line_direction)  # Normalize

    # Find the plane normal that is most parallel to the line direction
    best_normal = None
    best_dot_product = 1

    for normal in face_normals:
        dot_product = np.abs(np.dot(normal, line_direction))  # Absolute dot product
        if dot_product < best_dot_product:
            best_dot_product = dot_product
            best_normal = normal

    # Ensure the normal points outward from the OBB
    centroid_to_p1 = np.array(point1) - obb.centroid
    if np.dot(centroid_to_p1, best_normal) < 0:
        best_normal = -best_normal

    plane_normal = best_normal

    return plane_normal


def ray_box_intersection(
    ray_origin: tuple[float, float, float],
    ray_direction: tuple[float, float, float],
    box,
) -> tuple[float, float, float]:
    """
    Compute the intersection point of a ray with an oriented bounding box.

    Parameters:
        ray_origin (tuple): The (x, y, z) origin of the ray.
        ray_direction (tuple): The normalized (dx, dy, dz) direction of the ray.
        box (trimesh.primitives.Box): The oriented bounding box.

    Returns:
        tuple or None: The (x, y, z) coordinates of the intersection point or None if no intersection.
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


def get_intersection_with_obb(
    mesh: Trimesh,
    point: tuple[float, float, float],
    orientation: tuple[float, float, float],
    precision: int = 3,
) -> tuple[float, float, float]:
    """
    Compute the intersection point of a backward ray with an oriented bounding box.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        point (tuple): The (x, y, z) coordinates of the starting point.
        orientation (tuple): The (nx, ny, nz) normal at the starting point.
        precision (int): Decimal places to round the coordinates.

    Returns:
        tuple: The (x, y, z) coordinates of the intersection point.

    Raises:
        ValueError: If the orientation vector is zero.
        RuntimeError: If no intersection is found between the backward direction and the bounding box.
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


def generate_path_on_box(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    box: Trimesh,
    restricted_face: list[int] = None,
) -> list:
    """
    Generates the shortest path along the edges of an oriented bounding box (OBB). The path does not go through the box, only along its edges.

    Parameters:
        start: Tuple (x, y, z) representing the start point on the edge of the box.
        end: Tuple (x, y, z) representing the end point on the edge of the box.
        box: A trimesh Trimesh object representing the oriented bounding box.

    Returns:
        List of tuples ((x, y, z), (nx, ny, nz)), where each entry contains a point and its normal.
    """
    # Get the 8 corner vertices of the box
    vertices = np.array(box.vertices)

    # Get the edges of the box (each edge is a pair of vertex indices)
    edges = set()
    for face in box.faces:
        for i in range(3):
            if i in restricted_face:
                continue
            else:
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

            # Compute normals for each point transition
            path_with_normals = []
            previous_normal = None

            for i in range(len(result_path) - 1):
                point1, point2 = result_path[i], result_path[i + 1]
                normal = tuple(get_normal_to_face(box, point1, point2))

                path_with_normals.append((point1, normal))
                previous_normal = normal

            # Add the last point with the last computed normal
            path_with_normals.append((result_path[-1], previous_normal))

            return path_with_normals

        if current in visited:
            continue
        visited.add(current)

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    normal = get_normal_to_face(box, start, end)
    return [(start, normal), (end, normal)]


def plot_path_on_box(
    mesh: Trimesh,
    start_point1: tuple[float, float, float],
    normal1: tuple[float, float, float],
    start_point2: tuple[float, float, float],
    normal2: tuple[float, float, float],
    paths: list,
    restricted_face: list[int] = None,
) -> None:
    """
    Plots the generated paths on an oriented bounding box.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        start_point1 (tuple): The (x, y, z) coordinates of the first starting point.
        normal1 (tuple): The (nx, ny, nz) normal at the first starting point.
        start_point2 (tuple): The (x, y, z) coordinates of the second starting point.
        normal2 (tuple): The (nx, ny, nz) normal at the second starting point.
        paths (list): A list of ((x, y, z), (nx, ny, nz)) containing path positions and their normals.
        restricted_face (list): A list of indices of restricted
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

    for i, face in enumerate(obb_faces):
        color = (
            "lightblue"
            if restricted_face is None or i not in restricted_face
            else "orange"
        )
        ax.add_collection3d(
            Poly3DCollection([face], alpha=0.3, edgecolor="black", facecolors=color)
        )

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
    path_positions = np.array([pos for pos, _ in paths])
    ax.plot(
        path_positions[:, 0],
        path_positions[:, 1],
        path_positions[:, 2],
        marker="o",
        linestyle="-",
        color="green",
        label="Path",
    )

    # Plot normals along the path
    for pos, normal in paths:
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            normal[0],
            normal[1],
            normal[2],
            color="cyan",
            length=0.5,
            normalize=True,
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

point = [0.0321, 0.0268, 0.04844]
orientation = (0.3356, 0.0207, 0.9417)

point_2 = [0.0276, 0.11949, -0.0129]
orientation_2 = (-0.42780, 0.84981, -0.307881)

exit_point = get_intersection_with_obb(mesh, point, orientation)
exit_point_2 = get_intersection_with_obb(mesh, point_2, orientation_2)

restricted_face = [3, 8]

path = generate_path_on_box(
    exit_point, exit_point_2, mesh.bounding_box_oriented, restricted_face
)

for i, (pos, normal) in enumerate(path):
    print(f"Point {i + 1}: {pos} with normal {normal}")

plot_path_on_box(
    mesh, point, orientation, point_2, orientation_2, path, restricted_face
)
