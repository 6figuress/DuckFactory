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


def ray_plane_intersection(
    ray_origin: tuple[float, float, float],
    ray_direction: tuple[float, float, float],
    plane_point: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    """
    Compute the intersection of a ray with an infinite plane.

    Parameters:
        ray_origin (tuple): The (x, y, z) origin of the ray.
        ray_direction (tuple): The normalized (dx, dy, dz) direction of the ray.
        plane_point (tuple): A point on the plane.
        plane_normal (tuple): The normal vector of the plane.

    Returns:
        tuple or None: The (x, y, z) coordinates of the intersection point or None if no intersection
    """
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction) / np.linalg.norm(ray_direction)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)

    denom = np.dot(ray_direction, plane_normal)
    if abs(denom) < 1e-6:
        return None  # Ray is parallel to the plane

    d = np.dot(plane_normal, plane_point - ray_origin) / denom
    intersection = ray_origin + d * ray_direction

    return intersection if d >= 0 else None  # Only forward intersections


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
    """
    # Normalize the orientation vector
    direction = np.array(orientation)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise point

    direction = -direction / norm  # Reverse direction for backward movement

    # Get the oriented bounding box
    obb = mesh.bounding_box_oriented

    # Compute ray-box intersection
    exit_point = ray_box_intersection(np.array(point), direction, obb)

    if exit_point is None:
        # Try intersecting with the extended faces of the OBB
        for face_normal, face_vertex in zip(
            obb.face_normals, obb.vertices, strict=False
        ):
            plane_intersection = ray_plane_intersection(
                point, direction, face_vertex, face_normal
            )
            if plane_intersection is not None:
                exit_point = plane_intersection
                break  # Stop at the first valid intersection

    if exit_point is None:
        exit_point = np.array(point)  # If all else fails, return the original point

    return np.round(exit_point, precision)


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


def compute_path_with_orientation(
    mesh: Trimesh,
    start_point: tuple[float, float, float],
    start_normal: tuple[float, float, float],
    end_point: tuple[float, float, float],
    end_normal: tuple[float, float, float],
    analyzer: PathAnalyzer,
    model_points: list[tuple[float, float, float]],
    nz_threshold: float = 0.0,
    step_size: float = 0.05,
    restricted_faces: list[int] = None,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """
    Compute a path with orientations from a start position to an end position, ensuring normals are adjusted if needed.

    Parameters:
        mesh (Trimesh): The input mesh.
        start_point (tuple): The (x, y, z) coordinates of the start position.
        start_normal (tuple): The normal (nx, ny, nz) at the start position.
        end_point (tuple): The (x, y, z) coordinates of the end position.
        end_normal (tuple): The normal (nx, ny, nz) at the end position.
        analyzer (PathAnalyzer): The reachability analyzer.
        model_points (list): A list of model points to check for collisions.
        nz_threshold (float): The threshold below which normals should be adjusted.
        step_size (float): The step size for adjusting orientation.
        restricted_faces (list): List of restricted face indices on the OBB.

    Returns:
        list: A list of (position, normal) tuples representing the path.
    """

    # Adjust normals if below threshold and compute intersection points
    def process_point(
        point: tuple[float, float, float], normal: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        Process a point and its normal, adjusting the normal if needed and computing the intersection point.

        Parameters:
            point (tuple): The (x, y, z) coordinates of the point.
            normal (tuple): The normal (nx, ny, nz) at the point.

        Returns:
            tuple: A tuple containing the intersection point and the adjusted normal.
        """
        if normal[2] < nz_threshold:
            target_normal = (normal[0], normal[1], 0)
            adjusted_positions, adjusted_normals = analyzer.adjust_and_move_backwards(
                point, normal, model_points, target_normal, step_size
            )
            final_point = get_intersection_with_obb(
                mesh, adjusted_positions[-1], adjusted_normals[-1]
            )
            return final_point, adjusted_normals[-1]
        else:
            return get_intersection_with_obb(mesh, point, normal), normal

    # Process start and end points
    start_exit, start_normal = process_point(start_point, start_normal)
    end_exit, end_normal = process_point(end_point, end_normal)

    # Generate path along the OBB
    path = generate_path_on_box(
        start_exit, end_exit, mesh.bounding_box_oriented, restricted_faces
    )

    # Add start and end points with their normals
    path.insert(0, (start_point, start_normal))
    path.append((end_point, end_normal))

    return path


from duck_factory.pointsToPaths import PathFinder
from duck_factory.point_sampling import (
    sample_mesh_points,
    cluster_points,
    Point,
    Color,
)


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


def mesh_to_paths(mesh: Trimesh, n_samples: int = 50_000, max_dist: float = 0.1):
    """
    Convert a mesh to a list of paths by sampling points and clustering them.

    Parameters:
        mesh (Trimesh): The mesh to convert to paths.
        n_samples (int): The number of points to sample.
        max_dist (float): The maximum distance between points to consider them connected.

    Returns:
        list[Path]: A list of paths with colors and points
        list[Point]: A list of all sampled points
    """
    sampled_points = sample_mesh_points(
        mesh, base_color=BASE_COLOR, colors=COLORS, n_samples=n_samples
    )

    clusters = cluster_points(sampled_points)

    all_points = [point for cluster in clusters for point in cluster[0]]
    paths = []
    for points, color, is_noise in clusters:
        if not is_noise:
            path_finder = PathFinder(points, max_dist)
            found_paths = path_finder.find_paths()
            formatted_paths = [
                (
                    point.coordinates[0],
                    point.coordinates[1],
                    point.coordinates[2],
                    point.normal,
                )
                for path in found_paths
                for point in path
            ]
            paths.append((color, formatted_paths))

    return paths, all_points


mesh = load_mesh("DuckComplete.obj")

point = [0.0321, 0.0268, 0.04844]
orientation = (0.3356, 0.0207, 0.9417)

point_2 = [0.0276, 0.11949, -0.0129]
orientation_2 = (-0.42780, 0.84981, -0.307881)

# exit_point = get_intersection_with_obb(mesh, point, orientation)
# exit_point_2 = get_intersection_with_obb(mesh, point_2, orientation_2)

restricted_face = [3, 8]

# path = generate_path_on_box(
#     exit_point, exit_point_2, mesh.bounding_box_oriented, restricted_face
# )

# for i, (pos, normal) in enumerate(path):
#     print(f"Point {i + 1}: {pos} with normal {normal}")

# plot_path_on_box(
#     mesh, point, orientation, point_2, orientation_2, path, restricted_face
# )


paths, all_points = mesh_to_paths(mesh)
all_points = [pt.coordinates for pt in all_points]

analyzer = PathAnalyzer(
    tube_length=5e1, diameter=2e-2, cone_height=1e-2, step_angle=10, num_vectors=24
)
model_points = all_points
nz_threshold = 0.0
step_size = 0.05

path_with_orientation = compute_path_with_orientation(
    mesh,
    point,
    orientation,
    point_2,
    orientation_2,
    analyzer,
    model_points,
    nz_threshold,
    step_size,
    restricted_face,
)

print(path_with_orientation)


def plot_path(
    mesh: Trimesh,
    path: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
):
    """
    Plots the computed path on the mesh.

    Parameters:
        mesh (Trimesh): The input mesh.
        path (list): A list of (position, normal) tuples representing the path.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot mesh bounding box
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
    for face in obb_faces:
        ax.add_collection3d(Poly3DCollection([face], alpha=0.3, edgecolor="black"))

    # Plot path
    path_positions = [pos for pos, _ in path]
    path_positions = list(map(list, zip(*path_positions, strict=False)))
    ax.plot(
        path_positions[0],
        path_positions[1],
        path_positions[2],
        marker="o",
        linestyle="-",
        color="green",
        label="Path",
    )

    # Plot normals
    for pos, normal in path:
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            normal[0],
            normal[1],
            normal[2],
            color="blue",
            length=0.05,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


plot_path(mesh, path_with_orientation)
