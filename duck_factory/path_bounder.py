from trimesh import Trimesh, load_mesh
import numpy as np
from collections import deque
from duck_factory.reachable_points import PathAnalyzer
from scipy.spatial.distance import cdist

Position = tuple[float, float, float]
Positions = list[Position]
Orientation = tuple[float, float, float, float]
PositionWithOrientation = tuple[Position, Orientation]
Path = list[PositionWithOrientation]


class PathBounder:
    """Class to generate a path on the bounding box of a mesh between two points with orientations. It's used to allow to join multiple paths together."""

    def __init__(
        self,
        mesh_path: str,
        analyzer: PathAnalyzer = None,
        model_points: Positions = None,
        nz_threshold: float = 0.0,
        step_size: float = 0.05,
        precision: float = 1e-6,
    ):
        """
        Initialize the PathBounder object.

        Parameters:
            mesh_path (str): The path to the mesh file to use for the bounding box
            analyzer (PathAnalyzer): The PathAnalyzer object to use for adjusting normals
            model_points (list[tuple[float, float, float]]): The list of model points to use for adjusting normals
            nz_threshold (float): The threshold for the z component of the normal to trigger adjustment
            step_size (float): The step size for the adjustment
            precision (precision): The precision for rounding the intersection points
        """
        self.mesh = load_mesh(mesh_path)
        # self.box = self.mesh.bounding_box_oriented
        self.box = self.mesh.bounding_box
        self.analyzer = analyzer
        self.model_points = model_points
        self.nz_threshold = nz_threshold
        self.step_size = step_size
        self.precision = precision

    def set_analyzer(self, analyzer: PathAnalyzer) -> None:
        """
        Set the PathAnalyzer object.

        Parameters:
            analyzer (PathAnalyzer): The PathAnalyzer object to use for adjusting normals
        """
        self.analyzer = analyzer

    def set_model_points(self, model_points: Positions) -> None:
        """
        Set the list of model points.

        Parameters:
            model_points (list[tuple[float, float, float]]): The list of model points to use for adjusting normals
        """
        self.model_points = model_points

    def set_nz_threshold(self, nz_threshold: float) -> None:
        """
        Set the threshold for the z component of the normal to trigger adjustment.

        Parameters:
            nz_threshold (float): The threshold for the z component of the normal to trigger adjustment
        """
        self.nz_threshold = nz_threshold

    def set_step_size(self, step_size: float) -> None:
        """
        Set the step size for the adjustment.

        Parameters:
            step_size (float): The step size for the adjustment
        """
        self.step_size = step_size

    def get_normal_to_face(self, point1: Position, point2: Position) -> Orientation:
        """
        Get the normal to the face of the bounding box that is closest to the line segment between two points.

        Parameters:
            point1 (tuple[float, float, float]): The first point of the line segment
            point2 (tuple[float, float, float]): The second point of the line segment

        Returns:
            tuple[float, float, float]: The normal to the face of the bounding box that is closest to the line segment
        """
        box_transform = self.box.primitive.transform

        face_normals = box_transform[:3, :3].T

        line_direction = np.array(point2) - np.array(point1)
        line_direction /= np.linalg.norm(line_direction)

        best_normal = None
        best_dot_product = 1

        for normal in face_normals:
            dot_product = np.abs(np.dot(normal, line_direction))
            if dot_product < best_dot_product:
                best_dot_product = dot_product
                best_normal = normal

        # Ensure the normal points outward from the OBB
        centroid_to_p1 = np.array(point1) - self.box.centroid
        if np.dot(centroid_to_p1, best_normal) < 0:
            best_normal = -best_normal

        return -best_normal

    def ray_box_intersection(
        self, origin: Position, direction: Orientation
    ) -> Position:
        """
        Compute the intersection of a ray with the bounding box.

        Parameters:
            origin (tuple[float, float, float]): The origin of the ray
            direction (tuple[float, float, float]): The direction of the ray

        Returns:
            tuple[float, float, float]: The intersection point with the bounding box
        """
        box_transform = self.box.primitive.transform
        box_center = box_transform[:3, 3]
        box_axes = box_transform[:3, :3]
        box_extents = self.box.primitive.extents

        inv_axes = np.linalg.inv(box_axes)
        local_origin = np.dot(inv_axes, np.array(origin) - box_center)
        local_direction = np.dot(inv_axes, direction)

        t_min = (-box_extents - local_origin) / local_direction
        t_max = (box_extents - local_origin) / local_direction

        t_near = np.max(np.minimum(t_min, t_max))
        t_far = np.min(np.maximum(t_min, t_max))

        if t_near > t_far or t_far < 0:
            return None

        intersection = local_origin + t_near * local_direction
        intersection_world = np.dot(box_axes, intersection) + box_center

        return intersection_world

    def ray_plane_intersection(
        self,
        origin: Position,
        direction: Orientation,
        point: Position,
        normal: Orientation,
    ) -> Position | None:
        """
        Compute the intersection of a ray with a extended plane of faces.

        Parameters:
            origin (tuple[float, float, float]): The origin of the ray
            direction (tuple[float, float, float]): The direction of the ray
            point (tuple[float, float, float]): A point on the plane
            normal (tuple[float, float, float]): The normal to the plane

        Returns:
            tuple[float, float, float]: The intersection point with the plane
        """
        ray_origin = np.array(origin)
        ray_direction = np.array(direction) / np.linalg.norm(direction)
        plane_point = np.array(point)
        plane_normal = np.array(normal) / np.linalg.norm(normal)

        denom = np.dot(ray_direction, plane_normal)
        if abs(denom) < self.precision:
            return None  # Ray is parallel to the plane

        d = np.dot(plane_normal, plane_point - ray_origin) / denom
        intersection = ray_origin + d * ray_direction

        return intersection if d >= 0 else None

    def get_intersection_with_obb(
        self, origin: Position, direction: Orientation
    ) -> Position | None:
        """
        Compute the intersection of a ray with the bounding box or its extended faces.

        Parameters:
            origin (tuple[float, float, float]): The origin of the ray
            direction (tuple[float, float, float]): The direction of the ray

        Returns:
            tuple[float, float, float]: The intersection point with the bounding box or its extended faces
        """
        # Compute ray box intersection
        intersection = self.ray_box_intersection(origin, direction)
        if intersection is None:
            # Try intersecting with extended faces
            for face_normal, face_vertex in zip(
                self.box.face_normals, self.box.vertices, strict=False
            ):
                plane_intersection = self.ray_plane_intersection(
                    origin, direction, face_vertex, face_normal
                )
                if intersection is not None:
                    intersection = plane_intersection
                    break
        if intersection is None:
            intersection = np.array(origin)

        return np.round(
            intersection, int(-np.log10(self.precision))
        )  # Round to precision

    def generate_path_on_box(
        self, start: Position, end: Position, restricted_face: list[int] = None
    ) -> Path:
        """
        Generate a path on the bounding box of the mesh between two points.

        Parameters:
            start (tuple[float, float, float]): The start point of the path
            end (tuple[float, float, float]): The end point of the path
            restricted_face (list[int]): The list of face indices to exclude from the path

        Returns:
            list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]: The path on the bounding box
        """
        vertices = np.asarray(self.box.vertices)
        faces = np.asarray(self.box.faces)

        edges = set()
        if restricted_face:
            restricted_face_set = set(restricted_face)
        else:
            restricted_face_set = set()

        for face_idx, face in enumerate(faces):
            if face_idx not in restricted_face_set:
                edges.update(
                    tuple(sorted((face[i], face[(i + 1) % 3]))) for i in range(3)
                )

        # Build adjacency list (dict of sets)
        adjacency = {i: set() for i in range(len(vertices))}
        for v1, v2 in edges:
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)

        start_idx = np.argmin(cdist([start], vertices))
        end_idx = np.argmin(cdist([end], vertices))

        # BFS to find shortest path
        queue = deque([(start_idx, [start_idx])])
        visited = set([start_idx])

        while queue:
            current, path = queue.popleft()
            if current == end_idx:
                result_path = [tuple(vertices[i]) for i in path]

                # Compute normals for each segment
                path_with_normals = []
                for i in range(len(result_path) - 1):
                    normal = tuple(
                        self.get_normal_to_face(result_path[i], result_path[i + 1])
                    )
                    path_with_normals.append((result_path[i], normal))

                path_with_normals.append(
                    (result_path[-1], normal)
                )  # Last point with last normal
                return path_with_normals

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # If no path is found, return start and end with normal
        normal = self.get_normal_to_face(start, end)
        return [(start, normal), (end, normal)]

    def compute_path_with_orientation(
        self,
        start: PositionWithOrientation,
        end: PositionWithOrientation,
        restricted_face: list[int] = None,
    ) -> Path:
        """
        Compute a path on the bounding box of the mesh between two points with orientations.

        Parameters:
            start (tuple[tuple[float, float, float], tuple[float, float, float, float]]): The start point and orientation of the path
            end (tuple[tuple[float, float, float], tuple[float, float, float, float]]): The end point and orientation of the path
            restricted_face (list[int]): The list of face indices to exclude from the path

        Returns:
            list[list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]]: The path on the bounding box
        """

        def adjust_normal_and_get_intersection(
            point: Position, normal: Orientation
        ) -> Path:
            """
            Adjust the normal of a point and compute the intersection with the bounding box.

            Parameters:
                point (tuple[float, float, float]): The point to adjust
                normal (tuple[float, float, float, float]): The normal of the point

            Returns:
                list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]: The adjusted path
            """
            if normal[2] < self.nz_threshold:
                target_normal = (normal[0], normal[1], self.nz_threshold)

                adjusted_positions, adjusted_normals = (
                    self.analyzer.adjust_and_move_backwards(
                        point=point,
                        normal=normal,
                        model_points=self.model_points,
                        target_normal=target_normal,
                        step_size=self.step_size,
                    )
                )

                # Convert all intermediate positions to (position, normal) tuples
                adjusted_path = [
                    (adjusted_positions[i], adjusted_normals[i])
                    for i in range(len(adjusted_positions))
                ]

                # Compute final intersection
                final_intersection = self.get_intersection_with_obb(
                    adjusted_positions[-1], adjusted_normals[-1]
                )

                adjusted_path.append((final_intersection, target_normal))
                return adjusted_path
            else:
                return [(self.get_intersection_with_obb(point, normal), normal)]

        start_adjusted_path = adjust_normal_and_get_intersection(*start)

        end_point, end_normal = end
        end_normal = -np.array(
            end_normal
        )  # Invert the normal to point towards the end point
        end_adjusted_path = adjust_normal_and_get_intersection(end_point, end_normal)
        end_adjusted_path.reverse()

        # Extract the last points from the adjusted paths
        start_exit, _ = start_adjusted_path[-1]
        end_exit, _ = end_adjusted_path[0]

        path = self.generate_path_on_box(
            start_exit, end_exit, restricted_face=restricted_face
        )

        final_path = start_adjusted_path
        final_path.extend(path)
        final_path.extend(end_adjusted_path)

        return final_path


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_path(
    mesh: Trimesh,
    path: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    restricted_face: list[int] = None,
):
    """
    Plots the computed path on the mesh.

    Parameters:
        mesh (Trimesh): The input mesh.
        path (list): A list of (position, normal) tuples representing the path.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    box = mesh.bounding_box
    # box = mesh.bounding_box_oriented

    # Plot mesh bounding box
    obb_vertices = box.vertices

    obb_faces = [[obb_vertices[i] for i in face] for face in box.faces]
    for i, face in enumerate(obb_faces):
        color = (
            "lightblue"
            if restricted_face is None or i not in restricted_face
            else "orange"
        )
        ax.add_collection3d(
            Poly3DCollection([face], alpha=0.3, edgecolor="black", facecolors=color)
        )

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

    # plot the start and end points in red and blue
    start_point, _ = path[0]
    end_point, _ = path[-1]

    ax.scatter(
        start_point[0], start_point[1], start_point[2], c="yellow", label="Start Point"
    )
    ax.scatter(end_point[0], end_point[1], end_point[2], c="orange", label="End Point")

    # Plot normals
    length = 0.05
    for pos, normal in path:
        end_pos = (
            np.array(pos) - np.array(normal) * length
        )  # Ending at the correct point
        ax.quiver(
            end_pos[0],
            end_pos[1],
            end_pos[2],
            normal[0],
            normal[1],
            normal[2],
            color="blue",
            length=length,
            normalize=True,
        )

    num_faces = min(500, len(mesh.faces))
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


from trimesh.sample import sample_surface


if __name__ == "__main__":
    analyzer = PathAnalyzer(
        tube_length=5e1, diameter=2e-2, cone_height=1e-2, step_angle=10, num_vectors=24
    )

    sampled_points = sample_surface(
        load_mesh("DuckComplete.obj"), 5_000, sample_color=False
    )

    all_points = sampled_points[0]

    path_finder = PathBounder(
        "DuckComplete.obj",
        analyzer,
        model_points=all_points,
    )

    start_point = [0.0321, 0.0268, 0.04844]
    start_normal = (0.3356, 0.0207, 0.9417)
    end_point = [0.0276, 0.11949, -0.0129]
    end_normal = (-0.42780, 0.84981, -0.307881)

    restricted_face = [3, 8]

    path_with_orientation = path_finder.compute_path_with_orientation(
        (start_point, start_normal),
        (end_point, end_normal),
        restricted_face=restricted_face,
    )

    print(f"Computed path with {len(path_with_orientation)} points")
    print(path_with_orientation)

    plot_path(path_finder.mesh, path_with_orientation, restricted_face=restricted_face)
