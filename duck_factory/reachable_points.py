import numpy as np
from trimesh import Trimesh, load_mesh
from duck_factory.points_to_paths import PathFinder
from duck_factory.point_sampling import (
    sample_mesh_points,
    cluster_points,
    Point,
    Color,
)

Quaternion = tuple[float, float, float, float]
PathPosition = tuple[*Point, *Quaternion]
Path = tuple[Color, list[PathPosition]]

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
) -> tuple[list[Path], list[Point]]:
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


class PathAnalyzer:
    """
    Class to analyze path reachability using a pen structure consisting of a cone followed by a tube.

    Attributes:
        tube_length (float): The length of the cylindrical tube.
        diameter (float): The diameter of both the tube and the cone base.
        cone_height (float): The height of the cone.
        step_angle (float): The angle in degrees to increment the normal during alternative searches.
        num_vectors (int): The number of vectors to generate around the normal.
    """

    def __init__(
        self,
        tube_length: float,
        diameter: float,
        cone_height: float,
        step_angle: int,
        num_vectors: int,
    ):
        self.tube_length = tube_length
        self.diameter = diameter
        self.cone_height = cone_height
        self.step_angle = step_angle
        self.num_vectors = num_vectors

    def is_reachable(
        self,
        point: tuple[float, float, float],
        normal: tuple[float, float, float],
        model_points: list[tuple[float, float, float]],
    ) -> bool:
        """
        Check if a given point with a specific normal is reachable without collision.

        Parameters:
            point (tuple): The (x, y, z) coordinates of the point.
            normal (tuple): The (dx, dy, dz) normal vector defining the main axis.
            model_points (list): A list of points representing the model to avoid.

        Returns:
            bool: True if the point is reachable, False otherwise.
        """
        return not self.are_points_inside_pen(
            np.array(model_points),
            point,
            normal,
        )

    def are_points_inside_pen(
        self,
        points: np.ndarray[tuple[float, float, float]],
        p0: tuple[float, float, float],
        normal: tuple[float, float, float],
    ) -> bool:
        """
        Check if a list of 3D points are inside a structure consisting of a cone followed by a tube, representing the pen.

        Parameters:
            points: NumPy array containing the (x, y, z) coordinates of the points.
            p0: The (x, y, z) coordinates of the cone tip (start of structure).
            normal: The (dx, dy, dz) normal vector defining the axis direction.

        Returns:
            bool: True if any of the points are inside the tube or the cone, False otherwise.
        """
        points = np.array(points)
        tip = np.array(p0)
        norm = np.array(normal) / np.linalg.norm(normal)  # Normalize

        # Exclude the cone tip explicitly
        not_cone_tip = ~np.all(points == tip, axis=1)

        # Early retunr if all points are cone tip
        if not np.any(not_cone_tip):
            return False

        # Filter out cone tip points
        points = points[not_cone_tip]

        # Calculate projections
        vecs_to_a = points - tip
        projection = np.dot(vecs_to_a, norm)

        # FIlter points by projection range
        mask_cone = (0 <= projection) & (projection <= self.cone_height)
        mask_tube = (self.cone_height < projection) & (projection <= self.tube_length)

        # Calculate distances from axis
        closest_points = tip + np.outer(projection, norm)
        distances = np.linalg.norm(points - closest_points, axis=1)

        # Check if points are inside the cone or tube
        radius = self.diameter / 2
        result_cone = np.zeros_like(projection, dtype=bool)
        if np.any(mask_cone):
            max_radius = radius * (projection[mask_cone] / self.cone_height)
            result_cone[mask_cone] = distances[mask_cone] <= max_radius

        result_tube = np.zeros_like(projection, dtype=bool)
        if np.any(mask_tube):
            result_tube[mask_tube] = distances[mask_tube] <= radius

        return np.any(result_cone | result_tube)

    def adjust_orientation_toward_target(
        self,
        point: tuple[float, float, float],
        normal: tuple[float, float, float],
        model_points: list[tuple[float, float, float]],
        target_normal: tuple[float, float, float],
    ) -> np.ndarray:
        """
        Adjust the orientation of the normal vector towards the target normal vector.

        Parameters:
            point (tuple): The (x, y, z) coordinates of the point.
            normal (tuple): The (dx, dy, dz) normal vector defining the main axis.
            model_points (list): A list of points representing the model to avoid.
            target_normal (tuple): The (dx, dy, dz) target normal vector.

        Returns:
            np.ndarray: The adjusted normal vector.
        """
        normal = np.array(normal)
        target_normal = np.array(target_normal)

        step = 0
        adjusted_normal = np.array(normal)
        while step <= 1.0:
            for i in range(3):  # Adjust each component separately (nx, ny, nz)
                adjusted_value = np.longdouble((1 - step)) * np.longdouble(
                    target_normal[i]
                ) + step * np.longdouble(normal[i])

                adjusted_normal[i] = adjusted_value
            # if np.all(np.isclose(adjusted_normal, target_normal, atol=1e-5)):
            if self.is_reachable(point, adjusted_normal, model_points):
                return adjusted_normal

            step += 0.1  # Move towards the original normal gradually
            # step += self.step_angle / 90  # Move towards the original normal gradually

        return normal  # Return original normal if no reachable alternative found

    def move_backwards(
        self,
        point: tuple[float, float, float],
        normal: tuple[float, float, float],
        distance: float = 0.05,
    ) -> tuple[float, float, float]:
        """Generate new position by moving backwards along the normal direction of a given distance.

        Returns:
            tuple: The new position after moving backwards.
        """
        return point - distance * np.array(normal)

    def adjust_and_move_backwards(
        self,
        point: tuple[float, float, float],
        normal: tuple[float, float, float],
        model_points: list[tuple[float, float, float]],
        target_normal: tuple[float, float, float],
        step_size: float = 0.05,
    ) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        """Adjust orientation and move backwards until the target normal is achieved.

        Returns:
            tuple: A tuple containing a list of points and a list of orientations.
        """
        point = point
        points = [point]
        orientations = [normal]
        while not np.all(np.isclose(normal, target_normal)):
            normal = self.adjust_orientation_toward_target(
                point, normal, model_points, target_normal
            )
            point = self.move_backwards(point, normal, step_size)
            points.append(point)
            orientations.append(normal)

        return points, orientations

    def generate_cone_vectors(
        self, normal: tuple[float, float, float], angle: float
    ) -> list[np.ndarray]:
        """
        Generate multiple vectors that form a cone shape around the normal, all at a given angle.

        Parameters:
            normal (tuple): The (dx, dy, dz) normal vector defining the main axis.
            angle (float): The angle in degrees between each generated vector and the normal.

        Returns:
            list[np.ndarray]: A list of unit vectors forming the cone.
        """
        normal = np.array(normal) / np.linalg.norm(normal)
        arbitrary_vector = (
            np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        )

        perp_vector = np.cross(normal, arbitrary_vector)
        perp_vector = perp_vector / np.linalg.norm(perp_vector)

        angle_rad = np.radians(angle)

        # Generate multiple vectors
        cone_vectors = []
        for i in range(self.num_vectors):
            theta = (
                2 * np.pi * i
            ) / self.num_vectors  # Evenly space angles around the normal
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

    def find_valid_orientation(
        self,
        point: tuple[float, float, float],
        normal: tuple[float, float, float],
        model_points: list[tuple[float, float, float]],
    ) -> tuple[bool, tuple[float, float, float]]:
        """
        Find a valid orientation if the default one causes a collision.

        Parameters:
            point (tuple): The (x, y, z) coordinates of the point.
            normal (tuple): The (dx, dy, dz) normal vector defining the main axis.
            model_points (list): A list of points representing the model to avoid.

        Returns:
            tuple[bool, tuple[float, float, float]]: A tuple containing a boolean indicating if a valid orientation was found and the valid normal direction.
        """
        if self.is_reachable(point, normal, model_points):
            return True, normal  # Default normal is valid

        for angle in range(self.step_angle, 91, self.step_angle):
            alternative_vectors = self.generate_cone_vectors(normal, angle)

            for alt_normal in alternative_vectors:
                if self.is_reachable(point, alt_normal, model_points):
                    # print(f"Found a valid alternative orientation at {angle}°")
                    return True, alt_normal  # Found a valid alternative

        # print("No valid orientation found")
        return False, normal  # No valid orientation foundreturn False, normal

    def filter_reachable_points(
        self,
        data_points: list[
            tuple[tuple[float, float, float], tuple[float, float, float]]
        ],
        model_points: list[tuple[float, float, float]],
    ) -> tuple[
        list[tuple[tuple[float, float, float], tuple[float, float, float]]],
        list[tuple[float, float, float]],
    ]:
        """
        Filter a list of data points to keep only the reachable ones.

        Parameters:
            data_points (list): A list of (point, normal) tuples to filter.
            model_points (list): A list of points representing the model to avoid.
            diameter (float): The diameter of both the tube and the cone base.

        Returns:
            tuple: A tuple containing a list of updated data points with reachable orientation and a list of unreachable points.
        """
        updated_data_points = []
        unreachable_points = []
        for point, normal in data_points:
            valid, new_normal = self.find_valid_orientation(point, normal, model_points)
            if valid:
                updated_data_points.append((point, new_normal))
            else:
                unreachable_points.append(point)
        return updated_data_points, unreachable_points


def filter_negative_nz_points(
    path_points: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """
    Filter points where the normal's z-component (nz) is negative.

    Parameters:
        path_points (list): List of tuples containing (point, normal)

    Returns:
        list: Filtered list of points and normal with nz < 0
    """
    filtered_points = [
        (point, normal) for point, normal in path_points if normal[2] < 0
    ]
    return filtered_points


if __name__ == "__main__":  # pragma: no cover
    analyzer = PathAnalyzer(
        tube_length=5e1, diameter=2e-2, cone_height=1e-2, step_angle=10, num_vectors=24
    )

    mesh = load_mesh("DuckComplete.obj")
    paths, all_points = mesh_to_paths(mesh)
    all_points = [pt.coordinates for pt in all_points]
    path_points = [
        ((x, y, z), normal) for color, path in paths for x, y, z, normal in path
    ]

    # updated_points, unreachable_points = analyzer.filter_reachable_points(
    #     path_points, all_points
    # )

    # print(f"Number of reachable points: {len(updated_points)}")
    # print(f"Number of unreachable points: {len(unreachable_points)}")

    point = [0.1, 0.1, 0.1]
    normal = [0, 0, -1]
    target = [normal[0], normal[1], 0]

    # new_norm = analyzer.adjust_orientation(point, normal, all_points, target)

    # print(f"Original normal: {normal}")
    # print(f"Adjusted normal: {new_norm}")
    # print(f"Target normal: {target}")

    filtered_points = filter_negative_nz_points(path_points)
    # filtered_points = filter_negative_nz_points(updated_points)

    for point, normal in filtered_points:
        target = [normal[0], normal[1], 0]
        if not analyzer.is_reachable(point, target, all_points):
            poi, dir = analyzer.adjust_and_move_backwards(
                point, normal, all_points, target
            )

            print(poi + dir)

            # new_norm = analyzer.adjust_orientation_toward_target(
            #     point, normal, all_points, target
            # )
            # print(f"Original normal: {normal}")
            # print(f"Adjusted normal: {new_norm}")
            # print(f"Target normal: {target}")
            # print("")
