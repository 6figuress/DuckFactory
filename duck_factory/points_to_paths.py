import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import trimesh

from duck_factory.point_sampling import SampledPoint, sample_mesh_points, cluster_points


class PointsSimplifier:
    """Simplifies a list of SampledPoint by removing points that are too close to each other and have similar normals."""

    def __init__(
        self, thickness: float, angle_threshold_deg: float, verbose: bool = False
    ):
        """
        Initializes the PointsSimplifier with a thickness and an angle threshold.

        Parameters:
            thickness: Maximum allowed distance between consecutive points
            angle_threshold_deg: Maximum allowed angle between normals of consecutive points in degrees
        """
        self.thickness = thickness
        self.angle_threshold = np.cos(np.radians(angle_threshold_deg))
        self.verbose = verbose

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """
        Normalizes a vector.

        Parameters:
            v: Vector to normalize

        Returns:
            Normalized vector
        """
        length = np.linalg.norm(v)
        return v / length if length != 0 else np.zeros_like(v)

    def simplify(self, points: list[SampledPoint]) -> list[SampledPoint]:
        """
        Simplifies a list of SampledPoint by removing points that are too close to each other and have similar normals.

        Parameters:
            points: List of SampledPoint to simplify

        Returns:
            List of simplified SampledPoint
        """
        if not points:
            return []

        if np.isclose(self.thickness, 0.0):
            return points

        simplified = [points[0]]
        last = points[0]
        last_pos = np.array(last.coordinates)
        last_dir = self._normalize(-np.array(last.normal))

        for point in points[1:]:
            pos = np.array(point.coordinates)

            if np.linalg.norm(pos - last_pos) >= self.thickness:
                direction = self._normalize(-np.array(point.normal))
                alignment = np.dot(last_dir, direction)
                if alignment >= self.angle_threshold:
                    simplified.append(point)
                    last_pos = pos
                    last_dir = direction

        if self.verbose:
            print(f"Simplication : {len(points)} -> {len(simplified)}")
        return simplified


class PathFinder:
    """Class for generating a graph from a point cloud. Finding connected components, and computing paths that ensure consecutive points are within a maximum distance."""

    def __init__(
        self,
        points: list[SampledPoint],
        max_distance: float,
        thickness: float = 0.0,
        angle_threshold_deg: float = 20,
        verbose: bool = False,
    ):
        """
        Initializes the PathFinder with a point cloud and a maximum distance.

        Parameters:
            points: List of SampledPoint representing sampled points
            max_distance: Maximum allowed distance between connected points
            thickness: Size of the simplification thickness
            angle_threshold_deg: Angle threshold for simplification in degrees
            verbose: Display log
        """
        if thickness <= 0.0:
            self.points = points
            if verbose:
                print("No simplification applied.")
        else:
            simplifier = PointsSimplifier(
                thickness, angle_threshold_deg, verbose=verbose
            )
            self.points = simplifier.simplify(points)
        self.max_distance = max_distance
        self.graph = self._create_graph()

    def _create_graph(self) -> nx.Graph:
        """
        Creates a graph where nodes are points and edges connect points within the max_distance limit.

        Nodes are added to the graph with their positions as attributes.
        Edges are added between nodes if the distance between them is less than or equal to max_distance, with the distance as the edge weight.

        Returns:
            NetworkX graph with nodes representing points and edges representing connections within max_distance
            :rtype: nx.Graph
        """
        G = nx.Graph()
        for i, point in enumerate(self.points):
            G.add_node(i, pos=point.coordinates)

        # Extract coordinates for distance matrix
        coords = np.array([point.coordinates for point in self.points])
        dist_matrix = distance_matrix(coords, coords)
        num_points = len(self.points)

        for i in range(num_points):
            for j in range(i + 1, num_points):
                if dist_matrix[i, j] <= self.max_distance:
                    G.add_edge(i, j, weight=dist_matrix[i, j])

        return G

    def find_connected_components(self) -> list[list[int]]:
        """
        Finds the connected components of the graph.

        Returns:
            List of lists containing point indices in each connected component
        """
        return [list(component) for component in nx.connected_components(self.graph)]

    def _solve_path(self, points: list[SampledPoint]) -> list[list[SampledPoint]]:
        """
        Finds paths within a connected component, ensuring that consecutive points respect the distance constraint.

        Args:
            points: List of SampledPoint representing coordinates

        Returns:
            List of paths, each path being a list of SampledPoint
        """
        num_points = len(points)
        if num_points <= 1:
            return [points] if num_points == 1 else []

        unvisited = set(range(num_points))
        paths = []

        while unvisited:
            path = []
            start = unvisited.pop()
            path.append(points[start])

            while unvisited:
                last = path[-1]
                nearest = min(
                    unvisited,
                    key=lambda i: np.linalg.norm(
                        np.array(points[i].coordinates) - np.array(last.coordinates)
                    ),
                )

                # Check distance constraint
                if (
                    np.linalg.norm(
                        np.array(points[nearest].coordinates)
                        - np.array(last.coordinates)
                    )
                    > self.max_distance
                ):
                    break

                unvisited.remove(nearest)
                path.append(points[nearest])

            paths.append(path)

        return paths

    def find_paths(self) -> list[list[SampledPoint]]:
        """
        Finds all possible paths from the connected components.

        Returns:
            List of paths, each path being a list of SampledPoint
        """
        components = self.find_connected_components()
        paths = []

        for component in components:
            component_points = [self.points[i] for i in component]
            component_paths = self._solve_path(component_points)
            paths.extend(component_paths)

        return paths


def plot_paths(paths: list[list[SampledPoint]]) -> None:  # pragma: no cover
    """
    Displays the computed paths in a 3D graph.

    :param paths: List of paths to plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, path in enumerate(paths):
        # Extract coordinates and color from SampledPoint
        coords = np.array([point.coordinates for point in path])
        colors = [
            [point.color[0] / 255, point.color[1] / 255, point.color[2] / 255]
            for point in path
        ]

        if len(coords) > 1:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, marker="o")
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                marker="o",
                linestyle="-",
                label=f"Path {i + 1}",
            )
        else:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="blue", marker="x")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def main() -> None:  # pragma: no cover
    """Main function to execute the PathFinder with a sample mesh and plot the paths."""
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

    mesh = trimesh.load_mesh("DuckComplete.obj")
    points_by_color = sample_mesh_points(mesh, BASE_COLOR, COLORS, n_samples=50000)
    clusters_flat = cluster_points(points_by_color)

    paths = []
    for points, _, _ in clusters_flat:
        path_finder = PathFinder(points, 0.01)
        paths.extend(path_finder.find_paths())

    n_points = sum([len(path) for path in paths])
    print(f"Number of points: {n_points}")

    print(
        f"Number of paths (length > 1): {len([path for path in paths if len(path) > 1])}"
    )
    print(
        f"Number of paths (length <= 1): {len([path for path in paths if len(path) <= 1])}"
    )

    plot_paths(paths)


if __name__ == "__main__":  # pragma: no cover
    main()
