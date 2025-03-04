import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

from duck_factory.model3d import Mesh

import random


class PathFinder:
    """Class for generating a graph from a point cloud. Finding connected components, and computing paths that ensure consecutive points are within a maximum distance."""

    def __init__(self, points: list[tuple[float, float, float]], max_distance: float):
        """
        Initializes the PathFinder with a point cloud and a maximum distance.

        :param points: List of tuples representing (x, y, z) coordinates
        :param max_distance: Maximum allowed distance between connected points
        """
        self.points = np.array(points)
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
        for i, (x, y, z) in enumerate(self.points):
            G.add_node(i, pos=(x, y, z))

        dist_matrix = distance_matrix(self.points, self.points)
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

    def _solve_path(
        self, points: list[tuple[float, float, float]]
    ) -> list[list[tuple[float, float, float]]]:
        """
        Finds paths within a connected component, ensuring that consecutive points respect the distance constraint.

        :param points: List of tuples representing (x, y, z) coordinates

        Returns:
            List of paths, each path being a list of (x, y, z) tuples
        """
        num_points = len(points)
        if num_points <= 1:
            return [points] if num_points == 1 else []

        unvisited = set(range(num_points))
        paths = []

        while unvisited:
            path = []
            start = unvisited.pop()
            path.append(tuple(points[start]))

            while unvisited:
                last = path[-1]
                nearest = min(unvisited, key=lambda i: np.linalg.norm(points[i] - last))

                # Check distance constraint
                if np.linalg.norm(points[nearest] - last) > self.max_distance:
                    break

                unvisited.remove(nearest)
                path.append(tuple(points[nearest]))

            paths.append(path)

        return paths

    def find_paths(self) -> list[list[tuple[float, float, float]]]:
        """
        Finds all possible paths from the connected components.

        Returns:
            List of paths, each path being a list of (x, y, z) tuples
        """
        components = self.find_connected_components()
        paths = []

        for component in components:
            component_points = [self.points[i] for i in component]
            component_paths = self._solve_path(component_points)
            paths.extend(component_paths)

        return paths

    def plot_paths(self, paths: list[list[tuple[float, float, float]]]) -> None:
        """
        Displays the computed paths in a 3D graph.

        :param paths: List of paths to plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i, path in enumerate(paths):
            path = np.array(path)
            if len(path) > 1:
                ax.scatter(path[:, 0], path[:, 1], path[:, 2], c="red", marker="o")
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    marker="o",
                    linestyle="-",
                    label=f"Path {i + 1}",
                )
            else:
                ax.scatter(path[:, 0], path[:, 1], path[:, 2], c="blue", marker="x")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()


def main() -> None:
    """Main function to execute the PathFinder with a sample mesh and plot the paths."""
    mesh = Mesh(
        "duck_factory/model.obj",
        "duck_factory/Génère_moi_un_canar_0219074804_texture.png",
    )
    points = mesh.get_point_cloud()
    random.shuffle(points)
    points = points[:1000]

    print(f"Number of points: {len(points)}")
    points_all = [
        point[0] for point in points
    ]  # Extracting (x, y, z) coordinates from the point cloud [(x, y, z), (r, g, b)]

    path_finder = PathFinder(points_all, 0.1)
    paths = path_finder.find_paths()

    print(
        f"Number of paths (length > 1): {len([path for path in paths if len(path) > 1])}"
    )
    print(
        f"Number of paths (length <= 1): {len([path for path in paths if len(path) <= 1])}"
    )

    path_finder.plot_paths(paths)


if __name__ == "__main__":
    main()
