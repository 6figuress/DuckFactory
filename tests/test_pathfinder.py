import pytest
import numpy as np
import networkx as nx
from duck_factory.point_sampling import SampledPoint

from duck_factory.points_to_paths import PathFinder


@pytest.fixture
def sample_points():
    points = [
        (0.0, 0.0, 0.0),
        (0.05, 0.05, 0.05),
        (0.1, 0.1, 0.1),
        (0.3, 0.3, 0.3),
        (0.35, 0.35, 0.35),
    ]

    return [
        SampledPoint(
            coordinates=point,
            color=(255, 255, 255),
            normal=(0.0, 0.0, 1.0),
        )
        for point in points
    ]


@pytest.fixture
def pathfinder(sample_points):
    return PathFinder(sample_points, max_distance=0.2)


# Test initialization
def test_initialization(pathfinder, sample_points):
    assert len(pathfinder.points) == len(sample_points)
    assert isinstance(pathfinder.graph, nx.Graph)
    assert pathfinder.max_distance == 0.2


# Test graph creation
def test_graph_connections(pathfinder):
    graph = pathfinder.graph
    assert graph.number_of_nodes() == len(pathfinder.points)

    # Ensure edges exist only within max distance
    for edge in graph.edges:
        p1 = np.array(pathfinder.points[edge[0]].coordinates)
        p2 = np.array(pathfinder.points[edge[1]].coordinates)
        assert np.linalg.norm(p1 - p2) <= pathfinder.max_distance


# Test connected components
def test_find_connected_components(pathfinder):
    components = pathfinder.find_connected_components()
    assert isinstance(components, list)
    assert all(isinstance(component, list) for component in components)

    # The first three points should be connected due to max_distance=0.2
    assert len(components) == 2  # Two separate components expected


# Test path finding
def test_find_paths(pathfinder):
    paths = pathfinder.find_paths()

    # Ensure paths are lists of tuples
    assert isinstance(paths, list)
    assert all(isinstance(path, list) for path in paths)
    assert all(isinstance(point, SampledPoint) for path in paths for point in path)

    # Should be at least one path with multiple points
    assert any(len(path) > 1 for path in paths)
