import pytest
import numpy as np
import trimesh
from duck_factory.point_sampling import sample_mesh_points, cluster_points, SampledPoint


# Helper function to create a simple mesh for testing
def create_test_mesh():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # Bottom face
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],  # Top face
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [2, 3, 0],
            [4, 5, 6],
            [6, 7, 4],
            [0, 1, 5],
            [5, 4, 0],
            [2, 3, 7],
            [7, 6, 2],
            [0, 3, 7],
            [7, 4, 0],
            [1, 2, 6],
            [6, 5, 1],
        ]
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def test_mesh():
    return create_test_mesh()


@pytest.fixture
def sample_colors():
    return [
        (255, 255, 0, 255),  # Yellow (Base Color)
        (0, 0, 0, 255),  # Black
        (0, 0, 255, 255),  # Blue
        (0, 255, 0, 255),  # Green
        (255, 0, 0, 255),  # Red
    ]


@pytest.fixture
def base_color():
    return (255, 255, 0, 255)  # Yellow (Base Color)


def test_sample_mesh_points(test_mesh, base_color, sample_colors):
    sampled_points = sample_mesh_points(
        test_mesh, base_color, sample_colors, n_samples=100
    )

    assert isinstance(sampled_points, list)
    assert all(isinstance(p, SampledPoint) for p in sampled_points)
    assert all(len(p.coordinates) == 3 for p in sampled_points)
    assert all(len(p.normal) == 3 for p in sampled_points)

    # Ensure base color points are removed
    assert not any(p.color == base_color for p in sampled_points)


def test_cluster_points():
    points = [
        SampledPoint((0.1, 0.1, 0.1), (255, 0, 0, 255), (0, 0, 1)),
        SampledPoint((0.1, 0.11, 0.1), (255, 0, 0, 255), (0, 0, 1)),
        SampledPoint((0.2, 0.2, 0.2), (0, 255, 0, 255), (0, 0, 1)),
        SampledPoint((0.21, 0.2, 0.2), (0, 255, 0, 255), (0, 0, 1)),
        SampledPoint((1.0, 1.0, 1.0), (0, 0, 255, 255), (0, 0, 1)),  # Noise point
    ]
    clusters = cluster_points(points, eps=0.02, min_samples=2)

    assert isinstance(clusters, list)
    assert all(isinstance(cluster, tuple) and len(cluster) == 3 for cluster in clusters)

    # Ensure at least one noise cluster
    assert any(cluster[2] for cluster in clusters)
