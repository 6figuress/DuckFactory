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


def test_cluster_cube_faces():
    points = [
        # Face 1
        SampledPoint(coordinates=[1, 0.05, 0.05], normal=[1, 0, 0], color=(0, 0, 0, 0)),
        SampledPoint(coordinates=[1, 0.12, 0.08], normal=[1, 0, 0], color=(0, 0, 0, 0)),
        SampledPoint(coordinates=[1, 0.14, 0.13], normal=[1, 0, 0], color=(0, 0, 0, 0)),
        # Face 2
        SampledPoint(coordinates=[0.05, 1, 0.05], normal=[0, 1, 0], color=(0, 0, 0, 0)),
        SampledPoint(coordinates=[0.12, 1, 0.08], normal=[0, 1, 0], color=(0, 0, 0, 0)),
        SampledPoint(coordinates=[0.14, 1, 0.13], normal=[0, 1, 0], color=(0, 0, 0, 0)),
        # Face 3
        SampledPoint(coordinates=[0.05, 0.05, 1], normal=[0, 0, 1], color=(0, 0, 0, 0)),
        SampledPoint(coordinates=[0.12, 0.08, 1], normal=[0, 0, 1], color=(0, 0, 0, 0)),
        SampledPoint(coordinates=[0.14, 0.13, 1], normal=[0, 0, 1], color=(0, 0, 0, 0)),
        # Noise point on face 3
        SampledPoint(coordinates=[0.9, 0.9, 1], normal=[0, 0, 1], color=(0, 0, 0, 0)),
        # Noise point on other face
        SampledPoint(coordinates=[0.9, -1, 0.5], normal=[0, -1, 0], color=(0, 0, 0, 0)),
    ]

    clusters = cluster_points(points, distance_eps=0.3, min_samples=2)

    # Count non-noise clusters
    valid_clusters = [cluster for cluster in clusters if not cluster[2]]
    assert len(valid_clusters) == 3, "Expected 3 valid face clusters"

    # Count noise clusters
    noise_clusters = [cluster for cluster in clusters if cluster[2]]
    assert len(noise_clusters) == 2, "Expected 2 noise clusters"

    # Check cluster sizes (number of points per face)
    cluster_sizes = sorted([len(cluster[0]) for cluster in valid_clusters])
    assert all(size == 3 for size in cluster_sizes), (
        "Expected clusters with 3 points each"
    )
