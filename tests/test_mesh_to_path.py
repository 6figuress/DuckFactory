import pytest
import numpy as np
import trimesh
from duck_factory.mesh_to_paths import mesh_to_paths, norm_to_quat


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


def test_norm_to_quat():
    normal = (0, 0, 1)
    quat = norm_to_quat(normal)

    assert isinstance(quat, np.ndarray)
    assert quat.shape == (4,)
    assert np.isclose(np.linalg.norm(quat), 1), "Quaternion should be normalized"

    normal = (1, 0, 0)
    quat = norm_to_quat(normal)
    assert isinstance(quat, np.ndarray)
    assert quat.shape == (4,)
