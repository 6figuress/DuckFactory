import pytest
import numpy as np
from trimesh import Trimesh
from duck_factory.path_bounder import PathBounder
from duck_factory.reachable_points import PathAnalyzer


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
    return Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def test_mesh():
    return create_test_mesh()


@pytest.fixture
def path_bounder(test_mesh):
    analyzer = PathAnalyzer(
        tube_length=5e1, diameter=2e-2, cone_height=1e-2, step_angle=10, num_vectors=24
    )
    return PathBounder(
        mesh=test_mesh, analyzer=analyzer, model_points=np.array([[0, 0, 0]])
    )


def test_ray_plane_intersection(path_bounder):
    origin = (0.5, 0.5, -1)
    direction = (0, 0, 1)
    point_on_plane = (0.5, 0.5, 0)
    plane_normal = (0, 0, 1)

    intersection = path_bounder.ray_plane_intersection(
        origin, direction, point_on_plane, plane_normal
    )

    assert intersection is not None, "Ray should intersect the plane"
    assert isinstance(intersection, np.ndarray), "Intersection should be a NumPy array"
    assert intersection.shape == (3,), "Intersection should be a 3D point"
    assert np.isclose(intersection[2], 0), "Intersection should be at z = 0"


def test_set_analyzer(path_bounder):
    new_analyzer = PathAnalyzer(
        tube_length=10, diameter=5e-2, cone_height=2e-2, step_angle=5, num_vectors=12
    )
    path_bounder.set_analyzer(new_analyzer)
    assert path_bounder.analyzer == new_analyzer


def test_set_model_points(path_bounder):
    new_model_points = [(0.5, 0.5, 0.5), (0.2, 0.3, 0.4)]
    path_bounder.set_model_points(new_model_points)
    assert path_bounder.model_points == new_model_points


def test_set_nz_threshold(path_bounder):
    new_threshold = 0.1
    path_bounder.set_nz_threshold(new_threshold)
    assert path_bounder.nz_threshold == new_threshold


def test_set_step_size(path_bounder):
    new_step_size = 0.02
    path_bounder.set_step_size(new_step_size)
    assert path_bounder.step_size == new_step_size


def test_get_normal_to_face(path_bounder):
    point1 = (0.5, 0.5, 0)
    point2 = (0.5, 0.5, 1)
    normal = path_bounder.get_normal_to_face(point1, point2)
    assert isinstance(normal, np.ndarray)
    assert normal.shape == (3,)
    assert np.isclose(np.linalg.norm(normal), 1)


def test_ray_box_intersection(path_bounder):
    origin = (0.5, 0.5, -1)
    direction = (0, 0, 1)
    intersection = path_bounder.ray_box_intersection(origin, direction)
    assert isinstance(intersection, np.ndarray)
    assert len(intersection) == 3


def test_generate_path_on_box(path_bounder):
    start = (0.1, 0.1, 0.1)
    end = (0.9, 0.9, 0.9)
    path = path_bounder.generate_path_on_box(start, end)
    assert isinstance(path, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in path)


def test_compute_path_with_orientation(path_bounder):
    start = ((0.1, 0.1, 0.1), (1, 0, 0))
    end = ((0.9, 0.9, 0.9), (0, 1, 0))
    path = path_bounder.compute_path_with_orientation(start, end)
    assert isinstance(path, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in path)


def test_merge_path(path_bounder):
    path1 = [((0.1, 0.1, 0.1), (1, 0, 0)), ((0.2, 0.2, 0.2), (0, 1, 0))]
    path2 = [((0.8, 0.8, 0.8), (0, 0, 1)), ((0.9, 0.9, 0.9), (-1, 0, 0))]
    merged_path = path_bounder.merge_path(path1, path2)
    assert isinstance(merged_path, list)
    assert len(merged_path) >= len(path1) + len(path2)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in merged_path)


def test_merge_all_path(path_bounder):
    paths = [
        [((0.1, 0.1, 0.1), (1, 0, 0)), ((0.2, 0.2, 0.2), (0, 1, 0))],
        [((0.4, 0.4, 0.4), (0, 0, 1)), ((0.5, 0.5, 0.5), (-1, 0, 0))],
        [((0.8, 0.8, 0.8), (0, -1, 0)), ((0.9, 0.9, 0.9), (1, 1, 1))],
    ]
    merged_path = path_bounder.merge_all_path(paths)
    assert isinstance(merged_path, list)
    assert len(merged_path) >= sum(len(p) for p in paths)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in merged_path)
