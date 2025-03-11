import pytest
import numpy as np
from duck_factory.reachable_points import PathAnalyzer


def test_is_point_inside_pen():
    analyzer = PathAnalyzer(
        tube_length=50.0, diameter=0.02, cone_height=0.01, step_angle=10, num_vectors=24
    )
    p0 = (0, 0, 0)
    normal = (0, 0, 1)

    inside_point = (0, 0, 0.005)
    assert analyzer.is_point_inside_pen(inside_point, p0, normal) == True

    inside_point = (0, 0.0019, 0.005)
    assert analyzer.is_point_inside_pen(inside_point, p0, normal) == True

    outside_point = (3, 0, 0.3)
    assert analyzer.is_point_inside_pen(outside_point, p0, normal) == False

    outside_point = (0, 0.2, 0.005)
    assert analyzer.is_point_inside_pen(outside_point, p0, normal) == False


def test_is_reachable():
    analyzer = PathAnalyzer(
        tube_length=50.0, diameter=0.02, cone_height=0.01, step_angle=10, num_vectors=24
    )
    point = (1, 1, 1)
    normal = (0, 0, -1)
    model_points = [(1, 1, 2), (2, 2, 2)]

    assert analyzer.is_reachable(point, normal, model_points) == True

    model_points.append((1, 1, 0.995))
    assert analyzer.is_reachable(point, normal, model_points) == False


def test_generate_cone_vectors():
    analyzer = PathAnalyzer(
        tube_length=50.0, diameter=0.02, cone_height=0.01, step_angle=10, num_vectors=24
    )
    normal = (0, 0, 1)
    angle = 30
    vectors = analyzer.generate_cone_vectors(normal, angle)

    assert len(vectors) == 24
    for vector in vectors:
        assert np.isclose(np.linalg.norm(vector), 1.0)


def test_find_valid_orientation():
    analyzer = PathAnalyzer(
        tube_length=50.0, diameter=0.02, cone_height=0.01, step_angle=10, num_vectors=24
    )
    point = (1, 1, 1)
    normal = (0, 0, 1)
    model_points = [(1, 1, 1.005)]

    valid, new_normal = analyzer.find_valid_orientation(point, normal, model_points)
    assert valid == True
    assert not np.array_equal(new_normal, normal)


def test_filter_reachable_points():
    analyzer = PathAnalyzer(
        tube_length=50.0, diameter=0.02, cone_height=0.01, step_angle=10, num_vectors=24
    )
    data_points = [((1, 1, 1), (0, 0, 1)), ((2, 2, 2), (0, 0, 1))]
    model_points = [(1, 1, 1.005)]

    updated, unreachable = analyzer.filter_reachable_points(data_points, model_points)

    assert len(updated) > 0
    assert len(unreachable) >= 0
