import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from duck_factory.mesh_to_paths import norm_to_quat

def _assert_quat_normalized(quat):
    """Assert that a quaternion is normalized."""
    assert np.equal(np.linalg.norm(quat), 1.0), f"Quaternion is not normalized: {quat}"


def _assert_quat_valid(quat):
    """Check that no component of the quat is +-1."""
    assert np.not_equal(quat, 1.0).all(), f"Quaternion component is 1.0: {quat}"
    assert np.not_equal(quat, -1.0).all(), f"Quaternion component is -1.0: {quat}"


def _assert_vecs_close(vec1, vec2, tol=5.0):
    """Assert that two angles are close within a tolerance."""
    v1_norm = np.array(vec1) / np.linalg.norm(vec1)
    v2_norm = np.array(vec2) / np.linalg.norm(vec2)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle = np.degrees(angle_rad)

    assert angle < tol, f"Angle between vectors is {angle} degrees, expected < {tol}"


def test_norm_to_quat_basic():
    # Test with a simple normal vector
    normal = (1.0, 0.0, 0.0)
    quat = norm_to_quat(normal)

    # _assert_quat_normalized(quat)
    _assert_quat_valid(quat)

    # Apply rotation to z-axis and check if it aligns with -normal
    r = Rotation.from_quat(quat)
    rotated = r.apply((0, 0, 1))
    expected = (-1.0, 0.0, 0.0)

    _assert_vecs_close(rotated, expected)


def test_norm_to_quat_z_up_norm():
    # Test for edge case where normal is parallel to z-axis
    normal = (0.0, 0.0, 1.0)
    quat = norm_to_quat(normal)

    # _assert_quat_normalized(quat)
    _assert_quat_valid(quat)

    # Apply rotation to z-axis and check if it aligns with -normal
    r = Rotation.from_quat(quat)
    rotated = r.apply((0, 0, 1))
    expected = (0.0, 0.0, -1.0)

    _assert_vecs_close(rotated, expected)


def test_norm_to_quat_z_down_norm():
    # Test for edge case where normal is parallel to minus z-axis
    normal = (0.0, 0.0, -1.0)
    quat = norm_to_quat(normal)

    # _assert_quat_normalized(quat)
    _assert_quat_valid(quat)

    # Apply rotation to z-axis and check if it aligns with -normal
    r = Rotation.from_quat(quat)
    rotated = r.apply((0, 0, 1))
    expected = (0.0, 0.0, 1.0)

    _assert_vecs_close(rotated, expected)


def test_norm_to_quat_diagonal():
    # Test with a diagonal normal vector
    normal = (1.0, 1.0, 1.0)
    norm_normal = normal / np.linalg.norm(normal)
    quat = norm_to_quat(norm_normal)

    # _assert_quat_normalized(quat)
    _assert_quat_valid(quat)
    
    # Check if quaternion is normalized
    assert np.isclose(np.linalg.norm(quat), 1.0)
    
    # Apply rotation to z-axis and check if it aligns with -normal
    r = Rotation.from_quat(quat)
    rotated = r.apply((0, 0, 1))
    expected = -norm_normal

    _assert_vecs_close(rotated, expected)


def test_norm_to_quat_arbitrary():
    # Test with arbitrary normal vectors
    test_vectors = [
        (0.5, -0.3, 0.8),
        (-0.7, 0.2, 0.4),
        (0.1, 0.9, -0.2),
    ]
    
    for normal in test_vectors:
        norm_normal = normal / np.linalg.norm(normal)
        quat = norm_to_quat(norm_normal)

        # _assert_quat_normalized(quat)
        _assert_quat_valid(quat)
        
        # Check if quaternion is normalized
        assert np.isclose(np.linalg.norm(quat), 1.0)
        
        # Apply rotation to z-axis and check if it aligns with -normal
        r = Rotation.from_quat(quat)
        rotated = r.apply((0, 0, 1))
        expected = -norm_normal

        _assert_vecs_close(rotated, expected)


def test_norm_to_quat_sphere():
    # Test with normal vectors systematically distributed around a sphere
    n_points = 64 
    
    for theta in np.linspace(0, np.pi, n_points):  
        for phi in np.linspace(0, 2*np.pi, n_points):
            # Generate point on sphere using spherical coordinates
            normal = (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            )
            
            # Normalize 
            normal = tuple(np.array(normal) / np.linalg.norm(normal))
            
            # Get quaternion from normal
            quat = norm_to_quat(normal)

            # _assert_quat_normalized(quat)
            _assert_quat_valid(quat)
            
            # Check there are no NaN values
            assert not np.isnan(quat).any()
            
            # Apply rotation to z-axis and check if it aligns with -normal
            r = Rotation.from_quat(quat)
            rotated = r.apply((0, 0, 1))
            expected = tuple(-np.array(normal))

            _assert_vecs_close(rotated, expected)
