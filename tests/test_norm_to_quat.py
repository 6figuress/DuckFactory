import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from duck_factory.mesh_to_paths import norm_to_quat


def test_norm_to_quat_basic():
    # Test with a simple normal vector
    normal = (1.0, 0.0, 0.0)
    quat = norm_to_quat(normal)
    
    # Check if quaternion is normalized
    assert np.isclose(np.linalg.norm(quat), 1.0)
    
    # Apply rotation to z-axis and check if it aligns with -normal
    r = Rotation.from_quat(quat)
    rotated = r.apply((0, 0, 1))
    assert np.allclose(rotated, (-1.0, 0.0, 0.0), atol=1e-3)


def test_norm_to_quat_z_up_norm():
    # Test for edge case where normal is parallel to z-axis 
    normal = (0.0, 0.0, 1.0)
    quat = norm_to_quat(normal)
    
    # Expected quaternion for z-axis pointing down
    assert np.allclose(quat, (0, 9.9e-1, 0, 5.06e-4), atol=1e-2)
    
    # Apply rotation to z-axis and check if it aligns with -normal
    r = Rotation.from_quat(quat)
    rotated = r.apply((0, 0, 1))
    assert np.allclose(rotated, (0.0, 0.0, -1.0), atol=1e-2)


def test_norm_to_quat_z_down_norm():
    # Test for edge case where normal is parallel to minus z-axis
    normal = (0.0, 0.0, -1.0)
    quat = norm_to_quat(normal)
    
    # Should be close to identity quaternion (z pointing up)
    assert np.allclose(quat, (0, 5.06e-4, 0, 9.9e-1), atol=1e-2)
    
    # Apply rotation to z-axis and check if it aligns with -normal
    r = Rotation.from_quat(quat)
    rotated = r.apply((0, 0, 1))
    assert np.allclose(rotated, (0.0, 0.0, 1.0), atol=1e-2)


def test_norm_to_quat_diagonal():
    # Test with a diagonal normal vector
    normal = (1.0, 1.0, 1.0)
    norm_normal = normal / np.linalg.norm(normal)
    quat = norm_to_quat(norm_normal)
    
    # Check if quaternion is normalized
    assert np.isclose(np.linalg.norm(quat), 1.0)
    
    # Apply rotation to z-axis and check if it aligns with -normal
    r = Rotation.from_quat(quat)
    rotated = r.apply((0, 0, 1))
    expected = -norm_normal
    assert np.allclose(rotated, expected, atol=1e-3)


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
        
        # Check if quaternion is normalized
        assert np.isclose(np.linalg.norm(quat), 1.0)
        
        # Apply rotation to z-axis and check if it aligns with -normal
        r = Rotation.from_quat(quat)
        rotated = r.apply((0, 0, 1))
        expected = -norm_normal
        assert np.allclose(rotated, expected, atol=1e-3)


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

            # Check if quaternion is normalized
            assert np.isclose(np.linalg.norm(quat), 1.0)
            
            # Check there are no NaN values
            assert not np.isnan(quat).any()
            
            # Apply rotation to z-axis and check if it aligns with -normal
            r = Rotation.from_quat(quat)
            rotated = r.apply((0, 0, 1))
            expected = tuple(-np.array(normal))
            assert np.allclose(rotated, expected, atol=1e-2)
