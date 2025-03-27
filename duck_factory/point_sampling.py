"""Provides functions to sample points from a mesh and cluster them by color."""

import numpy as np
import trimesh
from trimesh.sample import sample_surface
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
from typing import List, Tuple
import pyray as ray
from math import radians, cos

Color = Tuple[int, int, int]
Point = Tuple[float, float, float]
Normal = Tuple[float, float, float]


@dataclass
class SampledPoint:
    """
    Represents a point sampled from a mesh surface with additional metadata.

    Attributes:
        coordinates: 3D coordinates of the point
        color: RGB color of the point
        normal: Surface normal at the point
    """

    coordinates: Point
    color: Color
    normal: Normal


def sample_mesh_points(
    mesh: trimesh.Trimesh,
    base_color: Color,
    colors: List[Color],
    n_samples: int = 500_000,
) -> List[SampledPoint]:
    """
    Samples points from the surface of a mesh and assigns them to the closest color in the palette.

    Notes:  Points with the base color are ignored.
            As some points may be discarded after sampling, the number of sampled points may be less than n_samples.

    Args:
        mesh: The mesh to sample points from.
        base_color: The base color of the mesh. Points with this color will be ignored.
        colors: A list of colors in the palette.
        n_samples: The number of points to sample.

    Returns:
        A list of SampledPoint objects, excluding points with the base color.
    """
    # Sample surface points
    sampled_surface_points = sample_surface(mesh, n_samples, sample_color=True)

    all_points = sampled_surface_points[0]
    all_faces = sampled_surface_points[1]
    all_colors = sampled_surface_points[2]

    processed_points = []
    for i in range(len(all_points)):
        # Find closest color in palette
        color = all_colors[i]
        distances = np.linalg.norm(np.array(colors) - color, axis=1)
        closest_color = colors[np.argmin(distances)]

        # Skip base color points
        if closest_color == base_color:
            continue

        # Create SampledPoint
        point_coords = tuple(all_points[i])
        point_normal = tuple(mesh.face_normals[all_faces[i]])
        processed_points.append(
            SampledPoint(
                coordinates=point_coords,
                color=closest_color,
                normal=point_normal,
            )
        )

    return processed_points


def cluster_points(
    points: List[SampledPoint],
    distance_eps: float = 0.0025 / 2,
    min_samples: int = 10,
    max_normal_angle: float = 45.0,
) -> List[Tuple[List[SampledPoint], Color, bool]]:
    """
    Clusters points that are close to each other and have similar normal vectors.

    Args:
        points: List of points to cluster
        distance_eps: Maximum distance between two samples for neighborhood
        min_samples: Minimum number of samples in a neighborhood
        max_normal_angle: Maximum angle (in degrees) between normals for points to be in the same cluster

    Returns:
        List of clusters, each containing a list of points, their color, and a flag indicating if the cluster is noise (i.e. points not in any cluster)
    """
    LARGE_DISTANCE = 1e6  # DNSCAN can't handle infinity, use 1000km instead

    # Group points by color
    color_groups = {}
    for point in points:
        if point.color not in color_groups:
            color_groups[point.color] = []
        color_groups[point.color].append(point)

    # Convert max angle to radians and calculate the cosine threshold
    # (Using cosine comparison is more efficient than calculating angles)
    cos_threshold = cos(radians(max_normal_angle))

    def custom_metric(a: tuple[*Point, *Normal], b: tuple[*Point, *Normal]) -> float:
        """
        Custom distance metric that considers both spatial distance and normal angle.

        Returns:
            Distance between the two points if they are within the
            allowed spatial distance and normal angle, otherwise a large value
        """
        p1, p2 = a[:3], b[:3]
        n1, n2 = a[3:6], b[3:6]

        # Calculate spatial distance
        spatial_dist = np.linalg.norm(p1 - p2)

        # Calculate dot product (cosine of angle between normals, as they are normalized)
        n1_norm = n1 / np.linalg.norm(n1)
        n2_norm = n2 / np.linalg.norm(n2)
        cos_angle = np.dot(n1_norm, n2_norm)

        # If normals are within the allowed angle and spatial distance is within eps
        if cos_angle >= cos_threshold and spatial_dist <= distance_eps:
            return spatial_dist
        else:
            return LARGE_DISTANCE

    clusters_flat = []
    for color, color_points in color_groups.items():
        # Extract point coordinates and normals
        point_data = np.array(
            [list(p.coordinates) + list(p.normal) for p in color_points]
        )

        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=distance_eps,
            min_samples=min_samples,
            n_jobs=-1,
            metric="precomputed",
        )

        # Calculate distance matrix using custom metric
        n_points = len(point_data)
        distance_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = custom_metric(point_data[i], point_data[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Make sure there are no NaN or infinite values
        distance_matrix = np.clip(distance_matrix, 0, LARGE_DISTANCE)

        clustering.fit(distance_matrix)

        labels = clustering.labels_

        # Group points by cluster label
        color_clusters = {}
        for i, label in enumerate(labels):
            if label not in color_clusters:
                color_clusters[label] = []
            color_clusters[label].append(color_points[i])

        # Flatten clusters
        for label, cluster_points in color_clusters.items():
            if cluster_points:
                clusters_flat.append((cluster_points, color, label == -1))

    return clusters_flat


if __name__ == "__main__":  # pragma: no cover
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

    points_by_color = sample_mesh_points(mesh, BASE_COLOR, COLORS)
    clusters_flat = cluster_points(points_by_color)

    all_points = []
    all_colors = []
    for points, color, _ in clusters_flat:
        all_points.extend(points)
        all_colors.extend([color] * len(points))

    tot_points = len(all_points)

    ray.init_window(1920, 1080, "Duck Factory")
    cam = ray.Camera3D()
    cam.up = (0, 1, 0)
    cam.position = (0, 0.2, 0.2)
    cam.target = (0, 0.05, 0)
    cam.fovy = 35.0
    cam.projection = ray.CameraProjection.CAMERA_PERSPECTIVE

    model = ray.load_model("DuckAllYellow.obj")

    while not ray.window_should_close():
        ray.begin_drawing()
        ray.clear_background(ray.DARKBLUE)

        ray.draw_text(f"Drawing {tot_points} points", 10, 10, 20, ray.RAYWHITE)

        ray.begin_mode_3d(cam)

        ray.draw_grid(10, 0.01)
        ray.update_camera(cam, ray.CameraMode.CAMERA_ORBITAL)
        ray.draw_model(model, (0, 0, 0), 1.0, ray.YELLOW)

        pen_width = 0.0018 / 2

        for i in range(len(all_points)):
            point = all_points[i]
            color = all_colors[i]
            ray.draw_sphere(point.coordinates, pen_width, color)

        ray.end_mode_3d()
        ray.end_drawing()

    ray.close_window()
