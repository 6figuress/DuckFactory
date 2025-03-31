"""Provides functions to sample points from a mesh and cluster them by color."""

import numpy as np
import trimesh
from trimesh.sample import sample_surface
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
from typing import List, Tuple
import pyray as ray
from math import radians, cos
from PIL import Image

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
    nopaint_mask: Image = None,
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
        nopaint_mask: An optional mask image to restrict sampling to certain areas of the mesh. White pixels mark "nopaint" areas.

    Returns:
        A list of SampledPoint objects, excluding points with the base color.
    """
    # If a nopaint mask is provided, replace masked areas with the base color in the texture to avoid sampling them
    if nopaint_mask is not None:
        texture = mesh.visual.material.image.convert("RGB")
        base_color_rgb = np.array(base_color[:3], dtype=np.uint8)

        if nopaint_mask.size != texture.size:
            nopaint_mask = nopaint_mask.resize(texture.size, Image.BICUBIC)

        mask_bw = nopaint_mask.convert("L")
        mask_array = np.array(mask_bw)
        white_mask = mask_array == 255

        texture_array = np.array(texture)
        texture_array[white_mask] = base_color_rgb

        mesh.visual.material.image = Image.fromarray(texture_array)

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
    min_samples: int = 5,
) -> List[Tuple[List[SampledPoint], Color, bool]]:
    """
    Clusters points that are close to each other and have similar normal vectors.

    Args:
        points: List of points to cluster
        distance_eps: Maximum distance between two samples for neighborhood
        min_samples: Minimum number of samples in a neighborhood

    Returns:
        List of clusters, each containing a list of points, their color, and a flag indicating if the cluster is noise (i.e. points not in any cluster)
    """
    normal_groups = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
        "front": [],
        "back": [],
    }

    # Threshold angle to check between the normal and the corresponding axis
    # A higher cosine value means a smaller angle with the axis
    # (cosine decreases as angle increases)
    cos_threshold = cos(radians(65))

    for point in points:
        nx, ny, nz = point.normal
        x, y, z = point.coordinates

        # Normalize normal vector
        normal_len = (nx**2 + ny**2 + nz**2) ** 0.5
        nx, ny, nz = nx / normal_len, ny / normal_len, nz / normal_len

        if nz > cos_threshold:
            normal_groups["right"].append(point)
        elif nz < -cos_threshold:
            normal_groups["left"].append(point)
        elif nx > cos_threshold:
            normal_groups["front"].append(point)
        elif nx < -cos_threshold:
            normal_groups["back"].append(point)
        elif ny > cos_threshold:
            normal_groups["top"].append(point)
        elif ny < -cos_threshold:
            normal_groups["bottom"].append(point)
        else:
            # Normal doesn't clearly align with any axis,
            # prioritize left/right sides based on point position
            normal_groups["right" if z > 0 else "left"].append(point)

    clusters_flat = []
    for _, normal_points in normal_groups.items():
        # Group points by color within this normal group
        color_groups = {}
        for point in normal_points:
            if point.color not in color_groups:
                color_groups[point.color] = []
            color_groups[point.color].append(point)

        # Process each color group within this normal group
        for color, color_points in color_groups.items():
            # Extract point coordinates
            point_data = np.array([p.coordinates for p in color_points])

            if len(point_data) < min_samples:
                # Not enough points for clustering, treat as one cluster
                if color_points:
                    clusters_flat.append((color_points, color, False))
                continue

            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=distance_eps,
                min_samples=min_samples,
                n_jobs=-1,
            ).fit(point_data)

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

    mesh = trimesh.load_mesh("spiderman.obj")
    mask = Image.open("nopaint_mask.png").convert("L")

    points_by_color = sample_mesh_points(
        mesh, BASE_COLOR, COLORS, n_samples=10_000, nopaint_mask=mask
    )
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
