"""Provides functions to sample points from a mesh and cluster them by color."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pyray as ray
import trimesh
from PIL import Image
from sklearn.cluster import DBSCAN
from trimesh.sample import sample_surface

Color = Tuple[int, int, int]
Point = Tuple[float, float, float]
Normal = Tuple[float, float, float]


def get_texture(mesh: trimesh.Trimesh):
    """Get texture from mesh, handling both PBR and regular materials."""
    if hasattr(mesh.visual.material, "baseColorTexture"):
        return mesh.visual.material.baseColorTexture
    elif hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "image"):
        return mesh.visual.material.image
    else:
        return Image.new("RGB", (256, 256), color=(255, 255, 0))


@dataclass
class SampledPoint:
    """Represents a point sampled from a mesh surface with additional metadata."""

    coordinates: Point
    color: Color
    normal: Normal


def sample_mesh_points(
    mesh: trimesh.Trimesh,
    base_color: Color,
    colors: List[Color],
    n_samples: int = 500_000,
) -> List[SampledPoint]:
    """Samples points from the surface of a mesh and assigns them to the closest color."""
    # Sample points and faces (without color initially)
    points, face_indices = sample_surface(mesh, n_samples, sample_color=False)

    # Try to get color from texture
    if (
        hasattr(mesh.visual.material, "baseColorTexture")
        and mesh.visual.material.baseColorTexture is not None
    ):
        print("Using baseColorTexture")
        texture = mesh.visual.material.baseColorTexture
        # Convert texture to numpy array if it's a PIL Image
        texture_array = np.array(texture)

        # Get UV coordinates
        if not hasattr(mesh.visual, "uv"):
            print("No UV coordinates found, using default color")
            return []

        uv_coords = mesh.visual.uv

        processed_points = []
        for i in range(len(points)):
            # Get face vertices
            face = mesh.faces[face_indices[i]]
            # Get UV coordinates for face vertices
            face_uvs = uv_coords[face]

            # Use barycentric coordinates to interpolate UV coordinates
            # For simplicity, just use the first UV coordinate of the face
            uv = face_uvs[0]

            # Convert UV to pixel coordinates
            x = int(uv[0] * (texture_array.shape[1] - 1))
            y = int((1 - uv[1]) * (texture_array.shape[0] - 1))  # Flip Y coordinate

            # Sample color from texture
            pixel_color = texture_array[y, x]
            if len(pixel_color) > 3:
                pixel_color = pixel_color[:3]  # Take only RGB components

            # Find closest color in palette
            distances = [np.linalg.norm(np.array(c[:3]) - pixel_color) for c in colors]
            closest_color = colors[np.argmin(distances)]

            # Don't skip base color points anymore
            point_coords = tuple(points[i])
            point_normal = tuple(mesh.face_normals[face_indices[i]])

            processed_points.append(
                SampledPoint(
                    coordinates=point_coords,
                    color=closest_color,
                    normal=point_normal,
                )
            )

            if i % 1000 == 0:
                print(f"Processed {i}/{len(points)} points")
    else:
        print("No texture found, using default color")
        processed_points = []
        for i in range(len(points)):
            point_coords = tuple(points[i])
            point_normal = tuple(mesh.face_normals[face_indices[i]])
            processed_points.append(
                SampledPoint(
                    coordinates=point_coords,
                    color=base_color,
                    normal=point_normal,
                )
            )

    print(f"Processed {len(processed_points)} points")
    # Print color distribution
    color_counts = {}
    for point in processed_points:
        color_counts[point.color] = color_counts.get(point.color, 0) + 1
    print("Color distribution:")
    for color, count in color_counts.items():
        print(f"Color {color}: {count} points")

    return processed_points


def cluster_points(
    points: List[SampledPoint],
    eps: float = 0.005,  # Reduced from 0.0025/2
    min_samples: int = 5,  # Reduced from 10
) -> List[Tuple[List[SampledPoint], Color, bool]]:
    """Clusters points that are close to each other."""
    print(f"Starting clustering with {len(points)} points")

    color_groups = {}
    for point in points:
        if point.color not in color_groups:
            color_groups[point.color] = []
        color_groups[point.color].append(point)

    clusters_flat = []
    for color, color_points in color_groups.items():
        print(f"Clustering {len(color_points)} points of color {color}")
        point_coords = np.array([p.coordinates for p in color_points])

        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(
            point_coords
        )
        labels = clustering.labels_

        # Count number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(
            f"Found {n_clusters} clusters and {n_noise} noise points for color {color}"
        )

        color_clusters = {}
        for i, label in enumerate(labels):
            if label not in color_clusters:
                color_clusters[label] = []
            color_clusters[label].append(color_points[i])

        for label, cluster_points in color_clusters.items():
            if cluster_points:  # Include noise points (-1 label)
                clusters_flat.append((cluster_points, color, label == -1))

    print(f"Total clusters generated: {len(clusters_flat)}")
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
