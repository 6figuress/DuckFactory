"""Provides functions to sample points from a mesh and cluster them by color."""

import numpy as np
import trimesh
from trimesh.sample import sample_surface
from sklearn.cluster import DBSCAN
import pyray as ray

type Color = tuple[int, int, int]
type Point = tuple[float, float, float]


def sample_mesh_points(
    mesh: trimesh.Trimesh,
    base_color: Color,
    colors: list[Color],
    n_samples: int = 500_000,
) -> dict[Color, list[Point]]:
    """
    Samples points from the surface of a mesh and assigns them to the closest color in the palette.

    Args:
        mesh: The mesh to sample points from.
        base_color: The base color of the mesh. Points with this color will be ignored.
        colors: A list of colors in the palette.
        n_samples: The number of points to sample. The function will return fewer points if the projected points don't cover the whole mesh.

    Returns:
        A dictionary mapping colors to lists of points.
    """
    sampled_surface_points = sample_surface(mesh, n_samples, sample_color=True)

    points_count = len(sampled_surface_points[0])
    all_points = sampled_surface_points[0]
    all_colors = sampled_surface_points[2]

    # Remap colors to palette colors
    for i in range(points_count):
        color = all_colors[i]
        distances = np.linalg.norm(np.array(colors) - color, axis=1)
        closest_color = colors[np.argmin(distances)]
        all_colors[i] = closest_color

    # Convert to tuples
    all_points = [tuple(point) for point in all_points]
    all_colors = [tuple(int(c) for c in color) for color in all_colors]

    colors_unique = list(set(all_colors))

    # Group points by color
    points_by_color = {color: [] for color in colors_unique}
    for i in range(points_count):
        color = all_colors[i]
        point = all_points[i]
        points_by_color[color].append(point)

    # Drop the base color
    points_by_color.pop(base_color, None)

    return points_by_color


def cluster_points(
    points_by_color: dict[Color, list[Point]],
    eps: float = 0.0025 / 2,
    min_samples: int = 10,
) -> list[tuple[list[Point], Color]]:
    """
    Clusters points that are close to each other and have the same color.

    Args:
        points_by_color: A dictionary mapping colors to points.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. See DBSCAN documentation for more details.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point. See DBSCAN documentation for more details.

    Returns:
        A list of clusters, each containing a list of points and a color.
    """
    clusters_flat = []

    for color, points in points_by_color.items():
        if not points:
            continue

        points = np.array(points)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(points)
        labels = clustering.labels_

        color_clusters = {}
        for i, label in enumerate(labels):
            if label not in color_clusters:
                color_clusters[label] = []
            color_clusters[label].append(points[i].tolist())

        for cluster_points in color_clusters.values():
            clusters_flat.append((cluster_points, color))

    return clusters_flat


if __name__ == "__main__":
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
    for points, color in clusters_flat:
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

    selected_cluster = -1
    while not ray.window_should_close():
        if ray.is_key_released(ray.KeyboardKey.KEY_SPACE):
            selected_cluster += 1
            selected_cluster %= len(clusters_flat) - 1

        if ray.is_key_released(ray.KeyboardKey.KEY_BACKSPACE):
            selected_cluster = -1

        ray.begin_drawing()
        ray.clear_background(ray.DARKBLUE)

        ray.draw_text(f"Drawing {tot_points} points", 10, 10, 20, ray.RAYWHITE)

        ray.begin_mode_3d(cam)

        ray.draw_grid(10, 0.01)
        ray.update_camera(cam, ray.CameraMode.CAMERA_ORBITAL)
        ray.draw_model(model, (0, 0, 0), 1.0, ray.YELLOW)

        pen_width = 0.0018 / 2

        if selected_cluster == -1:
            for i in range(len(all_points)):
                point = all_points[i]
                color = all_colors[i]
                ray.draw_sphere(point, pen_width, color)
        else:
            cpoints, color = clusters_flat[selected_cluster]
            for point in cpoints:
                ray.draw_sphere(tuple(point), 0.00075, color)

        ray.end_mode_3d()
        ray.end_drawing()

    ray.close_window()
