import json
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from trimesh import Trimesh, load_mesh
from trimesh.sample import sample_surface

from duck_factory.dither_class import Dither
from duck_factory.path_bounder import PathBounder
from duck_factory.point_sampling import (
    Color,
    Point,
    cluster_points,
    sample_mesh_points,
)
from duck_factory.points_to_paths import PathFinder
from duck_factory.reachable_points import PathAnalyzer

Normal = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
PathPosition = tuple[*Point, *Quaternion]
Path = tuple[Color, list[PathPosition]]

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


def get_texture(mesh: Trimesh):
    """Get texture from mesh, handling both PBR and regular materials."""
    if hasattr(mesh.visual.material, "baseColorTexture"):
        return mesh.visual.material.baseColorTexture
    elif hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "image"):
        return mesh.visual.material.image
    else:
        return Image.new("RGB", (256, 256), color=BASE_COLOR[:3])


def mesh_to_paths(
    mesh: Trimesh,
    n_samples: int = 50_000,
    max_dist: float = 0.1,
    home_point: tuple[Point, Normal] = ((0, 0, 0.25), (0, 0, -1)),
) -> list[Path]:
    """
    Do the full conversion from a textured mesh to a list of IK-ready paths.
    """
    # If the mesh has no color information, try to add some
    if not hasattr(mesh.visual, "vertex_colors"):
        if hasattr(mesh.visual.material, "baseColorFactor"):
            color = mesh.visual.material.baseColorFactor
        else:
            color = BASE_COLOR
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))

    # Sample points from the mesh and cluster them by proximity
    sampled_points = sample_mesh_points(
        mesh, base_color=BASE_COLOR, colors=COLORS, n_samples=n_samples
    )

    path_analyzer = PathAnalyzer(
        tube_length=5e1,
        diameter=2e-2,
        cone_height=1e-2,
        step_angle=36,
        num_vectors=12,
    )

    mesh_points = sample_surface(mesh, n_samples, sample_color=False)[0]

    counter = 0
    valid_points = []
    for point in sampled_points:
        counter += 1

        # Convert from y-up, x-forward, z-right to z-up, x-forward, y-left
        point.coordinates = (
            point.coordinates[0],
            point.coordinates[2],
            -point.coordinates[1],
        )
        point.normal = (
            point.normal[0],
            point.normal[2],
            -point.normal[1],
        )

        # Find a valid orientation for the point
        valid, new_norm = path_analyzer.find_valid_orientation(
            point.coordinates, point.normal, mesh_points
        )

        if valid:
            point.normal = new_norm
        else:
            continue

        valid_points.append(point)

    # Cluster the points by proximity and color
    clusters = cluster_points(valid_points)

    # Compute the paths for each cluster, and store lists of paths by color
    color_paths = defaultdict(list)
    for points, color, is_noise in clusters:
        print(
            f"Processing cluster of {len(points)} points with color {color} (noise: {is_noise})"
        )
        if is_noise:
            # Create individual paths for noise points
            for point in points:
                color_paths[color].append([point])
        else:
            path_finder = PathFinder(points, max_dist)
            paths_positions = path_finder.find_paths()

            print(f"Found {len(paths_positions)} paths for cluster")
            for path in paths_positions:
                color_paths[color].append(path)

    # Merge the paths for each color by inserting paths between them
    rpaths = []
    for color, paths in color_paths.items():
        print(f"Processing {len(paths)} paths of color {color}")
        if not paths:
            continue

        bounder = PathBounder(mesh, path_analyzer, mesh_points)

        # Convert paths to position-normal format
        prepped_paths = [
            [(point.coordinates, point.normal) for point in path] for path in paths
        ]

        # Finish the path at the home point
        prepped_paths = prepped_paths + [[home_point]]

        # Merge the paths
        try:
            merged = bounder.merge_all_path(prepped_paths, restricted_face=[3, 8])
            # Convert normals to quaternions
            converted_path = [(pos, norm_to_quat(norm)) for pos, norm in merged]
            rpaths.append((color, converted_path))
        except Exception as e:
            print(f"Error merging paths for color {color}: {e}")
            # Try to salvage what we can by creating individual paths
            for path in prepped_paths:
                if path and path != [home_point]:
                    converted_path = [(pos, norm_to_quat(norm)) for pos, norm in path]
                    rpaths.append((color, converted_path))
    return rpaths


def norm_to_quat(normal: Normal) -> Quaternion:
    """Convert a normal vector to a quaternion."""
    normal = (-normal[0], -normal[1], -normal[2])

    if np.allclose(normal, [0, 0, 0]):
        raise ValueError("Cannot normalize a zero vector (normal is [0, 0, 0])")

    normal = normal / np.linalg.norm(normal)

    if np.allclose(normal, [0, 0, 1]):
        quat = (0, 5.06e-4, 0, 9.9e-1)
        return quat / np.linalg.norm(quat)
    elif np.allclose(normal, [0, 0, -1]):
        quat = (0, 9.9e-1, 0, 5.06e-4)
        return quat / np.linalg.norm(quat)

    rotation_axis = np.cross((0, 0, 1), normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    cos_angle = np.dot([0, 0, 1], normal)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    r = Rotation.from_rotvec(angle * rotation_axis)
    quat = r.as_quat()

    if np.isnan(quat).any():
        raise ValueError("Quaternion contains NaNs")

    close_to_pos_1 = np.isclose(quat, 1, atol=1e-3)
    close_to_min_1 = np.isclose(quat, -1, atol=1e-3)
    quat[close_to_pos_1] = 9.9e-1
    quat[close_to_min_1] = -9.9e-1

    quat = quat / np.linalg.norm(quat)

    return quat


if __name__ == "__main__":  # pragma: no cover
    mesh = load_mesh("duck.obj")
    print(f"Loaded mesh with {len(mesh.vertices)} vertices")
    print(f"Mesh has vertex colors: {hasattr(mesh.visual, 'vertex_colors')}")
    print(f"Mesh has material: {hasattr(mesh.visual, 'material')}")
    print(f"Mesh has UV coordinates: {hasattr(mesh.visual, 'uv')}")
    if hasattr(mesh.visual, "material"):
        print(f"Material type: {type(mesh.visual.material)}")
        if hasattr(mesh.visual.material, "baseColorFactor"):
            print(f"Base color factor: {mesh.visual.material.baseColorFactor}")
        if hasattr(mesh.visual.material, "baseColorTexture"):
            print(
                f"Has base color texture: {mesh.visual.material.baseColorTexture is not None}"
            )
            if mesh.visual.material.baseColorTexture is not None:
                texture = mesh.visual.material.baseColorTexture
                print(f"Texture type: {type(texture)}")
                if hasattr(texture, "size"):
                    print(f"Texture size: {texture.size}")
                # Save texture for inspection
                texture.save("texture.png")
                print("Saved texture to texture.png")

    paths = mesh_to_paths(mesh, n_samples=50000, max_dist=0.024)

    print("\nProcessing complete:")
    print(f"Number of paths: {len(paths)}")
    total_points = sum([len(path[1]) for path in paths])
    print(f"Number of points: {total_points}")

    # Print path details
    for i, (color, path) in enumerate(paths):
        print(f"\nPath {i}:")
        print(f"Color: {color}")
        print(f"Number of points: {len(path)}")
        if len(path) > 0:
            print(f"First point: {path[0]}")
            print(f"Last point: {path[-1]}")

    with open("paths.json", "w") as f:
        json.dump(paths, f, indent=4)

    print("\nPaths exported to paths.json")
