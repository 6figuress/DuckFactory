import numpy as np
from trimesh import Trimesh, load_mesh
from trimesh.sample import sample_surface
from duck_factory.dither_class import Dither
from duck_factory.reachable_points import PathAnalyzer
from duck_factory.points_to_paths import PathFinder
from duck_factory.point_sampling import (
    sample_mesh_points,
    cluster_points,
    Point,
    Color,
)
from duck_factory.path_bounder import PathBounder
from scipy.spatial.transform import Rotation
import json
from collections import defaultdict

import time

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


def modify_mesh_position(mesh):
    mesh.vertices = np.column_stack(
        (
            -mesh.vertices[:, 0],  # -X
            mesh.vertices[:, 2],  # Z
            mesh.vertices[:, 1],  # Y
        )
    )

    # Transform all vertex normals, if they exist
    if mesh.vertex_normals is not None:
        mesh.vertex_normals = np.column_stack(
            (
                -mesh.vertex_normals[:, 0],
                mesh.vertex_normals[:, 2],
                mesh.vertex_normals[:, 1],
            )
        )

    mesh.apply_translation([0, 0, 0.05])
    return mesh


DEFAULT_DITHER = Dither(factor=1, algorithm="fs", nc=2)
DEFAULT_PATH_ANALYZER = PathAnalyzer(
    tube_length=5e1, diameter=2e-2, cone_height=1e-2, step_angle=36, num_vectors=12
)


DISPLAY_ORIENTATION = True


def mesh_to_paths(
    mesh: Trimesh,
    n_samples: int = 50_000,
    max_dist: float = 0.1,
    home_point: tuple[Point, Normal] = ((0, 0, 0.25), (0, 0, -1)),
    verbose: bool = False,
    ditherer: Dither = None,
    path_analyzer: PathAnalyzer = None,
    bbox_scale: float = 2,
    nz_threshold: float = -1,
) -> list[Path]:
    """
    Do the full conversion from a textured mesh to a list of IK-ready paths.

    Convert a mesh to a list of paths, each path being a 3d position and quaternion (x, y, z, w)
    and being of a certain color.

    Args:
        mesh: The mesh to convert to paths
        n_samples: The number of samples to take from the mesh
        max_dist: The maximum distance between two samples for neighborhood
        home_point: The point where the robot should start and end its paths, represented by a point and a normal

    Returns:
        List of paths, each containing a color and a list of PathPosition (point and quaternion)
    """
    mesh = modify_mesh_position(mesh)

    if ditherer is None:
        ditherer = DEFAULT_DITHER
    if path_analyzer is None:
        path_analyzer = DEFAULT_PATH_ANALYZER

    if verbose:
        dither_start = time.time()
        print("Starting dithering")
    # Dither the mesh's texture
    img = mesh.visual.material.image
    img = ditherer.apply_dithering(img.convert("RGB"))
    mesh.visual.material.image = img
    if verbose:
        print(f"Dithering took {time.time() - dither_start:.2f} seconds")
        print("Starting mesh sampling")
        start_sampling = time.time()

    # Sample points from the mesh and cluster them by proximity
    sampled_points = sample_mesh_points(
        mesh, base_color=BASE_COLOR, colors=COLORS, n_samples=n_samples
    )

    if verbose:
        print(f"Sampling took {time.time() - start_sampling:.2f} seconds")
        print("Starting sample surface")
        start_surface = time.time()

    # TODO: sample duck + stand
    mesh_points = sample_surface(mesh, n_samples, sample_color=False)[0]

    if verbose:
        print(f"Sample surface took {time.time() - start_surface:.2f} seconds")
        print("Starting point processing")
        start_processing = time.time()

    valid_points = []
    for point in sampled_points:
        # Find a valid orientation for the point
        valid, new_norm = path_analyzer.find_valid_orientation(
            point.coordinates, point.normal, mesh_points
        )

        if valid:
            point.normal = new_norm
        else:
            # Can't find a way to reach this point, skip it
            continue

        valid_points.append(point)

    if verbose:
        print(f"Point processing took {time.time() - start_processing:.2f} seconds")
        print("Starting clustering")
        start_clustering = time.time()

    # Cluster the points by proximity and color
    clusters = cluster_points(valid_points, distance_eps=max_dist)
    if verbose:
        print(f"Clustering took {time.time() - start_clustering:.2f} seconds")
        print("Starting path computation")
        start_path = time.time()

    # Compute the paths for each cluster, and store lists of paths by color
    color_paths = defaultdict(list)
    for points, color, is_noise in clusters:
        if is_noise:
            # The noise clusters contain points that we don't want to connect
            # Create a new path for each point
            for point in points:
                # color_paths[color].append([point])
                pass
        else:
            # Connect the points in the cluster to form paths
            path_finder = PathFinder(
                points=points,
                max_distance=max_dist,
                thickness=max_dist,
                angle_threshold_deg=20,
            )
            paths_positions = path_finder.find_paths()

            for path in paths_positions:
                color_paths[color].append(path)

    if verbose:
        print(f"Path computation took {time.time() - start_path:.2f} seconds")
        print("Starting path merging")
        start_merge = time.time()

    # Merge the paths for each color by inserting paths between them
    rpaths = []
    for color, paths in color_paths.items():
        bounder = PathBounder(
            mesh,
            path_analyzer,
            mesh_points,
            bbox_scale=bbox_scale,
            nz_threshold=nz_threshold,
        )

        # Convert paths to position-normal format
        prepped_paths = [
            [
                (
                    point.coordinates,
                    (-point.normal[0], -point.normal[1], -point.normal[2]),
                )
                for point in path
            ]
            for path in paths
        ]

        # Finish the path at the home point
        prepped_paths = [[home_point]] + prepped_paths + [[home_point]]

        # Merge the paths
        merged = bounder.merge_all_path(prepped_paths)

        merged = [(pos, norm) for pos, norm in merged if norm is not None]

        if DISPLAY_ORIENTATION:
            converted_path = merged
        else:
            # Convert normals to quaternions
            converted_path = [(pos, norm_to_quat(norm)) for pos, norm in merged]

        rpaths.append((color, converted_path))

    if verbose:
        print(f"Path merging took {time.time() - start_merge:.2f} seconds")
        print(f"Finished path computation in {time.time() - dither_start:.2f} seconds")

    # TODO: Merge the different colors paths together with pen-switching

    return rpaths


def norm_to_quat(normal: Normal) -> Quaternion:
    """
    Convert a normal vector to a quaternion, in the robot/simulator coordinate system.

    The normal vector is to be in the same coordinate system as the robot/simulator (z-up, x-forward, y-left),
    and it should be the normal of the surface at which the robot should point (not the direction the robot should point to).

    Args:
        normal: The normal vector to convert

    Returns:
        The quaternion representing the rotation to align the z-axis with the normal

    Raises:
        ValueError: If the resulting quaternion contains NaNs
    """
    # the normal points "away" from the point, we want our robot to point towards it
    # normal = (-normal[0], -normal[1], -normal[2])

    if np.allclose(normal, [0, 0, 0]):
        raise ValueError("Cannot normalize a zero vector (normal is [0, 0, 0])")

    # normalize the normal, just to be sure
    normal = normal / np.linalg.norm(normal)

    # handle edge cases where the normal is parallel to the z-axis
    if np.allclose(normal, [0, 0, 1]):
        quat = (0, 5.06e-4, 0, 9.9e-1)
        return quat / np.linalg.norm(quat)
    elif np.allclose(normal, [0, 0, -1]):
        quat = (0, 9.9e-1, 0, 5.06e-4)
        return quat / np.linalg.norm(quat)

    # the hand of the robot points towards the positive z-axis, so
    # we want the rotation that aligns the z-axis with our normal

    # find the rotation axis which is perpendicular to both the z-axis and the normal
    # (i.e. the axis perpendicular to the plane they form)
    rotation_axis = np.cross((0, 0, 1), normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # find the rotation angle between the z-axis and the normal
    # this is the angle to rotate around the rotation axis
    # dot(a, b) = |a| * |b| * cos(angle) = cos(angle) because |a| = |b| = 1
    cos_angle = np.dot([0, 0, 1], normal)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # convert the rotation to a quaternion
    r = Rotation.from_rotvec(angle * rotation_axis)
    quat = r.as_quat()

    # fail if the quat contains NaNs
    if np.isnan(quat).any():
        raise ValueError("Quaternion contains NaNs")

    # if one of the 4 components of the quat is too close to +/- 1, there's a risk for it to mess up the IK
    # so set these components to a value close to 1
    close_to_pos_1 = np.isclose(quat, 1)
    close_to_min_1 = np.isclose(quat, -1)
    quat[close_to_pos_1] = 9.9e-1
    quat[close_to_min_1] = -9.9e-1

    assert not np.isclose(quat, 1).any(), "Quaternion is too close to 1"
    assert not np.isclose(quat, -1).any(), "Quaternion is too close to -1"

    return quat


# --------------------------------------------------------------- MAIN ---------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    mesh = load_mesh("cube_8mm.obj")

    dither = Dither(factor=0.1, algorithm="SimplePalette", nc=2)

    paths = mesh_to_paths(
        mesh, max_dist=0.0024, n_samples=50_000, verbose=True, ditherer=dither
    )

    # for color, path in paths:
    #     print(f"Color: {color}")
    #     for point in path:
    #         print(f"Point: {point}")
    #     print()

    print(f"Number of paths: {len(paths)}")
    print(f"Number of points: {sum([len(path) for _, path in paths])}")

    # convert all numpy arrays and tuples to lists
    paths = [
        (color, [(list(pos), list(quat)) for pos, quat in path])
        for color, path in paths
    ]

    # filter out positions where either the position or the quaternion is NaN
    paths = [
        (
            color,
            [
                (p, q)
                for p, q in path
                if not np.isnan(p).any() and not np.isnan(q).any()
            ],
        )
        for color, path in paths
    ]

    # with open("paths.json", "w") as f:
    #     json.dump(paths, f, indent=4)

    # print("Paths exported to paths.json")

    # ------------------------------- DISPLAY -------------------------------

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    box = mesh.bounding_box_oriented

    obb_vertices = box.vertices

    obb_faces = [[obb_vertices[i] for i in face] for face in box.faces]
    for i, face in enumerate(obb_faces):
        color = "lightblue"
        ax.add_collection3d(
            Poly3DCollection([face], alpha=0.3, edgecolor="black", facecolors=color)
        )

    for color, path in paths:
        coords = [pos for pos, _ in path]
        ax.plot(
            [c[0] for c in coords],
            [c[1] for c in coords],
            [c[2] for c in coords],
            label=str(color),
        )

        start_point, _ = path[0]
        end_point, _ = path[-1]

        if DISPLAY_ORIENTATION:
            # To be able to display the normal vectors, we need to disable the conversion to quaternions
            length = 0.05
            for pos, normal in path:
                # Draw the normal
                end_pos = (
                    np.array(pos) - np.array(normal) * length
                )  # Ending at the correct point
                ax.quiver(
                    end_pos[0],
                    end_pos[1],
                    end_pos[2],
                    normal[0],
                    normal[1],
                    normal[2],
                    color="magenta",
                    length=length,
                    normalize=True,
                )

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")

    plt.show()
