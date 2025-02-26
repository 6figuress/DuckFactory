from PIL import Image

import matplotlib.pyplot as plt

def parse_obj(file_path):
    vertices = {}  # Store vertices by index
    textures = {}  # Store texture coordinates by index
    normals = {}   # Store normals by index
    faces = []     # Store face definitions

    vertex_count = 1  # Indexing starts from 1 in .obj
    texture_count = 1
    normal_count = 1

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()

            if not parts or parts[0].startswith('#'):  # Skip comments or empty lines
                continue

            if parts[0] == 'v':  # Vertex
                vertices[vertex_count] = tuple(map(float, parts[1:4]))
                vertex_count += 1

            elif parts[0] == 'vt':  # Texture coordinate
                textures[texture_count] = tuple(map(float, parts[1:3]))  # Only (u, v)
                texture_count += 1

            elif parts[0] == 'vn':  # Normal
                normals[normal_count] = tuple(map(float, parts[1:4]))
                normal_count += 1

            elif parts[0] == 'f':  # Faces
                face_vertices = []
                for vert in parts[1:]:
                    v_indices = vert.split('/')

                    v_idx = int(v_indices[0]) if v_indices[0] else None
                    vt_idx = int(v_indices[1]) if len(v_indices) > 1 and v_indices[1] else None
                    vn_idx = int(v_indices[2]) if len(v_indices) > 2 and v_indices[2] else None

                    face_vertices.append((v_idx, vt_idx, vn_idx))

                faces.append(face_vertices)

    return vertices, textures, normals, faces

"""
# Example Usage
obj_file = "Dithering_test/Génère_moi_un_canar_0217112440_texture.obj"
vertices, textures, normals, faces = parse_obj(obj_file)

# Print some extracted data
print("Vertices:", list(vertices.items())[:5])  # Print first 5 vertices
print("Texture Coordinates:", list(textures.items())[:5])  # Print first 5 texture coordinates
print("Normals:", list(normals.items())[:5])  # Print first 5 normals
print("Faces:", faces[:5])  # Print first 5 faces
"""

def parse_obj_with_data(file_path):
    vertices = {}  # Store vertices by index
    textures = {}  # Store texture coordinates by index
    normals = {}   # Store normals by index
    faces = []     # Store face data with actual values

    vertex_count = 1  # Indexing starts from 1 in .obj
    texture_count = 1
    normal_count = 1

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()

            if not parts or parts[0].startswith('#'):  # Skip comments or empty lines
                continue

            if parts[0] == 'v':  # Vertex
                vertices[vertex_count] = tuple(map(float, parts[1:4]))
                vertex_count += 1

            elif parts[0] == 'vt':  # Texture coordinate
                textures[texture_count] = tuple(map(float, parts[1:3]))  # Only (u, v)
                texture_count += 1

            elif parts[0] == 'vn':  # Normal
                normals[normal_count] = tuple(map(float, parts[1:4]))
                normal_count += 1

            elif parts[0] == 'f':  # Faces
                face_data = []
                for vert in parts[1:]:
                    v_indices = vert.split('/')

                    v_idx = int(v_indices[0]) if v_indices[0] else None
                    vt_idx = int(v_indices[1]) if len(v_indices) > 1 and v_indices[1] else None
                    vn_idx = int(v_indices[2]) if len(v_indices) > 2 and v_indices[2] else None

                    # Get the actual vertex, texture, and normal data
                    vertex = vertices.get(v_idx, None)
                    texture = textures.get(vt_idx, None)
                    normal = normals.get(vn_idx, None)

                    face_data.append((vertex, texture, normal))  # Store full data

                faces.append(face_data)

    return vertices, textures, normals, faces

def get_pixel_color(image_path, coords):
    """
    Reads an image and returns the RGB color at the specified normalized (x, y) coordinates.
    
    :param image_path: Path to the image file
    :param coords: Tuple (x_norm, y_norm) where x and y are normalized (0 to 1)
    :return: (R, G, B) tuple
    """
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Get image dimensions
    width, height = image.size

    # Convert normalized coordinates to absolute pixel values
    x_pixel = int(coords[0] * (width - 1))  # Scale x to pixel width
    y_pixel = int(coords[1] * (height - 1)) # Scale y to pixel height

    # Retrieve the pixel color (PIL uses (x, y) indexing)
    pixel_color = image.getpixel((x_pixel, y_pixel))

    return pixel_color  # Return as (R, G, B)


def plot_3d_points(points):
    """
    Plots a 3D scatter plot from a list of points.

    :param points: List of (x, y, z) tuples representing 3D coordinates
    """
    # Unpack x, y, and z coordinates
    x_vals, y_vals, z_vals = zip(*points)  # Extract each coordinate list

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the points
    ax.scatter(x_vals, y_vals, z_vals, c='blue', marker='o')

    # Labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot')

    # Show the plot
    plt.show()

def plot_faces_points(faces):
    """
    Plots a 3D scatter plot with one point per face.

    :param faces: List of faces, where each face is a list of (vertex, texture, normal) tuples.
                  The vertex is stored as (x, y, z).
    """
    face_points = []

    for face in faces:
        if face:  # Ensure the face has at least one vertex
            vertex = face[0][0]  # Extract the first vertex (x, y, z) from the face
            if vertex:
                face_points.append(vertex)

    # Unpack x, y, and z coordinates
    x_vals, y_vals, z_vals = zip(*face_points)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the selected points
    ax.scatter(x_vals, y_vals, z_vals, c='red', marker='o')

    # Labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('One Point per Face in 3D')

    # Show the plot
    plt.show()

def plot_faces_points_with_color(faces, image_path):
    """
    Plots a 3D scatter plot with one point per face, colored based on an image.

    :param faces: List of faces, where each face is a list of (vertex, texture, normal) tuples.
                  The vertex is stored as (x, y, z), and the texture contains (u, v).
    :param image_path: Path to the texture image for color retrieval.
    """
    face_points = []
    colors = []

    for face in faces:
        if face:  # Ensure the face has at least one vertex
            vertex, texture, normal = face[0]  # Extract first vertex and its texture coordinates
            if vertex and texture:
                face_points.append(vertex)  # Store the vertex (x, y, z)

                # Normalize texture coordinates (u, v) for color retrieval
                uv_coords = (texture[0], 1 - texture[1])  # Flip V coordinate for image mapping
                color = get_pixel_color(image_path, uv_coords)
                
                # Normalize color to [0,1] for Matplotlib
                colors.append([c / 255.0 for c in color])

    if not face_points:
        print("No valid face points to plot.")
        return

    # Unpack x, y, and z coordinates
    x_vals, y_vals, z_vals = zip(*face_points)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with colors
    ax.scatter(x_vals, y_vals, z_vals, c=colors, marker='o')

    # Labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('One Point per Face with Image-based Color')

    # Show the plot
    plt.show()

# Example Usage
obj_file = "Dithering_test/Génère_moi_un_canar_0217112440_texture.obj"
vertices, textures, normals, faces = parse_obj_with_data(obj_file)

# print the number of faces
print("Number of faces: ", len(faces))

# Print some extracted data
# print("First 5 Faces with Data:")
# for face in faces[:5]:
#     print(face)  # Each face is now a list of tuples containing (vertex, texture, normal)
#     face_colors = [get_pixel_color("Dithering_test/Génère_moi_un_canar_0217112440_texture.png", tex) for _, tex, _ in face]
#     print("Face Colors:", face_colors)  # Get the colors of the texture coordinates


# plot_faces_points(faces[:1000])  # Plot one point per face in 3D

# shuffle the faces
import random
random.shuffle(faces)

plot_faces_points_with_color(faces[:2000], "Dithering_test/Génère_moi_un_canar_0217112440_texture.png")  # Plot with image-based color


