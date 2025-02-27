from PIL import Image
import matplotlib.pyplot as plt
import random
import os

class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Texture:
    def __init__(self, u, v, image=None):
        self.u = u
        self.v = v
        self.color = (0, 0, 0)
        self.set_color(image)

    def extract_pixel_color(self, image):
        width, height = image.size
        x_pixel = int(self.u * (width - 1))
        y_pixel = int(self.v * (height - 1))

        return image.getpixel((x_pixel, y_pixel))
    
    def set_color(self, image):
        if image:
            self.color = self.extract_pixel_color(image)

class Normal:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Point:
    def __init__(self, vertex, normal, texture):
        self.vertex = vertex
        self.normal = normal
        self.texture = texture

    def get_coordinates(self):
        return (self.vertex.x, self.vertex.y, self.vertex.z)
    
    def get_color(self):
        return self.texture.color
    
    def get_texture_coordinates(self):
        return (self.get_coordinates(), self.get_color())


class Face:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            points = args[0]
        elif len(args) == 3:
            points = args
        else:
            raise ValueError("Invalid arguments for Face constructor")
        
        if not all(isinstance(point, Point) for point in points):
            raise ValueError("All elements of the Face must be of type Point")
        
        if len(points) != 3:
            raise ValueError("Face must have 3 points")
        
        self.p1, self.p2, self.p3 = points

    
class Mesh:
    def __init__(self, file_path, texture_path=None):
        self.faces = []
        self.image = None
        self.vertices = []
        self.textures = []
        self.normals = []
        
        self.load_files(file_path, texture_path)
        self.extract_faces()

    def load_texture(self, texture_path):
        valid_texture_extensions = [".png", ".jpg", ".jpeg"]

        if texture_path:
            if not any(texture_path.endswith(ext) for ext in valid_texture_extensions):
                raise ValueError("Unsupported texture file format")
            if not os.path.exists(texture_path):
                raise FileNotFoundError(f"Texture file not found: {texture_path}")
            self.image = Image.open(texture_path)


    def load_files(self, file_path=None, texture_path=None):
        if file_path and file_path.endswith(".obj"):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"OBJ file not found: {file_path}")
            with open(file_path, "r") as file:
                self.file_lines = file.readlines()
        else:
            raise ValueError("Invalid OBJ file path")

        self.load_texture(texture_path)

    def extract_faces(self):
        for line in self.file_lines:
            parts = line.strip().split()

            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v":
                self.vertices.append(self.extract_vertex(parts))
            elif parts[0] == "vt":
                self.textures.append(self.extract_texture(parts))
            elif parts[0] == "vn":
                self.normals.append(self.extract_normal(parts))
            elif parts[0] == "f":
                self.faces.append(self.extract_face(parts))
    
    def extract_vertex(self, parts):
        if len(parts) != 4:
            raise ValueError("Invalid vertex format in OBJ file")
        return Vertex(float(parts[1]), float(parts[2]), float(parts[3]))
    
    def extract_texture(self, parts):
        if len(parts) < 3:
            raise ValueError("Invalid texture coordinate format in OBJ file")
        return Texture(float(parts[1]), float(parts[2]), self.image)
    
    def extract_normal(self, parts):
        if len(parts) != 4:
            raise ValueError("Invalid normal format in OBJ file")
        return Normal(float(parts[1]), float(parts[2]), float(parts[3]))
    
    def extract_face(self, parts):
        if len(parts) != 4:
            raise ValueError("Invalid face format in OBJ file, must have 3 vertices")
        points = []
        for i in range(1, 4):
            indices = parts[i].split("/")
            if len(indices) != 3:
                raise ValueError("Face format should be v/vt/vn")
            vertex_index, texture_index, normal_index = map(int, indices)
            points.append(Point(
                self.vertices[vertex_index - 1],
                self.normals[normal_index - 1],
                self.textures[texture_index - 1]
            ))
        return Face(points)
    
    def get_point_cloud(self):
        return [point.get_texture_coordinates() for face in self.faces for point in (face.p1, face.p2, face.p3)]
    
    def update_texture(self, texture_path):
        self.load_texture(texture_path)
        for texture in self.textures:
            texture.set_color(self.image)

    def plot(self, n_points=1000):
        points = self.get_point_cloud()
        random.shuffle(points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # display points to points.get_coordinates() and color to points.get_color()
        for point in points[:n_points]:
            color = [channel / 255 for channel in point[1]]
            ax.scatter(*point[0], c=[color], s=10)
        plt.show()

# Usage

# Creating a mesh object
mesh = Mesh("Dithering_test/Génère_moi_un_canar_0217112440_texture.obj", "Dithering_test/Génère_moi_un_canar_0217112440_texture.png")

# Displaying a subset of the mesh
mesh.plot(n_points=1000)

# Updating the texture of the mesh
mesh.update_texture("Dithering_test/Génère_moi_un_canar_0219074804_texture_fs.png")

# Displaying the mesh with the updated texture
mesh.plot(n_points=1000)

# Example of an usable output:
points = mesh.get_point_cloud()[:10]
for point in points:
    print(f"Coordinates: {point[0]}, Color: {point[1]}")

