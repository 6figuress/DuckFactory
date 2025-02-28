from PIL import Image
import matplotlib.pyplot as plt
import random
import os

class Vertex:
    """
    Represents a 3D vertex with x, y, and z coordinates.
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Texture:
    """
    Represents a texture coordinate and extracts color from an image.
    """
    def __init__(self, u, v, image=None):
        self.u = u
        self.v = v
        self.set_color(image)
    
    def set_color(self, image):
        """
        Sets the texture color by extracting it from the provided image.
        """
        if image:
            width, height = image.size
            x_pixel = int(self.u * (width - 1))
            y_pixel = int(self.v * (height - 1))

            self.color = image.getpixel((x_pixel, y_pixel))
        else:
            self.color = (255, 255, 255)

class Normal:
    """
    Represents a normal vector with x, y, and z components.
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Point:
    """
    Represents a point in the 3D space, including vertex, normal, and texture data.
    """
    def __init__(self, vertex, normal, texture):
        self.vertex = vertex
        self.normal = normal
        self.texture = texture

    def get_coordinates(self):
        """
        Returns the coordinates of the vertex.
        """
        return (self.vertex.x, self.vertex.y, self.vertex.z)
    
    def get_color(self):
        """
        Returns the texture color of the point.
        """
        return self.texture.color
    
    def get_texture_coordinates(self):
        """
        Returns the vertex coordinates and associated texture color.
        """
        return (self.get_coordinates(), self.get_color())


class Face:
    """
    Represents a triangular face consisting of three points.
    """
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
    """
    Represents a 3D mesh loaded from an OBJ file with optional texture mapping.
    """
    def __init__(self, file_path, texture_path=None):
        self.faces = []
        self.image = None
        self.vertices = []
        self.textures = []
        self.normals = []
        
        self.load_files(file_path, texture_path)
        self.parse_object()

    def load_texture(self, texture_path):
        """
        Loads the texture image from the specified path.
        """
        valid_texture_extensions = [".png", ".jpg", ".jpeg"]

        if texture_path:
            if not any(texture_path.endswith(ext) for ext in valid_texture_extensions):
                raise ValueError("Unsupported texture file format")
            if not os.path.exists(texture_path):
                raise FileNotFoundError(f"Texture file not found: {texture_path}")
            self.image = Image.open(texture_path)


    def load_files(self, file_path=None, texture_path=None):
        """
        Loads the OBJ file and associated texture file.
        """
        if file_path and file_path.endswith(".obj"):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"OBJ file not found: {file_path}")
            with open(file_path, "r") as file:
                self.file_lines = file.readlines()
        else:
            raise ValueError("Invalid OBJ file path")

        self.load_texture(texture_path)

    def parse_object(self):
        """
        Parses the OBJ file and extracts vertices, texture coordinates, normals, and faces.
        """
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
        """
        Parses a vertex definition from the OBJ file.
        """
        if len(parts) != 4:
            raise ValueError("Invalid vertex format in OBJ file")
        return Vertex(float(parts[1]), float(parts[2]), float(parts[3]))
    
    def extract_texture(self, parts):
        """
        Parses a texture coordinate definition from the OBJ file.
        """
        if len(parts) < 3:
            raise ValueError("Invalid texture coordinate format in OBJ file")
        return Texture(float(parts[1]), float(parts[2]), self.image)
    
    def extract_normal(self, parts):
        """
        Parses a normal vector definition from the OBJ file.
        """
        if len(parts) != 4:
            raise ValueError("Invalid normal format in OBJ file")
        return Normal(float(parts[1]), float(parts[2]), float(parts[3]))
    
    def extract_face(self, parts):
        """
        Parses a face definition and constructs a Face object.
        """
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
        """
        Returns the point cloud of the mesh with vertex coordinates and texture colors.
        """
        return [point.get_texture_coordinates() for face in self.faces for point in (face.p1, face.p2, face.p3)]
    
    def update_texture(self, texture_path):
        """
        Updates the texture of the mesh with a new image.
        """
        self.load_texture(texture_path)
        for texture in self.textures:
            texture.set_color(self.image)

    def plot(self, n_points=1000):
        """
        Displays a subset of the mesh points in a 3D plot.
        """
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
def main():
    # Creating a mesh object
    mesh = Mesh("duck_factory/model.obj", "duck_factory/Génère_moi_un_canar_0219074804_texture.png")

    # Displaying a subset of the mesh
    mesh.plot(n_points=1000)

    # Updating the texture of the mesh
    mesh.update_texture("duck_factory/Génère_moi_un_canar_0219074804_texture_fs.png")

    # Displaying the mesh with the updated texture
    mesh.plot(n_points=1000)

    # Example of an usable output:
    points = mesh.get_point_cloud()[:10]
    for point in points:
        print(f"Coordinates: {point[0]}, Color: {point[1]}")

if __name__ == "__main__":
    main()