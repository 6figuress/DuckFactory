from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from typing import List, Optional, Tuple

from dataclasses import dataclass


@dataclass
class Vertex:
    """Represents a 3D vertex with x, y, and z coordinates."""

    x: float
    y: float
    z: float


class Texture:
    """Represents a texture coordinate and extracts color from an image."""

    def __init__(self, u: float, v: float, image: Image.Image = None):
        self.u = u
        self.v = v
        self.set_color(image)

    def set_color(self, image: Optional[Image.Image]) -> None:
        """
        Sets the texture color by extracting it from the provided image.

        Parameters:
        image (PIL.Image.Image): The image from which to extract the color. If None, the color is set to white.
        """
        if image:
            width, height = image.size
            x_pixel = int(self.u * (width - 1))
            y_pixel = int(self.v * (height - 1))

            self.color = image.getpixel((x_pixel, y_pixel))
        else:
            self.color = (255, 255, 255)


@dataclass
class Normal:
    """Represents a normal vector with x, y, and z components."""

    x: float
    y: float
    z: float


class Point:
    """Represents a point in the 3D space, including vertex, normal, and texture data."""

    def __init__(self, vertex: Vertex, normal: Normal, texture: Texture):
        self.vertex = vertex
        self.normal = normal
        self.texture = texture

    def get_coordinates(self) -> tuple:
        """Returns the coordinates of the vertex as a tuple (x, y, z)."""
        return (self.vertex.x, self.vertex.y, self.vertex.z)

    def get_color(self) -> tuple:
        """
        Get the texture color of the point as an RGB tuple.

        Returns the texture color of the point.

        Returns:
            tuple: The texture color as an RGB tuple.
        """
        return self.texture.color

    def get_texture_coordinates(self) -> tuple:
        """
        Get the vertex coordinates and associated texture color.

        Returns:
                tuple: A tuple containing the vertex coordinates and the associated texture color.
        """
        return (self.get_coordinates(), self.get_color())


class Face:
    """Represents a triangular face consisting of three points."""

    def __init__(self, *args: Point):
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

    Attributes:
        faces (list): List of faces in the mesh.
        image (PIL.Image.Image or None): Texture image.
        vertices (list): List of vertices in the mesh.
        textures (list): List of texture coordinates.
        normals (list): List of normal vectors.
        file_lines (list): Lines read from the OBJ file.

    Methods:
        __init__(file_path, texture_path=None): Initializes the Mesh object and loads files.
        load_texture(texture_path): Loads the texture image from the specified path.
        load_files(file_path=None, texture_path=None): Loads the OBJ file and associated texture file.
        parse_object(): Parses the OBJ file and extracts vertices, texture coordinates, normals, and faces.
        extract_vertex(parts): Parses a vertex definition from the OBJ file.
        extract_texture(parts): Parses a texture coordinate definition from the OBJ file.
        extract_normal(parts): Parses a normal vector definition from the OBJ file.
        extract_face(parts): Parses a face definition and constructs a Face object.
        get_point_cloud(): Returns the point cloud of the mesh with vertex coordinates and texture colors.
        update_texture(texture_path): Updates the texture of the mesh with a new image.
        plot(n_points=1000): Displays a subset of the mesh points in a 3D plot.
    """

    def __init__(self, file_path: str, texture_path: Optional[str] = None):
        self.faces = []
        self.image = None
        self.vertices = []
        self.textures = []
        self.normals = []

        self.load_files(file_path, texture_path)
        self.parse_object()

    def load_texture(self, texture_path: Optional[str]) -> None:
        """
        Loads the texture image from the specified path.

        Args:
            texture_path (Optional[str]): The file path to the texture image.

        Raises:
            ValueError: If the texture file format is unsupported.
            FileNotFoundError: If the texture file is not found.
        """
        valid_texture_extensions = [".png", ".jpg", ".jpeg"]

        if texture_path:
            if not any(texture_path.endswith(ext) for ext in valid_texture_extensions):
                raise ValueError("Unsupported texture file format")
            if not os.path.exists(texture_path):
                raise FileNotFoundError(f"Texture file not found: {texture_path}")
            self.image = Image.open(texture_path)

    def load_files(
        self, file_path: Optional[str] = None, texture_path: Optional[str] = None
    ) -> None:
        """
        Loads the OBJ file and associated texture file.

        Parameters:
        file_path (str, optional): The path to the OBJ file. Must end with '.obj'.
        texture_path (str, optional): The path to the texture file.

        Raises:
        FileNotFoundError: If the OBJ file does not exist.
        ValueError: If the file_path is not a valid OBJ file path.
        """
        if file_path and file_path.endswith(".obj"):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"OBJ file not found: {file_path}")
            with open(file_path, "r") as file:
                self.file_lines = file.readlines()
        else:
            raise ValueError("Invalid OBJ file path")

        self.load_texture(texture_path)

    def parse_object(self) -> None:
        """
        Parses the OBJ file and extracts vertices, texture coordinates, normals, and faces.

        This method iterates over each line in the OBJ file, identifies the type of data
        (vertex, texture coordinate, normal, or face), and appends the extracted data to
        the corresponding list.

        Attributes:
            self.file_lines (list of str): Lines of the OBJ file.
            self.vertices (list): List to store vertex coordinates.
            self.textures (list): List to store texture coordinates.
            self.normals (list): List to store normal vectors.
            self.faces (list): List to store face definitions.
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

    def extract_vertex(self, parts: list) -> Vertex:
        """
        Parses a vertex definition from the OBJ file.

        Args:
            parts (list): A list of strings representing the vertex components.

        Raises:
            ValueError: If the length of parts is not equal to 4.

        Returns:
            Vertex: A Vertex object with x, y, and z coordinates.
        """
        if len(parts) != 4:
            raise ValueError("Invalid vertex format in OBJ file")
        return Vertex(float(parts[1]), float(parts[2]), float(parts[3]))

    def extract_texture(self, parts: list) -> Texture:
        """
        Parses a texture coordinate definition from the OBJ file.

        Args:
            parts (list): A list of strings representing parts of the texture coordinate definition.

        Raises:
            ValueError: If the texture coordinate format in the OBJ file is invalid.

        Returns:
            Texture: A Texture object with u, v coordinates and color extracted from the image.
        """
        if len(parts) < 3:
            raise ValueError("Invalid texture coordinate format in OBJ file")
        return Texture(float(parts[1]), float(parts[2]), self.image)

    def extract_normal(self, parts: list) -> Normal:
        """
        Parses a normal vector definition from the OBJ file.

        Args:
            parts (list): A list of strings representing the components of the normal vector.

        Raises:
            ValueError: If the length of parts is not equal to 4.

        Returns:
            Normal: A Normal object with x, y, and z components.
        """
        if len(parts) != 4:
            raise ValueError("Invalid normal format in OBJ file")
        return Normal(float(parts[1]), float(parts[2]), float(parts[3]))

    def extract_face(self, parts: list) -> Face:
        """Parses a face definition and constructs a Face object.

        Args:
            parts (list): A list of strings representing the face definition.

        Raises:
            ValueError: If the face format is invalid.

        Returns:
            Face: A Face object constructed from the parsed definition.
        """
        if len(parts) != 4:
            raise ValueError("Invalid face format in OBJ file, must have 3 vertices")
        points = []
        for i in range(1, 4):
            indices = parts[i].split("/")
            if len(indices) != 3:
                raise ValueError("Face format should be v/vt/vn")
            vertex_index, texture_index, normal_index = map(int, indices)
            points.append(
                Point(
                    self.vertices[vertex_index - 1],
                    self.normals[normal_index - 1],
                    self.textures[texture_index - 1],
                )
            )
        return Face(points)

    def get_point_cloud(self) -> List[Tuple[float, float, float]]:
        """
        Returns the point cloud of the mesh with vertex coordinates and texture colors.

        Returns:
            List[Tuple[float, float, float]]: A list of tuples containing the texture coordinates of each vertex in the mesh.
        """
        return [
            point.get_texture_coordinates()
            for face in self.faces
            for point in (face.p1, face.p2, face.p3)
        ]

    def update_texture(self, texture_path: Optional[str]) -> None:
        """Updates the texture of the mesh with a new image."""
        self.load_texture(texture_path)
        for texture in self.textures:
            texture.set_color(self.image)

    def plot(self, n_points: int = 1000) -> None:
        """Displays a subset of the mesh points in a 3D plot."""
        points = self.get_point_cloud()
        random.shuffle(points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # display points to points.get_coordinates() and color to points.get_color()
        for point in points[:n_points]:
            color = [channel / 255 for channel in point[1]]
            ax.scatter(*point[0], c=[color], s=10)
        plt.show()


# Usage
def main() -> None:
    """Main function to create a mesh, display it, update its texture, and display it again."""
    # Creating a mesh object
    mesh = Mesh(
        "duck_factory/model.obj",
        "duck_factory/Génère_moi_un_canar_0219074804_texture.png",
    )

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
