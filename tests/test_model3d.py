import pytest
from PIL import Image
import os
from duck_factory.model3d import Vertex, Texture, Normal, Point, Face, Mesh

@pytest.fixture
def sample_vertex():
    return Vertex(1.0, 2.0, 3.0)

@pytest.fixture
def sample_normal():
    return Normal(0.0, 1.0, 0.0)

@pytest.fixture
def sample_texture():
    return Texture(0.5, 0.5)

@pytest.fixture
def sample_point(sample_vertex, sample_normal, sample_texture):
    return Point(sample_vertex, sample_normal, sample_texture)

@pytest.fixture
def sample_face(sample_point):
    return Face(sample_point, sample_point, sample_point)

@pytest.fixture
def mock_texture_image(tmp_path):
    image_path = tmp_path / "test_texture.png"
    image = Image.new('RGB', (100, 100), color=(255, 0, 0))
    image.save(image_path)
    return str(image_path)

@pytest.fixture
def mock_obj_file(tmp_path):
    obj_path = tmp_path / "test_model.obj"
    obj_content = """
    v 0.0 0.0 0.0
    v 1.0 0.0 0.0
    v 0.0 1.0 0.0
    vt 0.0 0.0
    vt 1.0 0.0
    vt 0.0 1.0
    vn 0.0 0.0 1.0
    f 1/1/1 2/2/1 3/3/1
    """.strip()
    obj_path.write_text(obj_content)
    return str(obj_path)

# Tests for Vertex class
def test_vertex(sample_vertex):
    assert sample_vertex.x == 1.0
    assert sample_vertex.y == 2.0
    assert sample_vertex.z == 3.0

# Tests for Normal class
def test_normal(sample_normal):
    assert sample_normal.x == 0.0
    assert sample_normal.y == 1.0
    assert sample_normal.z == 0.0

# Tests for Texture class
def test_texture(sample_texture):
    assert sample_texture.u == 0.5
    assert sample_texture.v == 0.5
    assert sample_texture.color == (255, 255, 255)  # Default white color

# Tests for Point class
def test_point(sample_point):
    assert sample_point.get_coordinates() == (1.0, 2.0, 3.0)
    assert sample_point.get_color() == (255, 255, 255)

# Tests for Face class
def test_face(sample_face):
    assert isinstance(sample_face.p1, Point)
    assert isinstance(sample_face.p2, Point)
    assert isinstance(sample_face.p3, Point)

# Tests for Mesh class
def test_mesh_loading(mock_obj_file, mock_texture_image):
    mesh = Mesh(mock_obj_file, mock_texture_image)
    assert len(mesh.vertices) == 3
    assert len(mesh.textures) == 3
    assert len(mesh.normals) == 1
    assert len(mesh.faces) == 1

    # Check if texture is properly loaded
    assert mesh.image is not None

# Tests for texture update
def test_mesh_texture_update(mock_obj_file, mock_texture_image, tmp_path):
    new_texture_path = tmp_path / "new_texture.png"
    new_texture = Image.new('RGB', (100, 100), color=(0, 255, 0))
    new_texture.save(new_texture_path)
    
    mesh = Mesh(mock_obj_file, mock_texture_image)
    mesh.update_texture(str(new_texture_path))
    assert mesh.image is not None

# Tests for point cloud extraction
def test_get_point_cloud(mock_obj_file, mock_texture_image):
    mesh = Mesh(mock_obj_file, mock_texture_image)
    point_cloud = mesh.get_point_cloud()
    assert len(point_cloud) == 3  # One triangle has 3 points
    assert isinstance(point_cloud[0], tuple)
    assert isinstance(point_cloud[0][0], tuple)
    assert isinstance(point_cloud[0][1], tuple)