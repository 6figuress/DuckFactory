import pytest
from PIL import Image
from duck_factory.apply_image import apply_image


# Helper function to create a test image
def create_test_image(width=50, height=50, color=(128, 128, 128, 255)):
    return Image.new("RGBA", (width, height), color)


@pytest.fixture
def base_image():
    return create_test_image()


@pytest.fixture
def decal_image():
    return create_test_image(width=20, height=20, color=(255, 0, 0, 255))


def test_apply_image(base_image, decal_image):
    rotation = 45
    box = ((10, 10), (40, 40))

    output_image = apply_image(base_image, decal_image, rotation, box)

    # Ensure the output is an Image object
    assert isinstance(output_image, Image.Image)

    # Ensure the output size matches the base image size
    assert output_image.size == base_image.size
