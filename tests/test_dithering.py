import pytest
import numpy as np
from PIL import Image
from duck_factory.dither_class import Dither


# Helper function to create a test image
def create_test_image(width=10, height=10, color=(128, 128, 128)):
    img = Image.new("RGB", (width, height), color)
    return img


@pytest.fixture
def test_image():
    return create_test_image()


# Test rescaling function
def test_rescale_image(test_image):
    dither = Dither(factor=0.5)
    rescaled_img = dither.rescale_image(test_image)
    assert rescaled_img.size == (5, 5)

    dither = Dither(factor=2)
    rescaled_img = dither.rescale_image(test_image)
    assert rescaled_img.size == (20, 20)


# Test Floyd-Steinberg dithering
def test_floyd_steinberg_dither(test_image):
    dither = Dither(algorithm="fs", nc=2)
    dithered_img = dither.floyd_steinberg_dither(test_image)
    assert isinstance(dithered_img, Image.Image)
    assert dithered_img.size == test_image.size
    assert np.array(dithered_img).dtype == np.uint8


# Test Simple Palette Reduction
def test_simple_palette_reduce(test_image):
    dither = Dither(algorithm="SimplePalette", nc=2)
    dithered_img = dither.simple_palette_reduce(test_image)
    assert isinstance(dithered_img, Image.Image)
    assert dithered_img.size == test_image.size
    assert np.array(dithered_img).dtype == np.uint8


def test_apply_dithering(test_image):
    dither = Dither(algorithm="fs", nc=2, factor=0.5)
    dithered_img = dither.apply_dithering(test_image)
    assert isinstance(dithered_img, Image.Image)

    expected_size = (
        int(test_image.size[0] * dither.factor),
        int(test_image.size[1] * dither.factor),
    )
    assert dithered_img.size == expected_size
    assert np.array(dithered_img).dtype == np.uint8


# Test invalid algorithm
def test_invalid_algorithm():
    with pytest.raises(ValueError):
        Dither(algorithm="invalid")
