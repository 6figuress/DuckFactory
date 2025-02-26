from PIL import Image
import pytest
import numpy as np

from duck_factory.color_zone import color_zones


def images_are_identical(image1: Image, image2: Image) -> bool:
    for y in range(image1.size[1]):
        for x in range(image1.size[0]):
            if image1.getpixel((x, y)) != image2.getpixel((x, y)):
                return False
    return True


def test_color_zones_raises_value_error_on_different_sizes() -> None:
    base = Image.new("RGB", (10, 10))
    mask = Image.new("RGB", (5, 5))
    replacements = {}

    with pytest.raises(ValueError):
        color_zones(base, mask, replacements)


def test_color_zones_replaces_colors_correctly() -> None:
    base = Image.new("RGB", (2, 2), color=(255, 255, 255))
    mask = Image.new("RGB", (2, 2), color=(0, 0, 0))

    replacements = {(0, 0, 0): (255, 0, 0)}
    expected = Image.new("RGB", (2, 2), color=(255, 0, 0))

    result = color_zones(base, mask, replacements)

    assert images_are_identical(result, expected)


def test_color_zones_replaces_multiple_colors_correctly() -> None:
    base = Image.new("RGB", (2, 2), color=(255, 255, 255))
    mask = Image.fromarray(
        np.array(
            [
                [(0, 0, 0), (0, 0, 0)],
                [(255, 255, 0), (255, 0, 255)],
            ],
            dtype=np.uint8,
        ),
        "RGB",
    )

    replacements = {
        (255, 255, 0): (0, 255, 0),
        (255, 0, 255): (0, 0, 255),
    }
    expected = Image.fromarray(
        np.array(
            [
                [(255, 255, 255), (255, 255, 255)],
                [(0, 255, 0), (0, 0, 255)],
            ],
            dtype=np.uint8,
        ),
        "RGB",
    )

    result = color_zones(base, mask, replacements)

    assert images_are_identical(result, expected)


def test_color_zones_works_with_no_replacements() -> None:
    base = Image.new("RGB", (2, 2), color=(255, 255, 255))
    mask = Image.new("RGB", (2, 2), color=(0, 0, 0))

    replacements = {}
    expected = Image.new("RGB", (2, 2), color=(255, 255, 255))

    result = color_zones(base, mask, replacements)

    assert images_are_identical(result, expected)
