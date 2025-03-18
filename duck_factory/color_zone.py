from PIL import Image

Color = tuple[int, int, int]


def color_zones(base: Image, mask: Image, replacements: dict[Color, Color]) -> Image:
    """
    Replaces colors in parts of an image defined by a mask.

    Args:
        base: The base image to colorize.
        mask: The mask image that defines the zones to colorize.
        replacements: A dictionary that maps mask colors to a replacement color.

    Returns:
        The new image with the colors replaced.

    Raises:
        ValueError: If the base and mask images have different sizes.
    """
    if base.size != mask.size:
        raise ValueError("Base and mask images must have the same size")

    out: Image = Image.new("RGB", base.size)

    # Make sure images are in RGB mode
    base = base.convert("RGB")
    mask = mask.convert("RGB")

    for y in range(base.size[1]):
        for x in range(base.size[0]):
            mask_color = mask.getpixel((x, y))
            base_color = base.getpixel((x, y))

            # Get the replacement color or use the base color
            out_color = replacements.get(mask_color, base_color)

            out.putpixel((x, y), out_color)

    return out
