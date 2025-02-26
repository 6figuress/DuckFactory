from PIL import Image


def apply_image(
    base: Image,
    decal: Image,
    rotation: float,
    box: tuple[tuple[int, int], tuple[int, int]],
) -> Image:
    """
    Apply a decal image to a base image at a given location while preserving aspect ratio.

    Args:
        base: The base image to apply the decal to.
        decal: The decal image to apply.
        rotation: The rotation to apply the decal with.
        box: The box to apply the decal in.

    Returns:
        The new image with the decal applied.
    """
    out = base.copy()
    rotated_decal = decal.rotate(rotation, expand=True)

    decal_width, decal_height = rotated_decal.size
    box_width = box[1][0] - box[0][0]
    box_height = box[1][1] - box[0][1]

    width_ratio = box_width / decal_width
    height_ratio = box_height / decal_height

    # Use the smaller ratio to maintain aspect ratio
    # This ensures the decal fits entirely within the box
    scale_factor = min(width_ratio, height_ratio)

    new_width = int(decal_width * scale_factor)
    new_height = int(decal_height * scale_factor)

    scaled_decal = rotated_decal.resize((new_width, new_height))

    # Calculate position to center the decal in the box
    x_offset = (box_width - new_width) // 2
    y_offset = (box_height - new_height) // 2

    out.paste(scaled_decal, (box[0][0] + x_offset, box[0][1] + y_offset), scaled_decal)

    return out
