from PIL import Image

type Color = tuple[int,int,int]

def color_zones(base: Image, mask: Image, replacements: dict[Color, Color]) -> Image:
    # create an image to store the new texture
    out: Image = Image.new("RGB", base.size)

    for y in range(base.size[1]):
        for x in range(base.size[0]):
            mask_color = mask.getpixel((x, y))
            base_color = base.getpixel((x, y))

            # replace the color if it is in the replacements dictionary, otherwise keep the original color
            out_color = replacements.get(mask_color, base_color)

            out.putpixel((x, y), out_color)

    return out


if __name__ == "__main__":
    base_file = "duck_base.png"
    mask_file = "duck_mask.png"
    replacements = {
        (255, 0, 0): (0, 255, 200),
        (0, 255, 0): (0, 255, 255),
    }

    # read base and mask files
    base_image = Image.open(base_file).convert("RGB")
    mask_image = Image.open(mask_file).convert("RGB")

    out = color_zones(base_image, mask_image, replacements)
    out.save("duck_colored.png")