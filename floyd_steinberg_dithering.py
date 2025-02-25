# Based on https://scipython.com/blog/floyd-steinberg-dithering/

import numpy as np
from PIL import Image
from pathlib import Path

img_name = 'rubber_duck_image_axe.jpg'

def get_new_val(old_val, nc):
    """
    Get the "closest" colour to old_val in the range [0,1] per channel divided
    into nc values.

    """

    return np.round(old_val * (nc - 1)) / (nc - 1)

def fs_dither(img, nc):
    """
    Floyd-Steinberg dither the image img into a palette with nc colours per
    channel.

    """

    arr = np.array(img, dtype=float) / 255

    width, height = img.size

    for ir in range(height):
        for ic in range(width):
            # NB need to copy here for RGB arrays otherwise err will be (0,0,0)!
            old_val = arr[ir, ic].copy()
            new_val = get_new_val(old_val, nc)
            arr[ir, ic] = new_val
            err = old_val - new_val
            # In this simple example, we will just ignore the border pixels.
            if ic < width - 1:
                arr[ir, ic+1] += err * 7/16
            if ir < height - 1:
                if ic > 0:
                    arr[ir+1, ic-1] += err * 3/16
                arr[ir+1, ic] += err * 5/16
                if ic < width - 1:
                    arr[ir+1, ic+1] += err / 16

    max_val = np.max(arr, axis=(0,1))
    if np.any(max_val > 0):
        arr /= max_val
    carr = np.array(arr * 255, dtype=np.uint8)

    return Image.fromarray(carr)

def palette_reduce(img, nc):
    """Simple palette reduction without dithering."""
    arr = np.array(img, dtype=float) / 255
    arr = get_new_val(arr, nc)

    carr = np.array(arr/np.max(arr) * 255, dtype=np.uint8)
    return Image.fromarray(carr)

def rename_file(file_name, new_name):
    p = Path(file_name)
    p.rename(new_name)
def add_suffix(file_name, suffix):
    p = Path(file_name)
    p.rename(p.stem + suffix + p.suffix)


def rescale_image(img, factor):
    height, width = img.size
    new_size = (int(height * factor), int(width * factor))
    return img.resize(new_size)



img = Image.open(img_name)

s_width, s_height = img.size

img = rescale_image(img, 0.5)

if img.mode != 'RGB':
    img = img.convert('RGB')

dim = fs_dither(img, nc=2)
img.close()
add_suffix(img_name, '_original')
dim = dim.resize((s_width, s_height))
dim.save(img_name)
