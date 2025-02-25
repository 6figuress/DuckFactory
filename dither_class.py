# Based on https://scipython.com/blog/floyd-steinberg-dithering/

import numpy as np
from PIL import Image

from argparse import ArgumentParser

class Dither():
    def __init__(self, factor=1, algorithm="fs", nc=2):
        self.factor = factor
        self.algorithm = algorithm
        self.nc = nc
        self.func = self.get_func(self.algorithm)

    def get_func(self, algorithm):
        """
        Get the function corresponding to the algorithm.
        """
        if algorithm == "fs":
            return self.floyd_steinberg_dither
        elif algorithm == "SimplePalette":
            return self.simple_palette_reduce
        else:
            raise ValueError("Invalid algorithm")
        
    def apply_threshold(self, value):
        """
        Get the "closest" colour to VALUE in the range [0,1] per channel divided
        into nc values.
        """
        return np.round(value * (self.nc - 1)) / (self.nc - 1)
    
    def floyd_steinberg_dither(self, img):
        """
        Floyd-Steinberg dither the image img into a palette with nc colours per channel.
        """
        arr = np.array(img, dtype=float) / 255

        width, height = img.size

        for ir in range(height):
            for ic in range(width):
                # need to copy here for RGB arrays otherwise err will be (0,0,0)!
                old_val = arr[ir, ic].copy()
                new_val = self.apply_threshold(old_val)
                arr[ir, ic] = new_val
                err = old_val - new_val
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
    
    def simple_palette_reduce(self, img):
        """Simple palette reduction without dithering."""
        arr = np.array(img, dtype=float) / 255
        arr = self.apply_threshold(arr)

        carr = np.array(arr/np.max(arr) * 255, dtype=np.uint8)
        return Image.fromarray(carr)
    
    def rescale_image(self, img):
        """
        Rescale the image by the factor.
        """
        return img.resize((int(img.size[0] * self.factor), int(img.size[1] * self.factor)))
    
    def apply_dithering(self, img):
        """
        Apply the dithering algorithm to the image.
        First rescale the image, then apply the dithering algorithm, and finally
        resize the image back to the original size.
        """
        original_size = img.size
        img = self.rescale_image(img)
        img = self.func(img)
        img = img.resize(original_size)
        return img

def main():
    parser = ArgumentParser(description="Image dithering")
    parser.add_argument("image_path", help="input image location")
    parser.add_argument("-o", help="output image location")
    args = parser.parse_args()

    dither = Dither(factor=0.5, algorithm="fs", nc=2)
    img = Image.open(args.image_path)
    img = dither.apply_dithering(img)
    if args.o:
        img.save(args.o)
    else:
        img.show()

    print("Done.")
    

if __name__ == "__main__":
    main()
    """
    # Example usage, to dither an image and save it to output.png:
    # python dither_class.py -o output.png input.png
    # 
    # Example usage to dither an image and display it:
    # python dither_class.py input.png
    """
        