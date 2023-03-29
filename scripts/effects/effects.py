
import cv2
import numpy as np
from numpy.typing import NDArray

def gaussian_blur(img: NDArray, size: int = 5, sigma: float = 0):
    """
    Default params: size=5, sigma=0
    params: size (odd), sigma
    """
    return cv2.GaussianBlur(img, (size, size), sigma)

def dither(img: NDArray, bits: int = 1):
    """
    Default params: bits=1
    params: bits (1-8)
    """
    if bits < 1 or bits > 8:
        raise ValueError("Quantization level must be between 1 and 8")

    height, width, _ = img.shape
    out = np.copy(img).astype(np.float32)
    levels = 2 ** bits - 1

    for y in range(height):
        for x in range(width):
            old_pixel = out[y, x]
            new_pixel = np.round(old_pixel / 255 * levels) / levels * 255
            out[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < width:
                out[y, x + 1] += error * 7/16
            if x - 1 >= 0 and y + 1 < height:
                out[y + 1, x - 1] += error * 3/16
            if y + 1 < height:
                out[y + 1, x] += error * 5/16
            if x + 1 < width and y + 1 < height:
                out[y + 1, x + 1] += error * 1/16

    return (np.clip(out, 0, 255)).astype(np.uint8)


def sharpen(img: NDArray, kernel_size: int = 3):
    """
    Default params: kernel_size=3
    params: kernel_size (odd)
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) * -1
    center = kernel_size // 2
    kernel[center, center] = kernel_size**2
    return cv2.filter2D(img, -1, kernel)