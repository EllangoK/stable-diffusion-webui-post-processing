import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from PIL import Image
import cv2

def luminance(rgb):
    return 0.2126 * (rgb[0]/255) + 0.7152 * (rgb[1]/255) + 0.0722 * (rgb[2]/255)

def luminace_mask(image, low_threshold=0.4, high_threshold=0.72):
    image = np.array(image)
    height, width, _ = image.shape
    mask = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            l = luminance(image[y, x])
            if low_threshold <= l <= high_threshold:
                mask[y, x] = 1
    return mask

def canny_mask(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    mask = np.where(edges > 0, 1, 0).astype(np.uint8)
    return mask


def sort_span(span, sort_by, reverse_sorting):
    if sort_by == 0:
        key = lambda x: x[1][0]
    elif sort_by == 1:
        key = lambda x: x[1][1]
    else:
        key = lambda x: x[1][2]

    span = sorted(span, key=key, reverse=reverse_sorting)
    return [x[0] for x in span]


def find_spans(mask, span_limit=None):
    spans = []
    start = None
    for i, value in enumerate(mask):
        if value == 0 and start is None:
            start = i 
        if value == 1 and start is not None:
            span_length = i - start
            if span_limit is None or span_length <= span_limit:
                spans.append((start, i))
            start = None
    if start is not None:
        span_length = len(mask) - start
        if span_limit is None or span_length <= span_limit:
            spans.append((start, len(mask)))

    return spans


def pixel_sort(img, low_threshold=0.4, high_threshold=0.72, mask_type: str = "luminance", horizontal_sort=False, span_limit=None, sort_by=0, reverse_sorting=False):
    """
    Default params: low_threshold=0.4, high_threshold=0.72, mask_type="luminance", horizontal_sort=False, span_limit=None, sort_by=0, reverse_sorting=False
    params: low_threshold, high_threshold, mask_type, horizontal_sort, span_limit, sort_by, reverse_sorting
    """
    height, width, _ = img.shape
    hsv_image = rgb2hsv(img)

    if mask_type == "luminance":
        mask = luminace_mask(img, low_threshold, high_threshold)
    else:
        mask = canny_mask(img, low_threshold, high_threshold)

    # loop over the rows and replace contiguous bands of 1s
    for i in range(height if horizontal_sort else width):
        in_band = False
        start = None
        end = None
        for j in range(width if horizontal_sort else height):
            if (mask[i, j] if horizontal_sort else mask[j, i]) == 1:
                if not in_band:
                    in_band = True
                    start = j
                end = j
            else:
                if in_band:
                    for k in range(start+1, end):
                        if horizontal_sort:
                            mask[i, k] = 0
                        else:
                            mask[k, i] = 0
                    in_band = False

        if in_band:
            for k in range(start+1, end):
                if horizontal_sort:
                    mask[i, k] = 0
                else:
                    mask[k, i] = 0


    mask_image = (mask * 255).astype(np.uint8)
    sorted_image = np.zeros_like(img)
    if horizontal_sort:
        for y in range(height):
            row_mask = mask[y]
            spans = find_spans(row_mask, span_limit)
            sorted_row = np.copy(img[y])
            for start, end in spans:
                span = [(img[y, x], hsv_image[y, x]) for x in range(start, end)]
                sorted_span = sort_span(span, sort_by, reverse_sorting)
                for i, pixel in enumerate(sorted_span):
                    sorted_row[start + i] = pixel
            sorted_image[y] = sorted_row
    else:
        for x in range(width):
            column_mask = mask[:, x]
            spans = find_spans(column_mask, span_limit)
            sorted_column = np.copy(img[:, x])
            for start, end in spans:
                span = [(img[y, x], hsv_image[y, x]) for y in range(start, end)]
                sorted_span = sort_span(span, sort_by, reverse_sorting)
                for i, pixel in enumerate(sorted_span):
                    sorted_column[start + i] = pixel
            sorted_image[:, x] = sorted_column

    return sorted_image
