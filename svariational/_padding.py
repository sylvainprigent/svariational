import math
import numpy as np


def zero_padding(image, final_shape):
    out_image = np.zeros(final_shape)
    offset_x = int((final_shape[0] - image.shape[0]) / 2)
    offset_y = int((final_shape[1] - image.shape[1]) / 2)

    out_image[offset_x:offset_x+image.shape[0],
              offset_y:offset_y+image.shape[1]] = image
    return out_image


def mirror_padding(image, size_x, size_y):
    """add a mirror padding to an image

    Parameters
    ----------
    image: np.array
        2D image as an np array
    size_x: int
        Size of the padding in the X direction
    size_y: int
        Size of the padding in the Z direction

    """
    out_image = np.zeros((image.shape[0]+2*size_x, image.shape[1]+2*size_y))
    shape_out = (image.shape[0]+2*size_x, image.shape[1]+2*size_y)
    # copy the input image
    out_image[size_x:-size_x, size_y:-size_y] = image

    # mirror horizontal
    for i in range(0, size_x):
        # top
        out_image[size_x-i-1, size_y:-size_y] = image[i, :]
        # bottom
        out_image[size_x+image.shape[0]+i, size_y:-size_y] = image[image.shape[0]-1-i, :]

    # mirror vertical
    for j in range(0, size_y):
        # left
        out_image[size_x:-size_x, size_y-j-1] = image[:, j]
        # right
        out_image[size_x:-size_x, size_y + image.shape[1] + j] = image[:, image.shape[1] - 1 - j]

    # corners
    for x in range(0, size_x):
        # top left corner
        for y in range(0, size_y):
            out_image[x, y] = out_image[(2*size_x-x), y]
        # bottom left corner
        for y in range(image.shape[0] + size_y, shape_out[1]):
            out_image[x, y] = out_image[(2*size_x-x), y]

    for x in range(0, size_x):
        # top right corner
        for y in range(0, size_y):
            out_image[x+image.shape[0]+size_x, y] = out_image[(image.shape[0]+size_x-x-1), y]
        # bottom right corner
        for y in range(image.shape[1] + size_y, shape_out[1]):
            out_image[x+image.shape[0]+size_x, y] = out_image[image.shape[0]+size_x-x-1, y]

    return out_image


def mirror_hanning_padding(image, size_x, size_y):
    mirror_image = mirror_padding(image, size_x, size_y)

    sx_out = mirror_image.shape[0]
    sy_out = mirror_image.shape[1]
    sx = image.shape[0]
    sy = image.shape[1]
    buffer_out = np.zeros((sx_out, sy_out))

    padding_x = int((int(sx_out) - int(sx)) / 2)
    padding_y = int((int(sy_out) - int(sy)) / 2)

    hann_N_x = 2 * padding_x
    hann_N_y = 2 * padding_y

    for y in range(0, sy_out):
        # vertical left
        for x in range(0, padding_x-1):
            coef = 0.5 * (1 - math.cos(2 * 3.14 * x / hann_N_x))
            mirror_image[x, y] = mirror_image[x, y] * coef
        # vertical right
        for x in range(1, padding_x):
            coef = 0.5 * (1 - math.cos(2 * 3.14 * (x-1) / hann_N_x))
            mirror_image[sx_out-x, y] = mirror_image[sx_out-x, y] * coef

    for x in range(0, sx_out):
        # horizontal top
        for y in range(0, padding_y-1):
            coef = 0.5 * (1 - math.cos(2 * 3.14 * y / hann_N_y))
            mirror_image[x, y] = mirror_image[x, y] * coef
        # horizontal bottom
        for y in range(1, padding_y):
            coef = 0.5 * (1 - math.cos(2 * 3.14 * (y-1) / hann_N_y))
            mirror_image[x, sy_out-y] = mirror_image[x, sy_out-y] * coef
    return mirror_image