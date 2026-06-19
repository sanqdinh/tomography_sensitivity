import numpy as np
from sympy.matrices.expressions.matexpr import MatrixElement
from pyomo.core.base.var import VarData as PyomoVarData

def get_line_abc_from_r_theta(r, theta):
    """
    This function calculates the line equation from a normal vector made up of two parameters: the angle and the distance from the origin.
    The line is represented as an equation ax + by + c = 0.

    Args:
        r (float): The distance from the origin.
        theta (float): The angle of the normal vector.
    Returns:
        numpy.ndarray: The line equation.

    Development note: Unit test generated.
    """
    a = np.cos(theta)
    b = np.sin(theta)
    c = -r
    return np.array([a, b, c])

def line_grid_intersections(a, b, c, image, x_range=[-10, 10], y_range=[-10, 10]):
    """
    Compute the intersection points between the line a*x + b*y + c = 0
    and the grid lines of a square grid with integer grid lines in the
    ranges x_range and y_range. For each intersection, return the point,
    the corresponding pixel indices, and the pixel value of the image.

    The image is assumed to be plotted with:
        height, width = image.shape
        extent = [-width/2, width/2, -height/2, height/2]

    Parameters:
        a, b, c : coefficients of the line equation.
        image   : 2D numpy array representing the image.
        x_range : list or tuple [x_min, x_max], default [-10, 10]
        y_range : list or tuple [y_min, y_max], default [-10, 10]

    Returns:
        A NumPy array of shape (N, 5) where each row is:
            [x, y, row_index, col_index, pixel_value]
        representing an intersection point (x, y), its corresponding pixel indices,
        and the pixel value.
    """
    intersections = set()
    x_min = int(x_range[0])
    x_max = int(x_range[1])
    y_min = int(y_range[0])
    y_max = int(y_range[1])

    # First, determine the grid intersections (in data coordinates)
    # Vertical grid lines: x = i for integer i in x_range.
    if b != 0:
        for i in range(x_min, x_max + 1):
            y = (-c - a * i) / b
            if y_min <= y <= y_max:
                intersections.add((i, y))
    else:
        # b == 0, line is vertical (if a != 0)
        if a != 0:
            x_val = -c / a
            if x_min <= x_val <= x_max:
                for j in range(y_min, y_max + 1):
                    intersections.add((x_val, j))
        # else degenerate: both a and b are 0

    # Horizontal grid lines: y = j for integer j in y_range.
    if a != 0:
        for j in range(y_min, y_max + 1):
            x = (-c - b * j) / a
            if x_min <= x <= x_max:
                intersections.add((x, j))
    else:
        # a == 0, so line is horizontal (if b != 0)
        if b != 0:
            y_val = -c / b
            if y_min <= y_val <= y_max:
                for i in range(x_min, x_max + 1):
                    intersections.add((i, y_val))
        # else degenerate

    # Prepare to sample the image
    height, width = image.shape
    # According to the provided extent, the image spans:
    #   x: [-width/2, width/2], y: [-height/2, height/2]
    # For a point (x, y) in data coordinates:
    #   col index = int(x + width/2)   [0 <= col < width]
    #   row index = int(height/2 - y)    [0 <= row < height]

    # result = []
    intersection_result = []
    image_intersection = []
    for x, y in sorted(intersections, key=lambda point: (point[0], point[1])):
        # Convert (x,y) to pixel indices (col, row)
        col = int(x + width / 2)
        row = int(height / 2 - y)
        # Check if the computed indices fall within the image bounds:
        if 0 <= col < width and 0 <= row < height:
            pixel_value = image[row, col]
            # result.append([x, y, row, col, pixel_value])
            intersection_result.append([x, y])
            image_intersection.append([row, col, pixel_value])
        # Otherwise, you could decide to ignore points that are outside the image.

    if isinstance(image_intersection[0][2], MatrixElement):
        radon = [0.0] * (len(intersection_result) - 1)
    elif isinstance(image_intersection[0][2],PyomoVarData):
        radon = [0.0] * (len(intersection_result) - 1)
    else:
        radon = np.zeros(len(intersection_result) - 1)
    intersection_lengths = np.zeros(len(intersection_result) - 1)
    for i in range(len(intersection_result) - 1):
        point = np.array(intersection_result[i])
        point_plus = np.array(intersection_result[i + 1])

        intersection_lengths[i] = np.linalg.norm(point - point_plus)
        radon[i] = intersection_lengths[i] * image_intersection[i][2]
    return (
        np.array(intersection_result),
        np.array(image_intersection),
        radon,
        intersection_lengths,
    )


def generate_x_y_coordinates(resolution, x_range=None, y_range=None):
    if x_range is None and y_range is None:
        xx_image, yy_image = np.meshgrid(
            np.linspace(
                start=-resolution[0] / 2 + 0.5,
                stop=resolution[0] / 2 - 0.5,
                num=resolution[0],
            ),
            np.linspace(
                start=-resolution[1] / 2 + 0.5,
                stop=resolution[1] / 2 - 0.5,
                num=resolution[1],
            ),
        )
    else:
        xx_image, yy_image = np.meshgrid(
            np.linspace(
                start=x_range[0],
                stop=x_range[1],
                num=resolution[0],
            ),
            np.linspace(
                start=y_range[0],
                stop=y_range[1],
                num=resolution[1],
            ),
        )

    return xx_image, yy_image

def recenter_image(image, x_range=None, y_range=None):

    if x_range is None:
        x_range = [-image.shape[0] / 2 + 0.5, image.shape[0] / 2 - 0.5]
    if y_range is None:
        y_range = [-image.shape[1] / 2 + 0.5, image.shape[1] / 2 - 0.5]

    # Change the index of the image to be centered at 0 and the axes to be in the correct orientation
    image = np.rot90(np.flip(image, axis=1), k=2)
    num_x_pixels = image.shape[0]
    num_y_pixels = image.shape[1]

    dx = (x_range[1] - x_range[0]) / (num_x_pixels - 1)
    dy = (y_range[1] - y_range[0]) / (num_y_pixels - 1)
    x_range_center = [x_range[0] + dx / 2, x_range[1] - dx / 2]
    y_range_center = [y_range[0] + dy / 2, y_range[1] - dy / 2]
    # xx_image, yy_image = generate_x_y_coordinates(
    #     [num_x_pixels, num_y_pixels], x_range=x_range_center, y_range=y_range_center
    # )
    xx_image, yy_image = generate_x_y_coordinates(
        [num_x_pixels, num_y_pixels], x_range=x_range, y_range=y_range
    )

    return image, xx_image, yy_image


def get_segment_polar(
    r_distance=0, angle=np.pi / 2, seg_range=[-10, 10], num_points=10
):
    # Unrotated segement
    d_seg = (seg_range[1] - seg_range[0]) / num_points
    segment_y = np.linspace(
        seg_range[1] - d_seg / 2, seg_range[0] + d_seg / 2, num_points
    )
    segment_x = np.ones(num_points) * r_distance

    # Rotate the segment
    segment_x_rotated = segment_x * np.cos(angle) - segment_y * np.sin(angle)
    segment_y_rotated = segment_x * np.sin(angle) + segment_y * np.cos(angle)

    return np.array(
        [[segment_x_rotated[i], segment_y_rotated[i]] for i in range(num_points)]
    )