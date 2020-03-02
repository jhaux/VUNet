from PIL import Image, ImageDraw, ImageColor
import numpy as np

from VUNet.data.keypoint_models import OPENPOSE_18

    
COLORS = [(255, 0, 0, 0),
          (0, 255, 0, 0),
          (0, 0, 255, 0)]


def kp2stick(kps, size=[256, 256], kp_model=OPENPOSE_18):
    '''Makes a stickman image from a set of keypoints.

    Parameters
    ----------
    kps : np.ndarray
        Set of keypoints. Should have the shape ``[K, 2]`` with ``K`` the
        number of joints. Joint locations must be in absolute pixel
        coordinates. Values with x and y coordinate <= 0 will be ignored.
    size : list(int, int)
        The size of the output image.
    kp_model : object
        Defines which points are connected to a line, which points are left,
        right or center (i.e. on the R, G, B channel) and which points a used
        to draw the body polygon.
    '''

    # Create canvas
    im = Image.fromarray(np.zeros(size + [3], dtype='uint8'))
    draw = ImageDraw.Draw(im)

    # Draw Body Polygon
    body = []
    for idx in kp_model.CENTER_BODY:
        point = kps[idx].tolist()
        if point[0] <= 0 and point[1] <= 0:
            continue
        body += [tuple(point)]
    draw.polygon(body, fill=COLORS[1])

    # Draw Lines
    all_lines = [
        kp_model.LEFT_LINES,
        kp_model.CENTER_LINES,
        kp_model.RIGHT_LINES
    ]
    for channel, lines in enumerate(all_lines):
        for p1idx, p2idx in lines:
            point1 = tuple(list(kps[p1idx]))
            point2 = tuple(list(kps[p2idx]))

            if (point1[0] <= 0 and point1[1] <= 0) \
                    or (point2[0] <= 0 and point2[1] <= 0):
                continue

            draw.line([point1, point2], fill=COLORS[channel], width=4)

    # Draw Points
    point_size = 3
    all_points = [
        kp_model.LEFT_POINTS,
        kp_model.CENTER_POINTS,
        kp_model.RIGHT_POINTS,
    ]
    for channel, points in enumerate(all_points):
        for pidx in points:
            point = kps[pidx]
            if (point[0] <= 0 and point[1] <= 0):
                continue
            box = list(point - point_size) + list(point + point_size)

            draw.ellipse(box, fill=COLORS[channel])

    del draw

    return np.array(im)
