import numpy as np

from rdp import rdp


def string_to_strokes(stroke_str, flip=True):
    """
    Convert Omniglot string-format data to an absolute stroke-3 format.
    :param stroke_str:
    :param epsilon:
    :param flip: flip (only use if omniglot)
    :return:
    """
    strokes_raw = [stroke.strip() for stroke in stroke_str.split('\n')]
    strokes = []

    for idx in range(len(strokes_raw)):
        stroke_raw = strokes_raw[idx]
        if stroke_raw == "START":
            curr_stroke = []
        elif stroke_raw == "BREAK":
            # next_stroke = np.fromstring(strokes_raw[min(idx+1, len(strokes_raw))], dtype=float, sep=',')[:-1]
            # prev_stroke = np.fromstring(strokes_raw[idx-1], dtype=float, sep=',')[:-1]
            # if len(next_stroke) > 0 and (np.linalg.norm(next_stroke - prev_stroke) < epsilon).all():
            #     continue
            if not curr_stroke:
                continue

            curr_stroke = np.array(curr_stroke)

            if flip:
                curr_stroke[:, 1] = -curr_stroke[:, 1]

            strokes.append(curr_stroke)
            curr_stroke = []
        else:
            xy = np.fromstring(stroke_raw, dtype=float, sep=',')[:2]
            curr_stroke.append(xy)

    return strokes


def apply_rdp(strokes, epsilon=1.5):
    """
    Apply Ramer-Douglas-Peucker algorithm to an absolute position stroke-3 format.
    :param strokes:
    :param epsilon:
    :return:
    """
    return [rdp(stroke, epsilon=epsilon) for stroke in strokes]


def strokes_to_stroke_three(strokes):
    """
    Convert strokes of constant point position to offset stroke-3 format for drawer
    :param strokes:
    :return:
    """

    stroke_three = []
    last_pt = np.array([0, 0])
    for stroke_idx in range(len(strokes)):
        stroke = strokes[stroke_idx]
        for point_idx in range(len(stroke)):
            if point_idx == len(stroke) - 1:
                if len(stroke) == 1:
                    new_pt = np.concatenate((stroke[point_idx] - last_pt, [0]))
                else:
                    new_pt = np.concatenate((stroke[point_idx] - last_pt, [1]))
            else:
                new_pt = np.concatenate((stroke[point_idx] - last_pt, [0]))
            stroke_three.append(new_pt)
            last_pt = stroke[point_idx]

        # Agument the dot if the stroke is only a single
        if len(stroke) == 1:
            stroke_three.append(np.array([0.5, 0.5, 0]))
            stroke_three.append(np.array([-0.5, -0.5, 1]))

    return np.array(stroke_three)

