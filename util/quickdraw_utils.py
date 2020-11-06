import os
from io import BytesIO

import numpy as np
import svgwrite
from PIL import Image

try:
    import cairosvg
except:
    cairosvg = None


def stroke_three_format(big_stroke):
    """
    Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3.
    This is only for SCALE INVARIANT and UNCENTERED stroke-5 format.
    """
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def stroke_five_format(sketch, max_len):
    """
    Pad the batch to be stroke-5 bigger format as described in paper.
    This is only for SCALE INVARIANT and UNCENTERED stroke-3 format.
    """
    result = np.zeros((max_len + 1, 5), dtype=float)
    sketch_length = len(sketch)

    result[0:sketch_length, 0:2] = sketch[:, 0:2]
    result[0:sketch_length, 3] = sketch[:, 2]
    result[0:sketch_length, 2] = 1 - result[0:sketch_length, 3]
    result[sketch_length:, 4] = 1
    # put in the first token, as described in sketch-rnn methodology
    # result[1:, :] = result[:-1, :]
    result[0, :] = np.array([0, 0, 1, 0, 0])

    return result


def stroke_three_format_centered(big_stroke):
    """
    Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3.
    Note: This is only for SCALED AND CENTERED stroke-5 format.
    """
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l-1, 3))
    result[:, 0:2] = big_stroke[1:l, 0:2]
    result[:, 2] = big_stroke[1:l, 3]
    return result


def stroke_five_format_centered(sketch, max_len):
    """
    Pad the batch to be stroke-5 bigger format as described in paper.
    This is only for SCALED AND CENTERED stroke-3 format
    """
    result = np.zeros((max_len + 2, 5), dtype=float)
    sketch_length = len(sketch)

    result[0:sketch_length, 0:2] = sketch[:, 0:2]
    result[0:sketch_length, 3] = sketch[:, 2]
    result[0:sketch_length, 2] = 1 - result[0:sketch_length, 3]
    result[sketch_length:, 4] = 1
    # put in the first token, as described in sketch-rnn methodology
    result[1:, :] = result[:-1, :]
    result[0, :] = np.array([0, 0, 0, 1, 0])

    return result

def scale_and_center_stroke_three(sketch, png_dimensions, padding):
    """
    Modifies parameters of a stroke-3 format sketch such that it is maximized in size and centered in an image of png_dimensions
    and provided padding.
    :param sketch:
    :param png_dimensions:
    :param padding:
    :return:
    """
    min_x, max_x, min_y, max_y = _get_bounds(sketch)
    try:
        x_scale = (png_dimensions[0] - padding) / (max_x - min_x)
    except:
        x_scale = float('inf')
    try:
        y_scale = (png_dimensions[1] - padding) / (max_y - min_y)
    except:
        y_scale = float('inf')
    scale = min(x_scale, y_scale)

    sketch[:, 0:2] *= scale
    sketch[0, 0:2] += np.array([(png_dimensions[0] / 2) - ((max_x + min_x) / 2)*scale, (png_dimensions[1] / 2) - ((max_y + min_y) / 2)*scale])
    return sketch

def rasterize(sketch, png_dimensions):
    """
    Renders sketch as a rasterized image.
    :param sketch:
    :param png_dimensions:
    :return:
    """
    drawing_bytestring = _get_svg_string(sketch, png_dimensions)

    png_image = Image.open(BytesIO(cairosvg.svg2png(bytestring=drawing_bytestring, scale=1.0)))

    padded_image = pad_image(png_image, png_dimensions)

    return padded_image

def color_rasterize(sketches, png_dimensions, stroke_width=1):
    """
    Renders sketch as a rasterized image with a rotating sequence of colors for each stroke.
    :param sketches:
    :param png_dimensions:
    :param stroke_width:
    :return:
    """
    drawing_bytestring = _get_colored_svg_string(sketches, png_dimensions, stroke_width)

    png_image = Image.open(BytesIO(cairosvg.svg2png(bytestring=drawing_bytestring, scale=1.0)))
    padded_image = pad_image(png_image, png_dimensions)
    return padded_image

def _get_colored_svg_string(sketches, png_dimensions, stroke_width):
    dims = png_dimensions
    colors = ['black', 'red', 'blue', 'green', 'orange', 'purple']
    command = "m"

    dwg = svgwrite.Drawing(size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    for idx, sketch in enumerate(sketches):
        color = colors[idx % len(colors)]

        start_x, start_y, lift_pen = sketch[0, 0:3]
        p = "M%s, %s " % (start_x, start_y)
        for i in range(1, len(sketch)):
            if lift_pen == 1:
                command = "m"
            elif lift_pen == 0:
                command = "l"
            x = float(sketch[i, 0])
            y = float(sketch[i, 1])
            lift_pen = sketch[i, 2]
            p += command + str(x) + ", " + str(y) + " "
        dwg.add(dwg.path(p).stroke(color, stroke_width).fill("none"))

    return dwg.tostring()


def _get_svg_string(sketch, png_dimensions):
    dims = png_dimensions
    stroke_width = 1
    color = "black"
    command = "m"

    dwg = svgwrite.Drawing(size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    start_x, start_y = sketch[0, 0:2]
    lift_pen = sketch[0, 2]
    p = "M%s, %s " % (start_x, start_y)
    for i in range(1, len(sketch)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(sketch[i, 0])
        y = float(sketch[i, 1])
        lift_pen = sketch[i, 2]
        p += command + str(x) + ", " + str(y) + " "

    dwg.add(dwg.path(p).stroke(color, stroke_width).fill("none"))

    return dwg.tostring()


def scale_and_rasterize(sketch, png_dimensions, stroke_width=1):
    """Converts unscaled Stroke-3 SVG image to PNG."""
    padding = round(min(png_dimensions) / 10.) * 2
    svg_dimensions, drawing_bytestring = _scale_and_get_svg_string(sketch, png_dimensions, padding=padding, stroke_width=stroke_width)

    svg_width, svg_height = svg_dimensions
    png_width, png_height = png_dimensions
    x_scale = (png_width) / svg_width
    y_scale = (png_height) / svg_height

    png_image = Image.open(BytesIO(cairosvg.svg2png(bytestring=drawing_bytestring, scale=min(x_scale, y_scale))))

    padded_image = pad_image(png_image, png_dimensions)

    return padded_image


def _scale_and_get_svg_string(svg, png_dimensions, padding, stroke_width=1):
    """Retrieves SVG native dimension and bytestring."""

    min_x, max_x, min_y, max_y = _get_bounds(svg)
    try:
        x_scale = (png_dimensions[0] - padding) / (max_x - min_x)
    except:
        x_scale = float('inf')
    try:
        y_scale = (png_dimensions[1] - padding) / (max_y - min_y)
    except:
        y_scale = float('inf')
    scale = min(x_scale, y_scale)
    dims = png_dimensions
    lift_pen = 1
    color = "black"
    command = "m"

    dwg = svgwrite.Drawing(size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    start_x = (png_dimensions[0] / 2) - ((max_x + min_x) / 2) * scale
    start_y = (png_dimensions[1] / 2) - ((max_y + min_y) / 2) * scale
    p = "M%s, %s " % (start_x, start_y)
    for i in range(len(svg)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(svg[i, 0]) * scale
        y = float(svg[i, 1]) * scale
        lift_pen = svg[i, 2]
        p += command + str(x) + ", " + str(y) + " "

    dwg.add(dwg.path(p).stroke(color, stroke_width).fill("none"))

    return dims, dwg.tostring()


def _get_bounds(svg):
    """Return bounds of data."""

    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    abs_x, abs_y = 0, 0

    for i in range(len(svg)):
        x, y = float(svg[i, 0]), float(svg[i, 1])
        abs_x += x
        abs_y += y
        min_x, min_y, max_x, max_y = min(min_x, abs_x), min(min_y, abs_y), max(max_x, abs_x), max(max_y, abs_y)

    return min_x, max_x, min_y, max_y


def pad_image(png, png_dimensions):
    """
    Pads png to ensure it is the correct dimensions after rasterization
    :param png:
    :param png_dimensions:
    :return:
    """
    png_curr_w = png.width
    png_curr_h = png.height

    padded_png = np.zeros(shape=[png_dimensions[1], png_dimensions[0], 3], dtype=np.uint8)
    padded_png.fill(255)

    if png_curr_w > png_curr_h:
        pad = int(round((png_curr_w - png_curr_h) / 2))
        padded_png[pad: pad + png_curr_h, :png_curr_w] = np.array(png, dtype=np.uint8)
    else:
        pad = int(round((png_curr_h - png_curr_w) / 2))
        padded_png[:png_curr_h, pad: pad + png_curr_w] = np.array(png, dtype=np.uint8)

    return padded_png


def get_normalizing_scale_factor(sketches):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    sketches = list(map(lambda x: x[0], sketches))
    sketches = np.concatenate(sketches, axis=0)[:, 0:2]
    sketches = sketches.flatten()
    return np.std(sketches)


def quickdraw_process(batch_data, max_seq_len, png_dims, save_path, normalizing_scale_factor,
                      gap_limit=1000, flip_x=False, flip_y=False):
    """Preprocess sketches to drop large gaps, produce sketch-5 format and generate rasterized images."""
    stroke_five_sketches = []
    rasterized_images = []
    class_names = []

    padding = round(min(png_dims)/10.) * 2

    for sketch, class_name in batch_data:
        # cast and scale
        try:
            sketch = np.array(sketch, dtype=np.float32)

            # removes large gaps from the data
            stroke_three = np.maximum(np.minimum(sketch, gap_limit), -gap_limit)

            # Centered and normalized strokes for training sequence
            stroke_three_normalized = stroke_three
            stroke_three_normalized[:, 0:2] /= normalizing_scale_factor
            stroke_five_sketch = stroke_five_format(stroke_three_normalized, max_seq_len)

            # Centered and pixel-scaled for rasterization to produce input image
            stroke_three_scaled_and_centered = scale_and_center_stroke_three(np.copy(stroke_three), png_dims, padding)

            if flip_x:
                stroke_five_sketch[:, 0] = -stroke_five_sketch[:, 0]
            if flip_y:
                stroke_five_sketch[:, 1] = -stroke_five_sketch[:, 1]

            raster_image = rasterize(stroke_three_scaled_and_centered, png_dims)
        except:
            continue
        stroke_five_sketches.append(stroke_five_sketch)
        rasterized_images.append(raster_image)
        class_names.append(class_name)

    # Image.fromarray(rasterized_images[0].astype(np.uint8)).save("rastertest.png")
    np.savez(save_path,
             stroke_five_sketches=np.array(stroke_five_sketches, dtype=np.float32),
             rasterized_images=np.array(rasterized_images, dtype=np.float32),
             class_name=np.array(class_names))

    return 1
