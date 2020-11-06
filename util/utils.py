import os

import numpy as np
import tensorflow as tf

from multiprocessing import Process, Queue

from PIL import Image

from .quickdraw_utils import stroke_three_format, scale_and_rasterize


def process_write_out(write_fn, fn_args, max_queue_size=5000):
    """
    Begins a parallelized writer that runs write_fn. Need to disable cuda devices when beginning Process.
    :param write_fn:
    :param fn_args:
    :param max_queue_size:
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    write_queue = Queue(maxsize=max_queue_size)
    process = Process(target=write_fn, args=fn_args + (write_queue,))
    process.start()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    return process, write_queue


def gaussFilter(fx, fy, sigma):
    """
    Creates a filter with gaussian blurring
    :param fx:
    :param fy:
    :param sigma:
    :return:
    """
    x = tf.range(-int(fx / 2), int(fx / 2) + 1, 1)
    y = x
    Y, X = tf.meshgrid(x, y)

    sigma = -2 * (sigma ** 2)
    z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
    k = 2 * tf.exp(tf.divide(z, sigma))
    k = tf.divide(k, tf.reduce_sum(k))
    return k

def gaussian_blur(image, filtersize, sigma):
    """
    Applies gaussian blur to image based on provided parameters
    :param image:
    :param filtersize:
    :param sigma:
    :return:
    """
    n_channels = image.shape[-1]

    fx, fy = filtersize[0], filtersize[1]
    filt = gaussFilter(fx, fy, sigma)
    filt = tf.stack([filt] * n_channels, axis=2)
    filt = tf.expand_dims(filt, 3)

    padded_image = tf.pad(image, [[0, 0], [fx, fx], [fy, fy], [0, 0]], constant_values=0.0)

    res = tf.nn.depthwise_conv2d(padded_image, filt, strides=[1, 1, 1, 1], padding="SAME")
    return res[:, fx:-fx, fy:-fy, :]

def bilinear_interpolate_4_vectors(vectors, interps=10):
    """
    Bilinear interplation of 4 vectors in a 2D square
    :param vectors: 
    :param interps: 
    :return: 
    """

    #build bilinear interpolation weights
    arr = np.zeros((interps, interps, 4))
    interval = 1.0/(interps-1)

    for i in range(interps):
        for j in range(interps):
            x_pt, y_pt = i * interval, j * interval

            arr[j, i, :] = np.array([(1-x_pt) * (1-y_pt), x_pt * (1-y_pt), (1-x_pt) * y_pt, x_pt * y_pt])

    return np.einsum('ija,ak->ijk', arr, vectors)

def interpolate(model, test_dataset, result_name, steps=1, generation_length=64, interps=20):
    """
    Used to generate 2D interpolated embeddings.
    :param model: 
    :param test_dataset: 
    :param result_name: 
    :param steps: 
    :param generation_length: 
    :param interps: 
    :return: 
    """
    sampling_dir = os.path.join(model._sampling_dir, result_name)
    os.makedirs(sampling_dir)
    test_dataset, _ = test_dataset

    # Begin Writing Child-Process
    for step, entry in enumerate(test_dataset):
        if step == steps:
            break

        if len(entry) == 2:
            x_image, class_names = entry
        else:
            y_sketch_gt, y_sketch_teacher, x_image, class_names = entry[0:4]

        z, _, _ = model.embed(x_image, training=False)
        for idx in range(0, z.shape[0], 4):
            embeddings = z[idx: idx+4].numpy()
            classes = class_names[idx: idx+4].numpy()
            interpolated_embeddings = bilinear_interpolate_4_vectors(embeddings, interps=interps)

            flattened_embeddings = np.reshape(interpolated_embeddings, (-1, z.shape[-1])).astype(np.float32)
            _, flattened_strokes = model.decode(flattened_embeddings, training=False, generation_length=generation_length).numpy()

            flattened_images = []
            for strokes in flattened_strokes:
                stroke_three = stroke_three_format(strokes)
                flattened_images.append(scale_and_rasterize(stroke_three, (28, 28), 1).astype('uint8'))
            flattened_images = np.array(flattened_images, dtype=np.uint8)

            interpolated_images = np.reshape(flattened_images, list(interpolated_embeddings.shape[:2]) + list(flattened_images.shape[1:]))
            image_rows = []
            for row in interpolated_images:
                concat_row = np.concatenate(row, axis=1)
                image_rows.append(concat_row)

            np_image = np.concatenate(image_rows, axis=0)
            Image.fromarray(np_image).save(os.path.join(sampling_dir, "{}-{}_{}_{}_{}.png".format(idx//4, *classes)))

