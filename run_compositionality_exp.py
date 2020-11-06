import os
import traceback

import numpy as np
import cv2
import sklearn
import umap
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from PIL import Image
from absl import app, flags, logging
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import models
import configs
import datasets
from models import ClassifierModel, DrawerModel, VAE
from util import HParams, scale_and_rasterize, stroke_three_format, scale_and_center_stroke_three, rasterize
from util import log_flags, log_hparams

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = flags.FLAGS

flags.DEFINE_string("dir", "/h/wangale/project/few-shot-sketch", "Project directory")
flags.DEFINE_string("data_dir", "/h/wangale/data", "Data directory")

flags.DEFINE_string("id", "comptest", "training_id")
flags.DEFINE_string("logfile", "comptest", "Logfile name")

flags.DEFINE_string("drawer_id", None, "Drawing model id")
flags.DEFINE_string("drawer_model", None, "Drawer model")
flags.DEFINE_string("drawer_cfgset", None, "Configuration set for the drawer model")
flags.DEFINE_string("drawer_cfgs", "", "Custom configuration for the drawer model configs")

flags.DEFINE_string("vae_id", None, "VAE model ID")
flags.DEFINE_string("vae_model", None, "VAE model")
flags.DEFINE_string("vae_cfgset", None, "Configuration set for the vae model")
flags.DEFINE_string("vae_cfgs", "", "Custom configuration for the vae model configs")

flags.DEFINE_integer("random_seed", 1, "Random seed")

flags.DEFINE_bool("conceptual_composition", True, "Add/sub embeddings.")

flags.DEFINE_bool("relation_count", True, "Count test")
flags.DEFINE_bool("relation_count_toy", True, "Count with toy examples")
flags.DEFINE_bool("relation_orient", True, "Orientation test")
flags.DEFINE_bool("relation_inout", True, "In Out test")
flags.DEFINE_bool("relation_four", True, "Four compose test")
flags.DEFINE_bool("relation_count_readout", True, "Count readout")
flags.DEFINE_bool("relation_four_readout", True, "Four discrete readout test")
flags.DEFINE_bool("relation_nest_readout", True, "Four discrete readout test")

flags.DEFINE_bool("latent_distance", True, "Distance test")
flags.DEFINE_bool("latent_angle", True, "Angle test")
flags.DEFINE_bool("latent_size", True, "Size test")
flags.DEFINE_bool("latent_angle_readout", True, "Angle linear readout test")
flags.DEFINE_bool("latent_distance_readout", True, "Distance linear readout test")
flags.DEFINE_bool('latent_size_readout', True, "Size linear readout test")

flags.DEFINE_bool("n_interpolate", False, "Interpolate between 2,3,4 objects per image")
flags.DEFINE_bool("four_rotate", False, "rotate four placed values")


def conceptual_composition(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "addembed")
    ex_per_class = 50
    os.makedirs(folder, exist_ok=True)

    snowman_config: HParams = configs.get_config("quickdraw")().parse("split=snowman,shuffle=False,batch_size={}".format(ex_per_class))
    snowman_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, snowman_config)
    snowman_dataset = snowman_dataset_proto.load(repeat=False)[0]

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(ex_per_class))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(ex_per_class))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    television_config = configs.get_config("quickdraw")().parse("split=television,shuffle=False,batch_size={}".format(ex_per_class))
    television_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, television_config)
    television_dataset = television_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]
    snowman_batch, snowman_names = next(snowman_dataset.__iter__())[2:4]
    square_batch, square_names = next(square_dataset.__iter__())[2:4]
    television_batch, television_names = next(television_dataset.__iter__())[2:4]

    for model in embedding_models:
        circle_embed = model.embed(circle_batch, training=False)[0]
        snowman_embed = model.embed(snowman_batch, training=False)[0]
        square_embed = model.embed(square_batch, training=False)[0]
        television_embed = model.embed(television_batch, training=False)[0]

        for i, embeds in enumerate([(snowman_embed, circle_embed, square_embed,
                                     snowman_batch, circle_batch, square_batch), (television_embed, square_embed, circle_embed,
                                                                                  television_batch, square_batch, circle_batch)]):
            os.makedirs(os.path.join(folder, str(i), model.__class__.__name__))
            a, b, c, a_im, b_im, c_im = embeds
            new_embed = a - b + c

            if "Drawer" in model.__class__.__name__:
                decodes = model.decode(new_embed, training=False, generation_length=64)[1]
                lst = []
                for decode in decodes:
                    lst.append(scale_and_rasterize(stroke_three_format(decode), png_dimensions=(28, 28), stroke_width=1))
                gen_image = np.array(lst).astype(np.float32)
            elif "VAE" in model.__class__.__name__:
                gen_image = tf.image.resize(model.decode(new_embed, training=False), (28, 28)) * 255.0
            else:
                logging.info("Error, wrong embedding model")

            inter_embed = a - b

            if "Drawer" in model.__class__.__name__:
                decodes = model.decode(inter_embed, training=False, generation_length=64)[1]
                lst = []
                for decode in decodes:
                    lst.append(scale_and_rasterize(stroke_three_format(decode), png_dimensions=(28, 28), stroke_width=1))
                gen_inter_image = np.array(lst).astype(np.float32)
            elif "VAE" in model.__class__.__name__:
                gen_inter_image = tf.image.resize(model.decode(inter_embed, training=False), (28, 28)) * 255.0
            else:
                logging.info("Error, wrong embedding model")

            for k in range(ex_per_class):
                np_img = np.concatenate((a_im[k], b_im[k], c_im[k], gen_image[k]), axis=1).astype(np.uint8)
                Image.fromarray(np_img).save(os.path.join(folder, str(i), model.__class__.__name__, '{}.png'.format(k)))

                for j, np_arr in enumerate((a_im[k], b_im[k], c_im[k], gen_image[k])):
                    os.makedirs(os.path.join(folder, str(i), str(j), model.__class__.__name__), exist_ok=True)
                    Image.fromarray(np.array(np_arr).astype(np.uint8)).save(os.path.join(folder, str(i), str(j), model.__class__.__name__, '{}.png'.format(k)))

def relation_count(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "count")
    ex_per_class = 20
    os.makedirs(folder, exist_ok=True)

    snowman_config: HParams = configs.get_config("quickdraw")().parse("split=snowman,shuffle=False,batch_size={}".format(ex_per_class))
    snowman_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, snowman_config)
    snowman_dataset = snowman_dataset_proto.load(repeat=False)[0]

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(ex_per_class))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(ex_per_class))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    television_config = configs.get_config("quickdraw")().parse("split=television,shuffle=False,batch_size={}".format(ex_per_class))
    television_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, television_config)
    television_dataset = television_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]
    snowman_batch, snowman_names = next(snowman_dataset.__iter__())[2:4]
    square_batch, square_names = next(square_dataset.__iter__())[2:4]
    television_batch, television_names = next(television_dataset.__iter__())[2:4]
    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        circle_embed = model.embed(circle_batch, training=False)[0]
        snowman_embed = model.embed(snowman_batch, training=False)[0]
        square_embed = model.embed(square_batch, training=False)[0]
        television_embed = model.embed(television_batch, training=False)[0]

        n_class = 4
        x = tf.concat((circle_embed, snowman_embed, square_embed, television_embed), axis=0)
        y_image = tf.concat((circle_batch, snowman_batch, square_batch, television_batch), axis=0)

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1-tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image/255.0, alphas), axis=-1).numpy()

        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]]
        for k, col_array in enumerate(colors):
            curr_imgs = y_image[k * ex_per_class: (k+1) * ex_per_class, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.08), axis=-1, keepdims=True)
            y_image[k * ex_per_class: (k + 1) * ex_per_class, :, :, :3] = black_mask * col_array

        for k in range(4):
            Image.fromarray((y_image[k * ex_per_class] * 255.0).astype('uint8')).save(os.path.join(folder, "legend-{}.png".format(k)))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "compositionality.png"))

def relation_count_toy(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "count_toy")
    ex_per_class = 14
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(ex_per_class))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(ex_per_class))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]
    square_batch, square_names = next(square_dataset.__iter__())[2:4]

    circle_batch, square_batch = circle_batch.numpy(), square_batch.numpy()

    composite_batch = np.ones((circle_batch.shape[0] * 4, *circle_batch.shape[1:])) * 255.0

    for idx in range(circle_batch.shape[0]):
        circle_orig, square_orig = circle_batch[idx], square_batch[idx]
        circle, square = cv2.resize(circle_orig, (14, 14)), cv2.resize(square_orig, (14, 14))

        composite_batch[idx * 4, 7:21, 7:21, :] = circle
        composite_batch[(idx * 4) + 1, 7:21, :, :] = np.concatenate((circle, circle), axis=1)
        composite_batch[(idx * 4) + 2, :, 14:, :] = np.concatenate((circle, circle), axis=0)
        composite_batch[(idx * 4) + 2, 7:21, :14, :] = circle
        place4((idx*4)+3, composite_batch, [circle, circle, circle, circle])

    composite_batch = np.array(composite_batch, dtype=np.float32)

    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch, training=False)[0]
        y_image = composite_batch

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1 - tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image / 255.0, alphas), axis=-1).numpy()

        # Set colors
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]]  # , [1, 0, 1], [0, 1, 1]]
        pre_idx = np.array([k for k in range(0, x.shape[0], len(colors))])
        for k, col_array in enumerate(colors):
            idx = pre_idx + k
            curr_imgs = y_image[idx][:, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.08), axis=-1, keepdims=True)
            y_image[idx, :, :, :3] = black_mask * col_array

        for k in range(4):
            Image.fromarray((y_image[k] * 255.0).astype('uint8')).save(os.path.join(folder, "legend-{}.png".format(k)))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "count_toy.png"))

def relation_orientation(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "orientation")
    ex_per_class = 10
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(ex_per_class))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(ex_per_class))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]
    square_batch, square_names = next(square_dataset.__iter__())[2:4]

    circle_batch, square_batch = circle_batch.numpy(), square_batch.numpy()

    composite_batch = np.ones((circle_batch.shape[0] * 4, *circle_batch.shape[1:])) * 255.0

    for idx in range(circle_batch.shape[0]):
        circle_orig, square_orig = circle_batch[idx], square_batch[idx]
        circle, square = cv2.resize(circle_orig, (14, 14)), cv2.resize(square_orig, (14, 14))

        composite_batch[idx*4, :, 7:21, :] = np.concatenate((circle, square), axis=0)
        composite_batch[(idx*4)+1, 7:21, :, :] = np.concatenate((circle, square), axis=1)
        composite_batch[(idx*4)+2, :, 7:21, :] = np.concatenate((square, circle), axis=0)
        composite_batch[(idx*4)+3, 7:21, :, :] = np.concatenate((square, circle), axis=1)

    composite_batch = np.array(composite_batch, dtype=np.float32)

    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch, training=False)[0]
        y_image = composite_batch

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1-tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image/255.0, alphas), axis=-1).numpy()

        # Set colors
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]]#, [1, 0, 1], [0, 1, 1]]
        pre_idx = np.array([k for k in range(0, x.shape[0], len(colors))])
        for k, col_array in enumerate(colors):
            idx = pre_idx + k
            curr_imgs = y_image[idx][:, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.08), axis=-1, keepdims=True)
            y_image[idx, :, :, :3] = black_mask * col_array

        for k in range(4):
            Image.fromarray((y_image[k] * 255.0).astype('uint8')).save(os.path.join(folder, "legend-{}.png".format(k)))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "orientation.png"))

def relation_inout(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "inout")
    ex_per_class = 35
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(ex_per_class))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(ex_per_class))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    circle_strokes, circle_batch, circle_names = next(circle_dataset.__iter__())[1:4]
    square_strokes, square_batch, square_names = next(square_dataset.__iter__())[1:4]

    circle_batch, square_batch = circle_batch.numpy(), square_batch.numpy()

    composite_batch = np.ones((circle_batch.shape[0] * 2, *circle_batch.shape[1:])) * 255.0
    png_dims = circle_batch[0].shape[-3:-1]

    for idx in range(circle_batch.shape[0]):
        circle_orig, square_orig = circle_batch[idx], square_batch[idx]
        circle, square = cv2.resize(circle_orig, (14, 14)), cv2.resize(square_orig, (14, 14))

        # circle_big, square_big = cv2.resize(circle_orig, (32, 32)), cv2.resize(square_orig, (32, 32))

        stroke_five = circle_strokes[idx]
        stroke_three = stroke_three_format(stroke_five)
        scaled_and_centered_stroke_three = scale_and_center_stroke_three(stroke_three, circle_batch[0].shape[-3:-1], 2)
        rasterized_image = rasterize(scaled_and_centered_stroke_three, png_dims)
        big_circle = rasterized_image
        stroke_five = square_strokes[idx]
        stroke_three = stroke_three_format(stroke_five)
        scaled_and_centered_stroke_three = scale_and_center_stroke_three(stroke_three, circle_batch[0].shape[-3:-1], 2)
        rasterized_image = rasterize(scaled_and_centered_stroke_three, png_dims)
        big_square = rasterized_image

        composite_batch[(idx*2), :, :, :] = big_square
        composite_batch[(idx*2), 8:20, 8:20, :] = circle[1:13, 1:13, :]

        composite_batch[(idx*2)+1, :, :, :] = big_circle
        composite_batch[(idx*2)+1, 8:20, 8:20, :] = square[1:13, 1:13, :]

    composite_batch = np.array(composite_batch, dtype=np.float32)

    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch, training=False)[0]
        y_image = composite_batch

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1-tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image/255.0, alphas), axis=-1).numpy()

        # Set colors
        colors = [[1, 0, 0], [0, 1, 0]]  # , [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        pre_idx = np.array([k for k in range(0, x.shape[0], len(colors))])
        for k, col_array in enumerate(colors):
            idx = pre_idx + k
            curr_imgs = y_image[idx][:, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.08), axis=-1, keepdims=True)
            y_image[idx, :, :, :3] = black_mask * col_array

        for k in range(2):
            Image.fromarray((y_image[k] * 255.0).astype('uint8')).save(os.path.join(folder, "legend-{}.png".format(k)))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "inout.png"))

def relation_four(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "four")
    ex_per_class = 15
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(ex_per_class))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(ex_per_class))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]
    square_batch, square_names = next(square_dataset.__iter__())[2:4]

    circle_batch, square_batch = circle_batch.numpy(), square_batch.numpy()

    composite_batch = np.ones((circle_batch.shape[0] * 4, *circle_batch.shape[1:])) * 255.0

    for idx in range(circle_batch.shape[0]):
        circle_orig, square_orig = circle_batch[idx], square_batch[idx]
        circle, square = cv2.resize(circle_orig, (14, 14)), cv2.resize(square_orig, (14, 14))

        place4((idx*4), composite_batch, [circle, circle, square, square])
        place4((idx*4)+1, composite_batch, [square, circle, circle, square])
        place4((idx*4)+2, composite_batch, [square, square, circle, circle])
        place4((idx*4)+3, composite_batch, [circle, square, square, circle])

    composite_batch = np.array(composite_batch, dtype=np.float32)

    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch, training=False)[0]
        y_image = composite_batch

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1-tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image/255.0, alphas), axis=-1).numpy()

        # Set colors
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]] #, [1, 0, 1], [0, 1, 1]]
        pre_idx = np.array([k for k in range(0, x.shape[0], len(colors))])
        for k, col_array in enumerate(colors):
            idx = pre_idx + k
            curr_imgs = y_image[idx][:, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.08), axis=-1, keepdims=True)
            y_image[idx, :, :, :3] = black_mask * col_array

        for k in range(4):
            Image.fromarray((y_image[k] * 255.0).astype('uint8')).save(os.path.join(folder, "legend-{}.png".format(k)))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "four.png"))

def relation_four_readout(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "four_readout")
    batch_size = 300
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(batch_size))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(batch_size))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]
    square_batch, square_names = next(square_dataset.__iter__())[2:4]

    circle_batch, square_batch = circle_batch.numpy(), square_batch.numpy()

    composite_batch = np.ones((batch_size, *circle_batch.shape[1:])) * 255.0
    y_class = np.zeros((batch_size, ))

    for idx in range(batch_size):
        circle_orig, square_orig = circle_batch[idx], square_batch[idx]
        circle, square = cv2.resize(circle_orig, (14, 14)), cv2.resize(square_orig, (14, 14))

        class_label = np.random.randint(0, 4)
        if class_label == 0:
            place4(idx, composite_batch, [circle, circle, square, square])
            # y_class[idx] = np.array([1, 0, 0, 0])
        elif class_label == 1:
            place4(idx, composite_batch, [square, circle, circle, square])
            # y_class[idx] = np.array([0, 1, 0, 0])
        elif class_label == 2:
            place4(idx, composite_batch, [square, square, circle, circle])
            # y_class[idx] = np.array([0, 0, 1, 0])
        elif class_label == 3:
            place4(idx, composite_batch, [circle, square, square, circle])
            # y_class[idx] = np.array([0, 0, 0, 1])
        y_class[idx] = class_label

    composite_batch = np.array(composite_batch, dtype=np.float32)

    reg_model = linear_model.LogisticRegression()

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch.astype(np.float32), training=False)[0]

        reg_model.fit(x[:100], y_class[:100])
        logging.info("%s, %f", model.__class__.__name__, reg_model.score(x[100:], y_class[100:]))

def relation_nest_readout(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "nest_readout")
    batch_size = 300
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(batch_size))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(batch_size))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    circle_strokes, circle_batch, circle_names = next(circle_dataset.__iter__())[1:4]
    square_strokes, square_batch, square_names = next(square_dataset.__iter__())[1:4]

    circle_batch, square_batch = circle_batch.numpy(), square_batch.numpy()

    composite_batch = np.ones((batch_size, *circle_batch.shape[1:])) * 255.0
    y_class = np.zeros((batch_size, ))
    png_dims = circle_batch[0].shape[-3:-1]

    for idx in range(batch_size):
        circle_orig, square_orig = circle_batch[idx], square_batch[idx]
        circle, square = cv2.resize(circle_orig, (14, 14)), cv2.resize(square_orig, (14, 14))

        stroke_five = circle_strokes[idx]
        stroke_three = stroke_three_format(stroke_five)
        scaled_and_centered_stroke_three = scale_and_center_stroke_three(stroke_three, circle_batch[0].shape[-3:-1], 2)
        rasterized_image = rasterize(scaled_and_centered_stroke_three, png_dims)
        big_circle = rasterized_image
        stroke_five = square_strokes[idx]
        stroke_three = stroke_three_format(stroke_five)
        scaled_and_centered_stroke_three = scale_and_center_stroke_three(stroke_three, circle_batch[0].shape[-3:-1], 2)
        rasterized_image = rasterize(scaled_and_centered_stroke_three, png_dims)
        big_square = rasterized_image

        class_label = np.random.randint(0, 2)

        if class_label == 0:
            composite_batch[idx, :, :, :] = big_circle
            composite_batch[idx, 8:20, 8:20, :] = square[1:13, 1:13, :]
        elif class_label == 1:
            composite_batch[idx, :, :, :] = big_square
            composite_batch[idx, 8:20, 8:20, :] = circle[1:13, 1:13, :]

        y_class[idx] = class_label

    composite_batch = np.array(composite_batch, dtype=np.float32)

    for i, model in enumerate(embedding_models):
        reg_model = linear_model.LogisticRegression()
        x = model.embed(composite_batch.astype(np.float32), training=False)[0]

        reg_model.fit(x[:100], y_class[:100])
        logging.info("%s, %f", model.__class__.__name__, reg_model.score(x[100:], y_class[100:]))

def latent_distance_cts(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "distance")
    batch_size = 300
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(batch_size))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]

    circle_batch = circle_batch.numpy()

    composite_batch = np.ones((circle_batch.shape[0], *circle_batch.shape[1:])) * 255.0
    y_dist = np.zeros((batch_size, 1))

    for idx in range(batch_size):
        circle_orig = circle_batch[idx]
        circle = cv2.resize(circle_orig, (12, 12))
        # circle -= (circle < 200).astype(np.float32) * 120
        # circle = np.maximum(np.zeros(circle.shape), circle)

        distance = np.random.randint(0, 7)

        y_dist[idx] = distance
        composite_batch[idx, 8:20, :11, :] = circle[:, 1:]
        composite_batch[idx, 8:20, 16 + distance - 6:16 + distance + 6, :] = circle

    composite_batch = np.array(composite_batch, dtype=np.float32)

    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch, training=False)[0]
        y_image = composite_batch

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1 - tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image / 255.0, alphas), axis=-1).numpy()

        for idx in range(batch_size):
            dist = y_dist[idx]
            color = [dist/6.0, 0, 1-(dist/6.0)]
            curr_imgs = y_image[idx, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.08), axis=-1, keepdims=True)
            y_image[idx, :, :, :3] = black_mask * color

        Image.fromarray((y_image[np.argmax(y_dist)] * 255.0).astype('uint8')).save(os.path.join(folder, "legend-{}.png".format(np.argmax(y_dist))))
        Image.fromarray((y_image[np.argmin(y_dist)] * 255.0).astype('uint8')).save(
            os.path.join(folder, "legend-{}.png".format(np.argmin(y_dist))))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)

    for i in range(10):
        Image.fromarray((y_image[i] * 255.0).astype('uint8')).save(
            os.path.join(folder, "legend-{}.png".format(y_dist[i])))

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "distance.png"))

def latent_angle_cts(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "angle")

    batch_size = 400
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(batch_size))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]

    circle_batch = circle_batch.numpy()

    composite_batch = np.ones((circle_batch.shape[0], *circle_batch.shape[1:])) * 255.0
    y_angles = np.zeros((batch_size, 1))

    for idx in range(circle_batch.shape[0]):
        circle_orig = circle_batch[idx]
        circle = cv2.resize(circle_orig, (12, 12))

        angle = np.random.uniform(-0.52, 0.52)  # -30 to 30 degrees, the limit of our canvas
        x_offs, y_offs = np.rint(14 * np.cos(angle)).astype(np.int32), np.rint(14 * np.sin(angle)).astype(np.int32)

        composite_batch[idx, 8:20, :11, :] = circle[:, 1:]
        composite_batch[idx, 14 - y_offs - 6:14 - y_offs + 6, 7 + x_offs - 6:7 + x_offs + 6, :] = circle
        y_angles[idx, 0] = np.round(angle, decimals=2)

    composite_batch = np.array(composite_batch, dtype=np.float32)

    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch, training=False)[0]
        y_image = composite_batch

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1 - tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image / 255.0, alphas), axis=-1).numpy()

        # Set colors
        for idx in range(batch_size):
            angle = y_angles[idx] + 0.52
            color = [angle / 1.04, 0, 1 - (angle / 1.04)]
            curr_imgs = y_image[idx, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.08), axis=-1, keepdims=True)
            y_image[idx, :, :, :3] = black_mask * color

        Image.fromarray((y_image[np.argmax(y_angles)] * 255.0).astype('uint8')).save(
            os.path.join(folder, "legend-{}.png".format(np.argmax(y_angles))))
        Image.fromarray((y_image[np.argmin(y_angles)] * 255.0).astype('uint8')).save(
            os.path.join(folder, "legend-{}.png".format(np.argmin(y_angles))))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)

    for i in range(10):
        Image.fromarray((y_image[i] * 255.0).astype('uint8')).save(
            os.path.join(folder, "legend-{}.png".format(y_angles[i])))

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "angle.png"))

def latent_size_cts(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "size")
    batch_size = 500
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=True,batch_size={}".format(batch_size))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    iter = circle_dataset.__iter__()
    circle_strokes, circle_batch, circle_names = next(iter)[1:4]

    circle_batch = circle_batch.numpy()

    composite_batch = np.ones((circle_batch.shape[0], *circle_batch.shape[1:])) * 255.0
    y_size = np.zeros((batch_size, 1))

    png_dims = circle_batch[0].shape[-3:-1]

    for idx in range(batch_size):
        circle_orig = circle_batch[idx]

        size = np.random.randint(4, 22)

        stroke_five = circle_strokes[idx]
        stroke_three = stroke_three_format(stroke_five)
        scaled_and_centered_stroke_three = scale_and_center_stroke_three(stroke_three, png_dims, 28-size)
        rasterized_image = rasterize(scaled_and_centered_stroke_three, png_dims)
        circle = rasterized_image

        y_size[idx] = size
        composite_batch[idx] = circle

    composite_batch = np.array(composite_batch, dtype=np.float32)

    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch, training=False)[0]
        y_image = composite_batch

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1 - tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image / 255.0, alphas), axis=-1).numpy()

        for idx in range(batch_size):
            size = y_size[idx]
            color = [(size-4)/18, 0, 1-((size-4)/18)]
            curr_imgs = y_image[idx, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.12), axis=-1, keepdims=True)
            y_image[idx, :, :, :3] = black_mask * color

        Image.fromarray((y_image[np.argmax(y_size)] * 255.0).astype('uint8')).save(os.path.join(folder, "legend-{}.png".format(np.argmax(y_size))))
        Image.fromarray((y_image[np.argmin(y_size)] * 255.0).astype('uint8')).save(
            os.path.join(folder, "legend-{}.png".format(np.argmin(y_size))))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)

    for i in range(10):
        Image.fromarray((y_image[i] * 255.0).astype('uint8')).save(
            os.path.join(folder, "legend-{}.png".format(y_size[i])))

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "size.png"))

def latent_count_readout(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "count_readout")
    batch_size = 300
    os.makedirs(folder, exist_ok=True)

    snowman_config: HParams = configs.get_config("quickdraw")().parse("split=snowman,shuffle=True,batch_size={}".format(batch_size))
    snowman_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, snowman_config)
    snowman_dataset = snowman_dataset_proto.load(repeat=False)[0]

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=True,batch_size={}".format(batch_size))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=True,batch_size={}".format(batch_size))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    television_config = configs.get_config("quickdraw")().parse("split=television,shuffle=True,batch_size={}".format(batch_size))
    television_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, television_config)
    television_dataset = television_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]
    snowman_batch, snowman_names = next(snowman_dataset.__iter__())[2:4]
    square_batch, square_names = next(square_dataset.__iter__())[2:4]
    television_batch, television_names = next(television_dataset.__iter__())[2:4]

    circle_batch, square_batch = circle_batch.numpy(), square_batch.numpy()

    composite_batch = np.ones((batch_size, *circle_batch.shape[1:])) * 255.0
    y_class = np.zeros((batch_size, ))

    for idx in range(batch_size):
        circle_orig, square_orig = circle_batch[idx], square_batch[idx]
        snowman_orig, television_orig = snowman_batch[idx], television_batch[idx]

        class_label = np.random.randint(0, 4)
        if class_label == 0:
            composite_batch[idx] = circle_orig
        elif class_label == 1:
            composite_batch[idx] = square_orig
        elif class_label == 2:
            composite_batch[idx] = snowman_orig
        elif class_label == 3:
            composite_batch[idx] = television_orig
        y_class[idx] = class_label

    composite_batch = np.array(composite_batch, dtype=np.float32)

    for i, model in enumerate(embedding_models):
        reg_model = linear_model.Ridge()
        reg_model = MLPRegressor(max_iter=2000)
        x = model.embed(composite_batch.astype(np.float32), training=False)[0]

        reg_model.fit(x[:100], y_class[:100])
        logging.info("%s, %f", model.__class__.__name__, reg_model.score(x[100:], y_class[100:]))

def latent_angle_readout(embedding_models, clustering_methods, base_dir):
    logging.info("---Angle Readout---")
    folder = os.path.join(base_dir, "angle_readout", "angled")
    num_train_ex = 100
    num_test_ex = 2000

    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(num_train_ex + num_test_ex))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    circle_batch = next(circle_dataset.__iter__())[2].numpy()

    composed_image_batch = np.ones((num_train_ex + num_test_ex, *circle_batch.shape[1:])) * 255.0
    y_angles = np.zeros((num_train_ex + num_test_ex, 1))

    for i in range(num_train_ex + num_test_ex):
        circle_orig = circle_batch[i]
        circle = cv2.resize(circle_orig, (12, 12))

        composed_image_batch[i, 8:20, 1:13, :] = circle

        angle = np.random.uniform(-0.52, 0.52)  # -30 to 30 degrees, the limit of our canvas
        x_offs, y_offs = np.rint(14*np.cos(angle)).astype(np.int32), np.rint(14*np.sin(angle)).astype(np.int32)

        composed_image_batch[i, 14-y_offs-6:14-y_offs+6, 7+x_offs-6:7+x_offs+6, :] = circle
        y_angles[i, 0] = np.round(angle, decimals=2)

    for i, model in enumerate(embedding_models):
        lin_readout(model, composed_image_batch, y_angles, num_train_ex)

def latent_distance_readout(embedding_models, clustering_methods, base_dir):
    logging.info("---Distance Readout---")
    folder = os.path.join(base_dir, "distance_readout")
    num_train_ex = 100
    num_test_ex = 2000

    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(num_train_ex + num_test_ex))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    circle_batch = next(circle_dataset.__iter__())[2].numpy()

    composed_image_batch = np.ones((num_train_ex + num_test_ex, *circle_batch.shape[1:])) * 255.0
    y_distance = np.zeros((num_train_ex + num_test_ex, 1))

    for i in range(num_train_ex + num_test_ex):
        circle_orig = circle_batch[i]
        circle = cv2.resize(circle_orig, (12, 12))
        circle -= (circle < 200).astype(np.float32) * 120
        circle = np.maximum(np.zeros(circle.shape), circle)

        composed_image_batch[i, 8:20, :10, :] = circle[:, 2:]

        distance = np.random.uniform(0.0, 6)  # -30 to 30 degrees, the limit of our canvas

        composed_image_batch[i, 8:20, np.rint(16+distance-6).astype(np.int32):np.rint(16+distance+6).astype(np.int32), :] = circle
        y_distance[i, 0] = np.round(distance, decimals=0)

    for i, model in enumerate(embedding_models):
        lin_readout(model, composed_image_batch, y_distance, num_train_ex)

def latent_size_readout(embedding_models, clustering_methods, base_dir):
    logging.info("---Size Readout---")
    folder = os.path.join(base_dir, "size_readout")
    num_train_ex = 100
    num_test_ex = 2000

    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse(
        "split=circle,shuffle=False,batch_size={}".format(num_train_ex + num_test_ex))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    iter = circle_dataset.__iter__()
    circle_strokes, circle_batch, circle_names = next(iter)[1:4]
    circle_batch = circle_batch.numpy()

    composed_image_batch = np.ones((num_train_ex + num_test_ex, *circle_batch.shape[1:])) * 255.0
    y_size = np.zeros((num_train_ex + num_test_ex, 1))

    png_dims = circle_batch[0].shape[-3:-1]

    for idx in range(num_train_ex + num_test_ex):
        size = np.random.randint(4, 22)

        stroke_five = circle_strokes[idx]
        stroke_three = stroke_three_format(stroke_five)
        scaled_and_centered_stroke_three = scale_and_center_stroke_three(stroke_three, png_dims, 28-size)
        rasterized_image = rasterize(scaled_and_centered_stroke_three, png_dims)
        circle = rasterized_image

        y_size[idx] = size
        composed_image_batch[idx] = circle

    for i, model in enumerate(embedding_models):
        lin_readout(model, composed_image_batch, y_size, num_train_ex)

def n_interpolate(embedding_models, clustering_methods, base_dir):
    folder = os.path.join(base_dir, "n_interpolate")
    batch_size = 20
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(batch_size))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    circle_batch = next(circle_dataset.__iter__())[2].numpy()
    composed_image_batch = np.ones((batch_size * 3, *circle_batch.shape[1:])) * 255.0

    for i in range(batch_size):
        circle_orig = circle_batch[i]
        circle = cv2.resize(circle_orig, (14, 14))
        shp = circle_orig.shape

        composed_image_batch[3*i, 7:21, 7:21, :] = circle
        composed_image_batch[3*i] -= (composed_image_batch[3*i] < 230).astype(np.float32) * 100
        composed_image_batch[3 * i] -= (composed_image_batch[3 * i] < 230).astype(np.float32) * 100
        composed_image_batch[3*i] = np.maximum(np.zeros(shp), composed_image_batch[3*i])

        composed_image_batch[3*i+1, 1:15, 7:21, :] = circle
        composed_image_batch[3*i+1, 13:27, 7:21, :] = circle
        composed_image_batch[3*i+1] -= (composed_image_batch[3*i+1] < 200).astype(np.float32)* 150
        composed_image_batch[3*i+1] = np.maximum(np.zeros(shp), composed_image_batch[3*i+1])

        composed_image_batch[3*i+2, 1:15, 7:21, :] = circle
        composed_image_batch[3*i+2, 13:27, 1:15, :] = circle
        composed_image_batch[3*i+2, 13:27, 13:27, :] = circle
        composed_image_batch[3*i+2] -= (composed_image_batch[3 * i + 2] < 200).astype(np.float32)* 150
        composed_image_batch[3*i+2] = np.maximum(np.zeros(shp), composed_image_batch[3 * i + 2])

    def interpolate3(x, y, z):
        res = np.zeros((21, *x.shape))
        for idx, w in enumerate(np.linspace(0.1, 0.9, 9)):
            res[idx + 1] = ((1.-w)*x) + (w*y)

        for idx, w in enumerate(np.linspace(0.1, 0.9, 9)):
            res[idx + 11] = ((1.-w)*y) + (w*z)

        res[0] = x
        res[10] = y
        res[20] = z

        return res

    for i, model in enumerate(embedding_models):
        x = model.embed(composed_image_batch.astype(np.float32), training=False)[0]
        for j in range(batch_size):
            two, three, four = x[3*j], x[3*j+1], x[3*j+2]

            decode_batch = interpolate3(two, three, four)

            if "Drawer" in model.__class__.__name__:
                decodes = model.decode(decode_batch.astype(np.float32), training=False, generation_length=64)[1]
                lst = []
                for decode in decodes:
                    lst.append(scale_and_rasterize(stroke_three_format(decode), png_dimensions=(28, 28), stroke_width=1))
                gen_inter_image = np.array(lst).astype(np.float)
                name = "drawer_{}.png".format(j)
            elif "VAE" in model.__class__.__name__:
                gen_inter_image = tf.image.resize(model.decode(decode_batch.astype(np.float32), training=False), (28, 28)) * 255.0
                name = "vae_{}.png".format(j)
            else:
                logging.fatal("Error, wrong embedding model")

            stitched_interpolation = np.hstack(gen_inter_image)
            Image.fromarray(stitched_interpolation.astype(np.uint8)).save(os.path.join(folder, name))

            stitched_original = np.hstack(composed_image_batch[3*j:3*j + 3])
            Image.fromarray(stitched_original.astype(np.uint8)).save(os.path.join(folder, "orig"+name))

def four_rotate(embedding_models, clustering_methods, base_dir="compositionality", title=True):
    folder = os.path.join(base_dir, "four_rotate")
    ex_per_class = 25
    os.makedirs(folder, exist_ok=True)

    circle_config: HParams = configs.get_config("quickdraw")().parse("split=circle,shuffle=False,batch_size={}".format(ex_per_class))
    circle_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, circle_config)
    circle_dataset = circle_dataset_proto.load(repeat=False)[0]

    square_config: HParams = configs.get_config("quickdraw")().parse("split=square,shuffle=False,batch_size={}".format(ex_per_class))
    square_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, square_config)
    square_dataset = square_dataset_proto.load(repeat=False)[0]

    circle_batch, circle_names = next(circle_dataset.__iter__())[2:4]
    square_batch, square_names = next(square_dataset.__iter__())[2:4]

    circle_batch, square_batch = circle_batch.numpy(), square_batch.numpy()

    composite_batch = np.ones((circle_batch.shape[0] * 4, *circle_batch.shape[1:])) * 255.0

    for idx in range(circle_batch.shape[0]):
        circle_orig, square_orig = circle_batch[idx], square_batch[idx]
        circle, square = cv2.resize(circle_orig, (14, 14)), cv2.resize(square_orig, (14, 14))

        circle -= (circle < 200).astype(np.float32) * 100
        circle = np.maximum(np.zeros(circle.shape), circle)

        square -= (square < 200).astype(np.float32) * 100
        square = np.maximum(np.zeros(square.shape), square)

        place4((idx*4), composite_batch, [circle, circle, circle, square])
        place4((idx*4)+1, composite_batch, [square, circle, circle, circle])
        place4((idx*4)+2, composite_batch, [circle, square, circle, circle])
        place4((idx*4)+3, composite_batch, [circle, circle, square, circle])

    composite_batch = np.array(composite_batch, dtype=np.float32)

    fig, axs = plt.subplots(len(embedding_models), len(clustering_methods), figsize=(10 * len(clustering_methods),  10 * len(embedding_models)))

    for i, model in enumerate(embedding_models):
        x = model.embed(composite_batch, training=False)[0]
        y_image = composite_batch

        alphas = y_image == (tf.ones(y_image.shape) * 255.0)
        alphas = 1-tf.cast(tf.reduce_all(alphas, axis=-1, keepdims=True), dtype=tf.float32)
        y_image = tf.concat((y_image/255.0, alphas), axis=-1).numpy()

        # Set colors
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]] #, [1, 0, 1], [0, 1, 1]]
        pre_idx = np.array([k for k in range(0, x.shape[0], len(colors))])
        for k, col_array in enumerate(colors):
            idx = pre_idx + k
            curr_imgs = y_image[idx][:, :, :, :3]
            black_mask = np.all((curr_imgs - tf.zeros(curr_imgs.shape) > 0.08), axis=-1, keepdims=True)
            y_image[idx, :, :, :3] = black_mask * col_array

        for k in range(4):
            Image.fromarray((y_image[k] * 255.0).astype('uint8')).save(os.path.join(folder, "legend-{}.png".format(k)))

        project_plot(clustering_methods, x, axs, title, model, i, y_image)

    fig.tight_layout()
    fig.savefig(os.path.join(folder, "four.png"))

def place4(idx, source, shapes):
    source[idx, 0:14, 0:14, :] = shapes[0]
    source[idx, 0:14, 14:, :] = shapes[1]
    source[idx, 14:, 0:14, :] = shapes[2]
    source[idx, 14:, 14:, :] = shapes[3]

def project_plot(clustering_methods, x, axs, title, model, i, y_image):
    for j, method in enumerate(clustering_methods):
        x_2d = method.fit_transform(x)

        if len(clustering_methods) == 1:
            ax = axs[i]
        else:
            ax = axs[i][j]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(x_2d[:, 0], x_2d[:, 1], facecolors='none', edgecolors='none')

        for k, img in enumerate(y_image):
            ab = AnnotationBbox(OffsetImage(img), (x_2d[k, 0], x_2d[k, 1]), frameon=False)
            ax.add_artist(ab)

        if title:
            ax.set_title("{}".format("SketchEmbedding" if model.__class__.__name__ == "DrawerEncTADAMModel" else model.__class__.__name__),
                         fontsize=50)
        else:
            ax.set_title("   ", fontsize=50)

def lin_readout(model, composed_image_batch, y, num_train_ex):
    reg_model_nl = MLPRegressor(max_iter=3000)
    reg_model_lin = linear_model.Ridge()
    x = model.embed(composed_image_batch.astype(np.float32), training=False)[0]

    reg_model_nl.fit(x[:num_train_ex], y[:num_train_ex, 0])
    reg_model_lin.fit(x[:num_train_ex], y[:num_train_ex, 0])
    logging.info("%s, linear:(%f, %f), nonlinear:(%f, %f)",
                 model.__class__.__name__,
                 reg_model_lin.score(x[num_train_ex:], y[num_train_ex:, 0]),
                 sklearn.metrics.mean_squared_error(y[num_train_ex:, 0], reg_model_lin.predict(x[num_train_ex:])),
                 reg_model_nl.score(x[num_train_ex:], y[num_train_ex:, 0]),
                 sklearn.metrics.mean_squared_error(y[num_train_ex:, 0], reg_model_nl.predict(x[num_train_ex:])),)


def main(argv):
    """Create directories and configure python settings"""

    # Setup Directory
    experiment_dir = os.path.join(FLAGS.dir, FLAGS.id)
    if not os.path.exists(experiment_dir):
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

    # Setup Logging
    FLAGS.alsologtostderr = True
    logging.get_absl_handler().use_absl_log_file(FLAGS.logfile, os.path.join(experiment_dir, "logs"))

    # Setup seeds
    if FLAGS.random_seed:
        np.random.seed(FLAGS.random_seed)
        tf.random.set_seed(FLAGS.random_seed)

    # Log Flags
    log_flags(FLAGS)
    
    drawer_id = FLAGS.drawer_id
    drawer_config: HParams = configs.get_config(FLAGS.drawer_cfgset)().parse(FLAGS.drawer_cfgs)
    drawer: DrawerModel = models.get_model(FLAGS.drawer_model)(FLAGS.dir, drawer_id, drawer_config, training=False)

    vae_id = FLAGS.vae_id
    print(vae_id)
    vae_config: HParams = configs.get_config(FLAGS.vae_cfgset)().parse(FLAGS.vae_cfgs)
    vae: VAE = models.get_model(FLAGS.vae_model)(FLAGS.dir, vae_id, vae_config, training=False)
    
    embedding_models = [drawer, vae]
    clustering_methods = [umap.UMAP()]

    try:
        if FLAGS.conceptual_composition:
            conceptual_composition(embedding_models, clustering_methods, experiment_dir)

        if FLAGS.relation_count:
            relation_count(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.relation_count_toy:
            relation_count_toy(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.relation_orient:
            relation_orientation(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.relation_inout:
            relation_inout(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.relation_four:
            relation_four(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.relation_count_readout:
            latent_count_readout(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.relation_four_readout:
            relation_four_readout(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.relation_nest_readout:
            relation_nest_readout(embedding_models, clustering_methods, experiment_dir)

        if FLAGS.latent_distance:
            latent_distance_cts(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.latent_angle:
            latent_angle_cts(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.latent_size:
            latent_size_cts(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.latent_angle_readout:
            latent_angle_readout(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.latent_distance_readout:
            latent_distance_readout(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.latent_size_readout:
            latent_size_readout(embedding_models, clustering_methods, experiment_dir)

        if FLAGS.n_interpolate:
            n_interpolate(embedding_models, clustering_methods, experiment_dir)
        if FLAGS.four_rotate:
            four_rotate(embedding_models, clustering_methods, experiment_dir)
    except:
        exception = traceback.format_exc()
        logging.info(exception)


if __name__ == "__main__":
    app.run(main)
