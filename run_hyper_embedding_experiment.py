import os
import traceback

import numpy as np
import cv2
import umap
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from PIL import Image
from absl import app, flags, logging
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import DBSCAN

import models
import configs
import datasets
from models import ClassifierModel, Protonet, DrawerModel, VAE
from util import HParams, scale_and_rasterize, stroke_three_format, scale_and_center_stroke_three, rasterize
from util import log_flags, log_hparams, color_rasterize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = flags.FLAGS

flags.DEFINE_string("dir", "/h/wangale/project/few-shot-sketch", "Project directory")
flags.DEFINE_string("data_dir", "/h/wangale/data", "Data directory")

flags.DEFINE_string("id", "hyper_state", "training_id")
flags.DEFINE_string("logfile", "comptest", "Logfile name")

flags.DEFINE_integer("random_seed", 1, "Random seed")

def hyper_embed(model: DrawerModel, dataset: tf.data.Dataset, clustering_method, png_dims=(48, 48), min_samples=3):
    folder = os.path.join(FLAGS.dir, FLAGS.id, clustering_method.__class__.__name__ + str(clustering_method.eps))
    os.makedirs(folder, exist_ok=True)
    color_img_folder = os.path.join(folder, "colorimgs")
    os.makedirs(color_img_folder, exist_ok=True)

    padding = round(min(png_dims) / 10.) * 2
    ds_iter = dataset.__iter__()
    init = False

    batch = next(ds_iter)[2]
    embedding = model.embed(batch, training=False)[0]

    if not init:
        model.decode(embedding, training=False, generation_length=64)
        init = True

    params, strokes, hyper_states = model.decode(embedding, training=False, generation_length=64, with_hyper_states=True)
    hyper_states = hyper_states[:, :, -512:]

    stroke_threes = []
    cut_hyper_states = []
    for i in range(len(hyper_states)):
        curr_strokes = strokes[i]
        curr_states = hyper_states[i]

        curr_strokes = stroke_three_format(curr_strokes)

        curr_strokes = scale_and_center_stroke_three(curr_strokes, png_dims, padding)
        curr_states = curr_states[:len(curr_strokes)]

        stroke_threes.append(curr_strokes)
        cut_hyper_states.append(curr_states)

    lengths = np.cumsum(np.array([0] + [len(x) for x in stroke_threes]))

    cut_hyper_states = np.vstack(cut_hyper_states)
    all_cluster_assignments = clustering_method.fit_predict(cut_hyper_states)
    all_images = []
    for i in range(len(hyper_states)):
        curr_strokes = stroke_threes[i]
        cluster_assignments = all_cluster_assignments[lengths[i]:lengths[i] + len(curr_strokes)]

        orig_strokes = np.copy(curr_strokes)

        curr_strokes_abs = np.copy(curr_strokes)
        curr_strokes_abs[:, :2] = np.cumsum(curr_strokes_abs[:, :2], axis=0)

        clustered_strokes = {cluster_assignments[0]: [curr_strokes[0]]}
        for j in range(1, len(cluster_assignments)):
            assignment = cluster_assignments[j]
            prev_assignment = cluster_assignments[j - 1]

            if assignment == prev_assignment:
                clustered_strokes[assignment].append(curr_strokes[j])
            elif assignment != prev_assignment:
                if assignment not in clustered_strokes:
                    clustered_strokes[assignment] = [curr_strokes_abs[j-1]]
                    # clustered_strokes[assignment][-1][-1] = 1
                elif assignment in clustered_strokes:
                    last_stroke_in_cluster = np.cumsum(clustered_strokes[assignment], axis=0)[-1]
                    previous_stroke = curr_strokes_abs[j-1]
                    clustered_strokes[assignment].append(previous_stroke - last_stroke_in_cluster)
                    clustered_strokes[assignment][-1][-1] = curr_strokes_abs[j - 1][-1]
                clustered_strokes[assignment].append(curr_strokes[j])

            if j+1 >= len(cluster_assignments):
                continue
            else:
                next_assignment = cluster_assignments[j+1]
                if assignment != next_assignment:
                    clustered_strokes[assignment].append(np.array([0., 0., 1.]))
                    clustered_strokes[assignment][-1][-1] = 1

        # images = [rasterize(orig_strokes, png_dims)]
        # images.append(np.zeros((png_dims[0], 1, 3)))
        images = []
        color_img = color_rasterize([np.array(x) for x in clustered_strokes.values()], png_dims, stroke_width=1)
        images.append(color_img)

        for key in range(-1, np.max(all_cluster_assignments)+1):
            if key in clustered_strokes:
                stroke_cluster = clustered_strokes[key]
                stroke_cluster = np.array(stroke_cluster)
                images.append(np.zeros((png_dims[0], 1, 3)))
                images.append(rasterize(stroke_cluster, png_dims))
            else:
                images.append(np.zeros((png_dims[0], 1, 3)))
                images.append(np.ones(list(png_dims) + [3]) * 255.0)

        final_image = np.concatenate(images, axis=1)
        all_images.append(final_image)
        all_images.append(np.zeros((1, final_image.shape[1], 3)))
        Image.fromarray(final_image.astype(np.uint8)).save(os.path.join(folder, "cluster-{}.png".format(i)))
        Image.fromarray(color_img.astype(np.uint8)).save(os.path.join(color_img_folder, "cluster-{}.png".format(i)))
    all_images = np.concatenate(all_images, axis=0)
    Image.fromarray(all_images.astype(np.uint8)).save(os.path.join(folder, "collage.png".format(i)))


def project_plot(clustering_methods, x, axs, title, model, y_image):
    for j, method in enumerate(clustering_methods):
        x_2d = method.fit_transform(x)

        if len(clustering_methods) == 1:
            ax = axs
        else:
            ax = axs[j]
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

    drawer_id = "05-22_quickdraw_sweep_pixel_weight_28/drawer_enc_tadam_huge_interval0.05_step10000_maxweight0.5-quickdraw_ST1_msl64_28"
    drawer_config: HParams = configs.get_config("drawer/huge")().parse("")
    drawer: DrawerModel = models.get_model("drawer_enc_tadam")(FLAGS.dir, drawer_id, drawer_config, training=False)

    dataset_config: HParams = configs.get_config("quickdraw")().parse("split=T2_msl64_28,shuffle=True,batch_size={}".format(16))
    dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, dataset_config)
    dataset = dataset_proto.load(repeat=False)[0]

    # clustering_methods = [sklearn.manifold.TSNE(n_components=2), sklearn.decomposition.PCA(n_components=2), umap.UMAP()]
    clustering_method = DBSCAN(eps=4.2)
    try:
        hyper_embed(drawer, dataset, clustering_method, min_samples=6)
    except:
        exception = traceback.format_exc()
        logging.info(exception)

    logging.info("Complete")


if __name__ == "__main__":
    app.run(main)
