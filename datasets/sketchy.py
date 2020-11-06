import os
from concurrent.futures.process import ProcessPoolExecutor
from time import time
from itertools import repeat
from math import ceil

import numpy as np
import psutil
import csv
import requests
import gc
from PIL import Image

from absl import logging
from svgpathtools import svg2paths

from datasets.base import register_dataset, DatasetBase
from util import sketch_process, miniimagenet_test, svg_to_stroke_three, get_normalizing_scale_factor, sketchy_train_list, \
    sketchy_val_list


@register_dataset("sketchy")
class SketchyDataset(DatasetBase):
    def __init__(self, data_dir, params):
        super(SketchyDataset, self).__init__(data_dir, params)

        self._dataset_path = os.path.join(self._data_dir, "sketchy")

    def load(self, repeat=True):
        data_path = os.path.join(self._dataset_path, 'caches', self._split)
        files = [os.path.join(data_path, shard_name) for shard_name in os.listdir(data_path)]

        return self._create_dataset_from_filepaths(files, repeat)

    def _filter_collections(self, files):
        """
        files = ['natural_image', 'strokes', 'rasterized_strokes', 'imagenet_id', 'sketch_id']
        :param files:
        :return: strokes_gt, strokes_teacher, natural_image, class_name, rasterized_strokes
        """
        return files[1], files[1], files[0], files[3], files[2]

    def prepare(self, FLAGS, epsilon=5.0, max_seq_len=100, png_dims=(84, 84), padding=None, shard_size=1000,
                exclusive_set_parents=miniimagenet_test,
                class_list=sketchy_train_list, flip_x=True, flip_y=False):
        """
        Parallelized processing function that converts .svg files into our model dataset. Normalizes and deduplicates from mini-ImageNet.
        :param FLAGS:
        :param epsilon:
        :param max_seq_len:
        :param png_dims:
        :param padding:
        :param shard_size:
        :param exclusive_set_parents:
        :param class_list:
        :param flip_x:
        :param flip_y:
        :return:
        """
        padding = padding if padding else round(min(png_dims) / 10.0) * 2

        save_dir = os.path.join(self._dataset_path, "caches", self._split)
        sample_dir = os.path.join(self._dataset_path, "processing-samples", self._split)
        raw_dir = os.path.join(self._dataset_path, "raw")

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        raw_info_dir = os.path.join(raw_dir, "info-06-04", "info")
        raw_photo_dir = os.path.join(raw_dir, "rendered_256x256", "256x256", "photo", "tx_000000000000")
        raw_sketch_dir = os.path.join(raw_dir, "rendered_256x256", "256x256", "sketch", "tx_000100000000")
        raw_svg_dir = os.path.join(raw_dir, "sketches-06-04", "sketches")

        logging.info("Processing Sketchy | png_dimensions: %s | padding: %s | epsilon %s | max_seq_len %s | shard_size %s | only_valid %s", png_dims, padding,
                     epsilon, max_seq_len, shard_size, only_valid)

        # We sometimes desire a set of sketchy training examples that are class exclusive of anything in our downstream few-shot tasks.
        # As our images are sourced from imagenet, this is an issue with datsets such as miniimagenet and tieredimagenet.
        # While we are not currently testing on tieredimagenet, we should dedupe classes from miniimagenet.
        # To do this, we identify all hyponyms of minimagenet test classes and remove any sketchy examples that are of imagenet IDs that
        # belong in the hyponym sets of all minimagenet test classes.
        # NOTE: This is deduped on a per-image basis, not a per-class in sketchy basis. Many sketchy classes contain multiple IDs,
        # Some of which may be excluded, some of which may not.
        exclusive_set = set()
        for parent in exclusive_set_parents:
            res = requests.get("http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={}&full=1".format(parent))
            hyponyms = res.content.decode("utf-8").replace("\r\n", "").split("-")
            exclusive_set.update(hyponyms)

        imagenet_to_bbox = {}
        with open(os.path.join(raw_info_dir, "stats.csv"), 'r', newline='\n') as statscsv:
            reader = csv.reader(statscsv, delimiter=',')
            next(reader)
            for row in reader:
                imagenet_id_plus_count, bbox, width_height = row[2], row[14:18], row[12:14]
                if imagenet_id_plus_count not in imagenet_to_bbox:
                    # Note: BBOX is in BBox_x, BBox_y, BBox_width, BBox_height
                    imagenet_to_bbox[imagenet_id_plus_count] = [int(x) for x in bbox], [int(x) for x in width_height]

        sketchy_imagenet_hyponyms = {}
        skipped = {}  # Keep track of skipped sets for logging purposes
        class_list = [x.replace(" ", "_") for x in class_list]
        all_examples = np.empty((0, 6))
        for class_name in class_list:
            logging.info("Loading Class: %s", class_name)
            photo_folder = os.path.join(raw_photo_dir, class_name)
            sketch_folder = os.path.join(raw_sketch_dir, class_name)
            svg_folder = os.path.join(raw_svg_dir, class_name)

            valid_sketches_for_class = open(os.path.join(svg_folder, "checked.txt")).read().splitlines()
            invalid_sketches_for_class = set(open(os.path.join(svg_folder, "invalid.txt")).read().splitlines())
            for photo_file in os.listdir(photo_folder):
                imagenet_id_plus_count = photo_file.split(".")[0]

                # Determine if the set of hyponyms of the miniimagenet dataset intersect with any hyponyms of this imagenet id.
                # If so, there is class overlap and as such we will pass the image.
                if exclusive_set:
                    imagenet_id = imagenet_id_plus_count.split("_")[0]
                    if imagenet_id not in sketchy_imagenet_hyponyms:
                        res = requests.get("http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={}&full=1".format(imagenet_id))
                        hyponyms = res.content.decode("utf-8").replace("\r\n", "").split("-")
                        sketchy_imagenet_hyponyms[imagenet_id] = set(hyponyms)

                    if not exclusive_set.isdisjoint(sketchy_imagenet_hyponyms[imagenet_id]):
                        if imagenet_id not in skipped:
                            skipped[imagenet_id] = 1
                            logging.info("Skipping ImageNet ID: %s | from sketchy class %s", imagenet_id, class_name)
                        else:
                            skipped[imagenet_id] += 1
                        continue

                sketches_for_photo = list(filter(lambda x: photo_file[:-4] in x, valid_sketches_for_class))
                valid_sketches_for_photo = list(filter(lambda sketch: sketch not in invalid_sketches_for_class, sketches_for_photo))

                natural_image = Image.open(os.path.join(photo_folder, photo_file))
                for valid_sketch in valid_sketches_for_photo:
                    valid_sketch_path = os.path.join(svg_folder, valid_sketch+".svg")
                    try:
                        svg = svg2paths(valid_sketch_path)[0]
                    except:
                        with open(valid_sketch_path, "r") as errorsvg:
                            val = errorsvg.read()
                        if "</svg>" not in val[-10:]:
                            with open(valid_sketch_path, "a") as errorsvg:
                                errorsvg.write("</svg>\n")
                            logging.info("fixed %s", valid_sketch_path)
                        else:
                            logging.info("still_broken %s", valid_sketch_path)
                        if only_valid:
                            continue
                        else:
                            svg = None

                    sketch_path = os.path.join(sketch_folder, valid_sketch + ".png")

                    x = np.array([[natural_image, sketch_path, svg, valid_sketch] + list(imagenet_to_bbox[imagenet_id_plus_count])])
                    all_examples = np.concatenate((all_examples, x))
        np.random.shuffle(all_examples)

        logging.info("Total Skipped: | %s", str(skipped))
        logging.info("Beginning Processing | %s sketches | %s classes ",
                     all_examples.shape[0], len(class_list))

        cpu_count = psutil.cpu_count(logical=False)
        workers_per_cpu = 0.5

        # First, parallelize our computation of stroke-three formats from our svg files
        # with ProcessPoolExecutor(max_workers=int(cpu_count * workers_per_cpu)) as executor:
        with ProcessPoolExecutor(max_workers=int(cpu_count * workers_per_cpu)) as executor:
            out_iter = executor.map(svg_to_stroke_three,
                                    (all_examples[i: i + shard_size, 2:3] for i in range(0, all_examples.shape[0], shard_size)),
                                    repeat(epsilon),
                                    repeat(flip_x),
                                    repeat(flip_y))
            try:
                count = 0
                for idx, data in enumerate(out_iter):
                    all_examples[idx * shard_size: (idx + 1) * shard_size, 2:3] = data
                    count += data.shape[0]
                    logging.info("Converted to stroke-three: %d/%d",
                                 count,
                                 all_examples.shape[0])
            except Exception as e:
                logging.info("SVGs Converted to stroke-three complete")

        # Clean up garbage to save RAM
        gc.collect()
        normalizing_scale_factor = get_normalizing_scale_factor(all_examples[:, 2:3])

        with ProcessPoolExecutor(max_workers=int(cpu_count * workers_per_cpu)) as executor:
            out = executor.map(sketch_process,
                               (all_examples[i: i + shard_size] for i in range(0, all_examples.shape[0], shard_size)),
                               repeat(padding),
                               repeat(max_seq_len),
                               repeat(png_dims),
                               repeat(normalizing_scale_factor),
                               (os.path.join(save_dir, "{}.npz".format(i)) for i in range(ceil(all_examples.shape[0] // shard_size) + 1)),
                               (os.path.join(sample_dir, "{}".format(i)) for i in range(ceil(all_examples.shape[0] // shard_size) + 1)),
                               chunksize=1)

            total_count = 0
            last_time = time()
            try:
                for write_signal in out:
                    total_count += write_signal
                    curr_time = time()
                    logging.info("Processed Total: {:8}/{:8} | Time/Batch: {:8.2f} | Time/Image: {:8.8f}"
                                .format(total_count,
                                        all_examples.shape[0],
                                        (curr_time - last_time) / cpu_count,
                                        (curr_time - last_time) / (cpu_count * shard_size)))
                    last_time = curr_time
            except Exception as e:
                logging.info("Processing Done")
                raise e

