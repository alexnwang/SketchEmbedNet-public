import os
from time import time

import numpy as np
from PIL import Image
from absl import logging

from util import stroke_three_format, scale_and_rasterize, stroke_three_format_centered, rasterize


def parallel_writer_sketches(path, queue, shard_size=1, cumul=False, png_dims=(28, 28)):
    logging.info("Archiving latent outputs to: %s", path)
    sample_path = path
    os.makedirs(sample_path, exist_ok=True)

    accumulate, count = None, 0
    while True:
        entry = queue.get()
        if not entry:
            break

        count += 1
        if count and count % shard_size == 0:
            if len(entry['stroke_five_sketches']) > 2:
                stroke_three_gt = stroke_three_format(entry['stroke_five_sketches'])
                if len(entry["stroke_five_sketches"]) == 65:
                    entry["rasterized_images"] = scale_and_rasterize(stroke_three_gt, png_dims, 2).astype("float32")
                elif len(entry['stroke_five_sketches']) == 101:
                    np_rasterized_gt_strokes = scale_and_rasterize(stroke_three_gt, png_dims, 2).astype('uint8')
                    rasterized_gt_strokes = Image.fromarray(np_rasterized_gt_strokes)
                    rasterized_gt_strokes.save(os.path.join(sample_path, "{}-{}_y_raster.jpg".format(entry["class_names"].decode("utf-8"), count)))
            else:
                stroke_three_gt = None
            stroke_three = stroke_three_format(entry["stroke_predictions"])
            entry["rasterized_predictions"] = scale_and_rasterize(stroke_three, png_dims, stroke_width=2).astype("float32")

            np_image = np.concatenate((entry["rasterized_images"], entry["rasterized_predictions"]))
            img = Image.fromarray(np_image.astype("uint8"))

            gt_image = entry["rasterized_images"].astype("uint8")
            predicted_image = entry["rasterized_predictions"].astype("uint8")

            gt_img = Image.fromarray(gt_image.astype("uint8"))
            pt_img = Image.fromarray(predicted_image.astype('uint8'))

            try:
                img.save(os.path.join(sample_path, "{}-{}.jpg".format(entry["class_names"].decode("utf-8"), count)))
                gt_img.save(os.path.join(sample_path, "{}-{}_x.jpg".format(entry["class_names"].decode("utf-8"), count)))
                pt_img.save(os.path.join(sample_path, "{}-{}_predicted.jpg".format(entry["class_names"].decode("utf-8"), count)))
            except:
                gt_img.save(os.path.join(sample_path, "{}-{}_x.jpg".format(entry["class_names"], count)))
                pt_img.save(os.path.join(sample_path, "{}-{}_predicted.jpg".format(entry["class_names"], count)))
                img.save(os.path.join(sample_path, "{}-{}.jpg".format(entry["class_names"], count)))

            if cumul:
                if stroke_three_gt is not None:
                    source_sample_dir = os.path.join(sample_path, "cumulative", "source", "{}-{}".format(entry['class_names'], count))
                    os.makedirs(source_sample_dir)

                    pen_strokes = np.copy(stroke_three_gt[:, 2])
                    for i in range(1, len(stroke_three_gt)):
                        copy_pen_strokes = np.copy(pen_strokes)
                        copy_pen_strokes[i:] = np.ones((len(stroke_three_gt) - i,))
                        stroke_three_gt[:, 2] = copy_pen_strokes
                        cumul_img = scale_and_rasterize(stroke_three_gt, png_dims, stroke_width=3).astype("float32")

                        cumul_img = Image.fromarray(cumul_img.astype('uint8'))
                        cumul_img.save(os.path.join(source_sample_dir, "gt_{}.jpg".format(i)))

                cum_sample_dir = os.path.join(sample_path, "cumulative", "predict", "{}-{}".format(entry['class_names'], count))
                os.makedirs(cum_sample_dir)

                pen_strokes = np.copy(stroke_three[:, 2])
                for i in range(1, len(stroke_three)):
                    copy_pen_strokes = np.copy(pen_strokes)
                    copy_pen_strokes[i:] = np.ones((len(stroke_three) - i,))
                    stroke_three[:, 2] = copy_pen_strokes
                    cumul_img = scale_and_rasterize(stroke_three, png_dims, stroke_width=3).astype("float32")

                    cumul_img = Image.fromarray(cumul_img.astype('uint8'))
                    cumul_img.save(os.path.join(cum_sample_dir, "pred_{}.jpg".format(i)))


def parallel_writer_vae_latent(path, queue, shard_size=1):
    logging.info("Archiving vae latent outputs to %s", path)

    sample_path = path
    os.makedirs(sample_path, exist_ok=True)

    start_time = last_time = time()
    count = 0
    while True:
        entry = queue.get()
        if not entry:
            break

        count += 1

        if count and count % shard_size == 0:
            np_image = np.concatenate((entry["rasterized_images"], entry["reconstructed_images"]))
            np_image = np_image.squeeze() * 255.0
            img = Image.fromarray(np_image.astype("uint8"))
            try:
                img.save(os.path.join(sample_path, "{}-{}.jpg".format(entry["class_names"].decode("utf-8"), count)))
            except:
                img.save(os.path.join(sample_path, "{}-{}.jpg".format(entry["class_names"], count)))

            curr_time = time()
            if count and count % 1000 == 0:
                logging.info("Samples complete: %6d | Time/Sample: %5.4f | Total Elapsed Time: %7d",
                             count, (curr_time - last_time) / shard_size, curr_time - start_time)
            last_time = curr_time
