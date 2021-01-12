import tensorflow as tf
import numpy as np
from PIL import Image
import io

def clip_by_value(input_, low=0, high=1):
    return np.minimum(high, np.maximum(low, input_))

def normalize_image(img, mean=0.5, stddev=0.5):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img - mean ) / stddev
    return img

def numpy_to_image(np_arr):
    img = (np.squeeze(np_arr) + 1.) * 127.5
    img = np.clip(img, a_min=0., a_max=255.).astype(np.uint8)
    return img

class Logger(object):
    def __init__(self, log_dir, suffix=None):
        self.writer = tf.summary.FileWriter(log_dir, filename_suffix=suffix)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def add_summary(self, summary, step):
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):

        def image_to_summary(img, tag):
            # Write the image to a string
            s = io.BytesIO()
            Image.fromarray(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                    height=img.shape[0],
                                    width=img.shape[1])
            return tf.Summary.Value(tag=tag, image=img_sum)

        img_summaries = []

        if len(images.shape) == 4:
            for i, img in enumerate(images):
                # Create a Summary value
                img_summaries.append(image_to_summary(img, '%s/%d' % (tag, i)))
        else:
            img_summaries.append(image_to_summary(images, tag))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
