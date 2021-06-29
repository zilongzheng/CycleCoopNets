import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import shutil
import json

class Logger(object):
    def __init__(self, log_dir, suffix=None):
        try:
            self.writer = tf.summary.FileWriter(log_dir, filename_suffix=suffix)
        except AttributeError:
            # log to local
            self.writer = None
            self.log_history = {}
            self.log_dir = log_dir
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir)

    def scalar_summary(self, tag, value, step):
        if self.writer:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)
        else:
            if not tag in self.log_history:
                self.log_history[tag] = {'step': [],  'value': []}
            self.log_history[tag]['step'].append(step)
            self.log_history[tag]['value'].append(value)

    def add_summary(self, summary, step):
        if self.writer:
            self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        if self.writer:
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
        else:
            log_img_dir = os.path.join(self.log_dir, 'images', tag)
            os.makedirs(log_img_dir, exist_ok=True)

            if len(images.shape) == 4:
                for i, img in enumerate(images):
                    Image.fromarray(img).save(os.path.join(log_img_dir, '{:05d}_{:d}.png'.format(step, i)), format='png')
            else:
                Image.fromarray(images).save(os.path.join(log_img_dir, '{:05d}.png'.format(step)), format='png')
            # flush
            if len(self.log_history) > 0:
                with open(os.path.join(self.log_dir, 'summary_log.json'), 'w') as f:
                    json.dump(self.log_history, f)
