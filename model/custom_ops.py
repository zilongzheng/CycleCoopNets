try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
import scipy.misc
from utils.data_util import img2cell
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO

def image_summary(tag, images, row_num=10, col_num=10, margin_syn=2):
    cell_images = img2cell(images, row_num=row_num, col_num=col_num, margin_syn=margin_syn)
    cell_image = cell_images[0]
    try:
        s = StringIO()
    except:
        s = BytesIO()
    scipy.misc.toimage(cell_image).save(s, format="png")
    img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=cell_image.shape[0], width=cell_image.shape[1])
    return tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])


    # return tf.contrib.layers.instance_norm(x, epsilon=epsilon, scope=name)