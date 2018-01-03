import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
import math

from alexnet import AlexNet
from image_loader import load_images

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

# load the saved tensorflow model and evaluate a list of paths to PNG files (must be 150x150)
num_classes = 1

X = tf.placeholder(tf.float32, shape=(None, 150, 150, 1))
Y = tf.placeholder(tf.float32, shape=(None, num_classes))
dropout = tf.placeholder(tf.float32)

model = AlexNet(X, dropout, num_classes)

predictions = model.logits > 0

saver = tf.train.Saver()

imgs = load_images([
    "./positives/images/*.png",
    "./positives/manually-collected/*.png"
    ])

with tf.Session() as sess:
    saver.restore(sess, "./tensorflow-ckpt/model.ckpt")

    first_hidden_layer_tensor = tf.get_default_graph().get_tensor_by_name("conv1/Relu:0")
    second_hidden_layer_tensor = tf.get_default_graph().get_tensor_by_name("conv2/Relu:0")
    first_hidden_layer, second_hidden_layer = sess.run([ first_hidden_layer_tensor, second_hidden_layer_tensor ], feed_dict={X: imgs, dropout: 0})

    fig = plt.figure()
    stride = 4
    filt_size = 7
    top_n = 9
    for f in range(96):
        activations = first_hidden_layer[:, :, :, f]
        composite_size = math.ceil(math.sqrt(top_n))
        composite_img = np.full(((filt_size + 1) * composite_size, (filt_size + 1) * composite_size), 255)
        kernel_argmax = largest_indices(activations, top_n)
        for m in range(top_n):
            loc = [t[m] for t in kernel_argmax]
            i = loc[0]
            j = loc[1] * stride
            k = loc[2] * stride
            patch = np.squeeze(imgs[i, j:j+filt_size, k:k+filt_size])
            x, y = np.unravel_index(m, (composite_size, composite_size))
            composite_img[x*(filt_size + 1):x*(filt_size + 1)+filt_size, y*(filt_size + 1):y*(filt_size + 1)+filt_size] = patch
        sub = fig.add_subplot(12, 8, f + 1)
        sub.imshow(composite_img, cmap="gray")
    plt.show()

