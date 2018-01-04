import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
import math

from alexnet import AlexNet
from image_loader import load_images

def largest_indices(ary, n):
    """Returns the indices for the n largest values from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def deconvolve_range(rng, filt_size, stride=1, pad=0):
    """Returns the "scope" or range of pixels in a prior convolution layer that influence a specified range in the current layer"""
    start, end = rng
    if (start > end):
        raise ValueError("start value {} cannot be greater than end value {}".format(start, end))
    prev_start = max(0, start * stride - pad)
    prev_end = max(0, end * stride + filt_size - pad - 1)
    return prev_start, prev_end

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

    # from our graph, get the tensors for our hidden layers
    first_hidden_layer_tensor = tf.get_default_graph().get_tensor_by_name("conv1/Relu:0")
    second_hidden_layer_tensor = tf.get_default_graph().get_tensor_by_name("conv2/Relu:0")
    first_hidden_layer, second_hidden_layer = sess.run([ first_hidden_layer_tensor, second_hidden_layer_tensor ], feed_dict={X: imgs, dropout: 0})

    # get the 9 strongest activations
    top_n = 9

    fig1 = plt.figure(figsize=(9, 13))

    # determine how big the image patch is from the first layer that influences a single activation in this hidden layer
    patch_range = deconvolve_range((0, 0), filt_size=7, stride=4, pad=0)
    patch_size = patch_range[1] - patch_range[0] + 1
    for f in range(96):
        activations = first_hidden_layer[:, :, :, f]
        # the 9 image patches will be pasted together into one "composite" image
        composite_grid_size = math.ceil(math.sqrt(top_n))
        composite_pixel_size = (patch_size + 1) * composite_grid_size
        # make the background white
        composite_img = np.full((composite_pixel_size, composite_pixel_size), 255)
        # make the bottom right corner pixel black. this is a hack to make sure the whole grayscale is present in the image
        composite_img[composite_pixel_size - 1, composite_pixel_size - 1] = 0

        kernel_argmax = largest_indices(activations, top_n)
        kernel_max_values = activations[kernel_argmax]

        for m in range(top_n):
            val = kernel_max_values[m]
            # if 0 is the max activation, we know we just need all white
            if val == 0:
                patch = np.full((patch_size, patch_size), 255)
            else:
                loc = [t[m] for t in kernel_argmax]
                i = loc[0]
                j = deconvolve_range((loc[1], loc[1]), 7, 4)
                k = deconvolve_range((loc[2], loc[2]), 7, 4)
                patch = np.squeeze(imgs[i, j[0]:j[1]+1, k[0]:k[1]+1])
                x, y = np.unravel_index(m, (composite_grid_size, composite_grid_size))
                # patches on the edge might be too small. pad to make sure they are the right size
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1])), "constant", constant_values=128)
            # add the patch to the composite image
            composite_img[x*(patch_size + 1):x*(patch_size + 1)+patch_size,
                          y*(patch_size + 1):y*(patch_size + 1)+patch_size] = patch
        sub = fig1.add_subplot(12, 8, f + 1)
        sub.imshow(composite_img, cmap="gray")
    fig1.savefig('readme-img/layer1.png')

    fig2 = plt.figure(figsize=(9, 72))
    def deconvolve_all(rng, ignore_pad=False):
        patch_range = deconvolve_range(rng, filt_size=5, stride=1, pad=(0 if ignore_pad else 2))
        patch_range = deconvolve_range(patch_range, 3, 2)
        return deconvolve_range(patch_range, 7, 4)
    patch_range = deconvolve_all((0, 0), True)
    patch_size = patch_range[1] - patch_range[0] + 1
    for f in range(256):
        activations = second_hidden_layer[:, :, :, f]
        composite_grid_size = math.ceil(math.sqrt(top_n))
        composite_pixel_size = (patch_size + 1) * composite_grid_size
        composite_img = np.full((composite_pixel_size, composite_pixel_size), 255)
        composite_img[composite_pixel_size - 1, composite_pixel_size - 1] = 0
        kernel_argmax = largest_indices(activations, top_n)
        kernel_max_values = activations[kernel_argmax]
        for m in range(top_n):
            val = kernel_max_values[m]
            if val == 0:
                patch = np.full((patch_size, patch_size), 255)
            else:
                loc = [t[m] for t in kernel_argmax]
                i = loc[0]
                j = deconvolve_all((loc[1], loc[1]))
                k = deconvolve_all((loc[2], loc[2]))
                patch = np.squeeze(imgs[i, j[0]:j[1]+1, k[0]:k[1]+1])
                x, y = np.unravel_index(m, (composite_grid_size, composite_grid_size))
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1])), "constant", constant_values=128)
            composite_img[x*(patch_size + 1):x*(patch_size + 1)+patch_size,
                          y*(patch_size + 1):y*(patch_size + 1)+patch_size] = patch
        sub = fig2.add_subplot(43, 6, f + 1)
        sub.imshow(composite_img, cmap="gray")
    fig2.savefig('readme-img/layer2.png')


