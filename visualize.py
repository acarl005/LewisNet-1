import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from math import ceil, sqrt

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

def visualize_layer(hidden_layer, layer_number, deconvolve, figsize, subplot_layout, top_n=9):
    """finds the argmax in the hidden layer, back tracks to the patch of image that caused the activation, and plots that patch"""
    fig = plt.figure(figsize=figsize)
    # this just tells us the max size of a patch from input space. we need this now to determine the size of the composite image
    patch_range = deconvolve((0, 0), True)
    patch_size = patch_range[1] - patch_range[0] + 1

    # loop over the number of filters (a.k.a. number of channels)
    for f in range(hidden_layer.shape[3]):
        # get activations for the filter/channel we care about
        activations = hidden_layer[:, :, :, f]
        # the composite image is the 9 crops of the images put together into a single image. makes plotting easier
        composite_grid_size = ceil(sqrt(top_n))
        composite_pixel_size = (patch_size + 1) * composite_grid_size
        composite_img = np.full((composite_pixel_size, composite_pixel_size), 255)
        composite_img[composite_pixel_size - 1, composite_pixel_size - 1] = 0
        # find locations and values of top n activations
        kernel_argmax = largest_indices(activations, top_n)
        kernel_max_values = activations[kernel_argmax]
        for m in range(top_n):
            val = kernel_max_values[m]
            # if nothing activated it (all 0s), just show a white patch
            if val == 0:
                patch = np.full((patch_size, patch_size), 255)
            else:
                loc = [t[m] for t in kernel_argmax]
                i = loc[0]
                # back-track it to the location in the original image
                # this is different for each layer, so the function to do this is passed in
                j = deconvolve((loc[1], loc[1]))
                k = deconvolve((loc[2], loc[2]))
                # get that patch
                patch = np.squeeze(imgs[i, j[0]:j[1]+1, k[0]:k[1]+1])
                # because of padding in the convolutions/pooling, some of the images on the edges are smaller.
                # use np.pad to fill that in with gray background
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1])), "constant", constant_values=128)
            x, y = np.unravel_index(m, (composite_grid_size, composite_grid_size))
            # add the patch to the composite
            composite_img[x*(patch_size + 1):x*(patch_size + 1)+patch_size,
                          y*(patch_size + 1):y*(patch_size + 1)+patch_size] = patch
        sub = fig.add_subplot(subplot_layout[0], subplot_layout[1], f + 1)
        sub.imshow(composite_img, cmap="gray")
    fig.savefig("readme-img/layer{}.png".format(layer_number))

# load the saved tensorflow model and evaluate a list of paths to PNG files (must be 150x150)
num_classes = 1

X = tf.placeholder(tf.float32, shape=(None, 150, 150, 1))
dropout = tf.constant(0)

model = AlexNet(X, dropout, num_classes)

graph = tf.get_default_graph()
# from our graph, get the tensors for our hidden layers
first_hidden_layer_tensor = graph.get_tensor_by_name("conv1/Relu:0")
second_hidden_layer_tensor = graph.get_tensor_by_name("conv2/Relu:0")
third_hidden_layer_tensor = graph.get_tensor_by_name("conv3/Relu:0")

saver = tf.train.Saver()

imgs = load_images([
    "./positives/images/*.png",
    "./positives/manually-collected/*.png"
    ])

with tf.Session() as sess:
    saver.restore(sess, "./tensorflow-ckpt/model.ckpt")

    first_hidden_layer, second_hidden_layer, third_hidden_layer = sess.run((first_hidden_layer_tensor,
                                                                            second_hidden_layer_tensor,
                                                                            third_hidden_layer_tensor),
                                                                           feed_dict={X: imgs})

    def deconvolve_layer_1(rng, ignore_pad=False):
        return deconvolve_range(rng, filt_size=7, stride=4)

    def deconvolve_layer_2(rng, ignore_pad=False):
        patch_range = deconvolve_range(rng, filt_size=5, stride=1, pad=(0 if ignore_pad else 2))
        patch_range = deconvolve_range(patch_range, 3, 2)
        return deconvolve_range(patch_range, 7, 4)

    def deconvolve_layer_3(rng, ignore_pad=False):
        patch_range = deconvolve_range(rng, filt_size=3, stride=1, pad=(0 if ignore_pad else 1))
        patch_range = deconvolve_range(patch_range, 3, 2)
        patch_range = deconvolve_range(patch_range, 5, 1, (0 if ignore_pad else 2))
        patch_range = deconvolve_range(patch_range, 3, 2)
        return deconvolve_range(patch_range, 7, 4)

    visualize_layer(first_hidden_layer, 1, deconvolve_layer_1, figsize=(9, 13), subplot_layout=(12, 8))
    visualize_layer(second_hidden_layer, 2, deconvolve_layer_2, figsize=(9, 70), subplot_layout=(43, 6))
    visualize_layer(third_hidden_layer, 3, deconvolve_layer_3, figsize=(9, 200), subplot_layout=(96, 4))


