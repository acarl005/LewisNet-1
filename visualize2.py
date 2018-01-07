import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt

from alexnet import AlexNet

# load the saved tensorflow model and evaluate a list of paths to PNG files (must be 150x150)
num_classes = 1

image_init = np.random.uniform(200, 255, (1, 150, 150, 1))
X = tf.Variable(image_init, dtype=tf.float32, name="X")
dropout = tf.constant(0)

model = AlexNet(X, dropout, num_classes, trainable=False)

graph = tf.get_default_graph()
# from our graph, get the tensors for our hidden layers
first_hidden_layer_tensor = graph.get_tensor_by_name("conv1/Relu:0")
second_hidden_layer_tensor = graph.get_tensor_by_name("conv2/Relu:0")
third_hidden_layer_tensor = graph.get_tensor_by_name("conv3/Relu:0")

activations = second_hidden_layer_tensor[:, :, :, 0]
activation_mean = tf.reduce_mean(activations)

with tf.name_scope("maximize_activation"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    optimization_op = optimizer.minimize(-activation_mean)

# pretrained_variables = [n for n in graph.as_graph_def().node if "Variable" in n.op and n.name != "X"]
all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
pretrained_variables = [var for var in all_variables if var.op.name != "X"]
saver = tf.train.Saver(pretrained_variables)

with tf.Session() as sess:
    sess.run(X.initializer)
    saver.restore(sess, "./tensorflow-ckpt/model.ckpt")
    for i in range(100):
        _, act = sess.run((optimization_op, activation_mean))
        print(act)

    image_end = sess.run(X)
    fig = plt.figure()
    sub1 = fig.add_subplot(2, 1, 1)
    sub1.imshow(np.squeeze(image_init), cmap="gray")
    sub2 = fig.add_subplot(2, 1, 2)
    sub2.imshow(np.squeeze(image_end), cmap="gray")
    fig.savefig("sample.png")

