import sys
import tensorflow as tf

from alexnet import AlexNet
from image_loader import load_images

# load the saved tensorflow model and evaluate a list of paths to PNG files (must be 150x150)
num_classes = 2

X = tf.placeholder(tf.float32, shape=(None, 150, 150, 1))
Y = tf.placeholder(tf.float32, shape=(None, num_classes))
dropout = tf.placeholder(tf.float32)

model = AlexNet(X, dropout, num_classes)

predictions = tf.argmax(model.logits, axis=1)

saver = tf.train.Saver()

files = sys.argv[1:]
imgs = load_images(files)

with tf.Session() as sess:
    saver.restore(sess, "./model.ckpt")
    preds = sess.run(predictions, feed_dict={X: imgs, dropout: 0})
    print([ str(i) + ": " + ("non-molecule" if pred else "molecule") for i, pred in enumerate(preds) ])

