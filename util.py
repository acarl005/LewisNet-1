import numpy as np

def shuffle(x, y):
    m = x.shape[0]
    rand_perm = np.random.permutation(m)
    x = x[rand_perm]
    y = y[rand_perm]
    return x, y

def get_batch(x, y, i, batch_size):
    index = i * batch_size
    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    return x_batch, y_batch

