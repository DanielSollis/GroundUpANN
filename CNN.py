import numpy as np
from keras.datasets import mnist


def convolve(image, filter, bias, step=1):
    (n_f, n_c_f, f, _) = filter.shape
    n_c, in_dim, _ = image.shape
    out_dim = int((in_dim - f)/step) + 1
    assert n_c == n_c_f, "Filter and Image Dimensions must match: \n" + str(n_c.shape) + "\n" + str(n_c_f.shape)
    out = np.zeros((n_f, out_dim, out_dim)) # holds convolution results
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filter[curr_f] * image[:,curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += step
                out_x += 1
            curr_y += step
            out_y += 1
    return out


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
convolve(x_train, x_train[1], 3)