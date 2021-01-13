# Imports the function needed to import data.
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as pyplot
import numpy as numpy

# Imports the MNIST training set.
# The argument one_hot=True is a way of encoding categorical data as numerical data. Read more here: https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Shuffles the data set, and returns 100 random samples.
# X is the set of features for each sample, and Y is the label.
# batch_xs[0] and batch_ys[0] correspond to sample 1.
# batch_xs[1] and batch_ys[1] correspond to sample 2.
batch_xs, batch_ys = mnist.train.next_batch(100)

# Prints the features and label for the first sample.
data = np.rint(batch_xs[0]).astype(int)
label = np.rint(batch_ys[0]).astype(int)
pixels = data.reshape((28,28))

print(data)
print(label)
plt.imshow(pixels, cmap="gray")
plt.show()