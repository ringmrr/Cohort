# Imports the function needed to import data.
from tensorflow.examples.tutorials.mnist import input_data

# Imports the MNIST training set.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Shuffles the data set, and returns 100 random samples.
# X is the set of features for each sample, and Y is the label.
# batch_xs[0] and batch_ys[0] correspond to sample 1.
# batch_xs[1] and batch_ys[1] correspond to sample 2.
batch_xs, batch_ys = mnist.train.next_batch(100)

# Prints the features and label for the first sample.
data = batch_xs[0]
label = batch_ys[0]

print(data)
print(label)
