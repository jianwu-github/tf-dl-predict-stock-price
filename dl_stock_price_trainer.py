import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

DEFAULT_STOCK_FILE = "data/data_stocks.csv"
DEFAULT_BATCH_SIZE = 256


def load_stock_data():
    data = pd.read_csv(DEFAULT_STOCK_FILE, sep=',')

    # Drop date variable
    data = data.drop(['DATE'], 1)

    return np.array(data)


def build_mlp(n_stocks, n_target, input_data):
    n_neurons_1 = 2048  # 1024
    n_neurons_2 = 1024  # 512
    n_neurons_3 = 512   # 256
    n_neurons_4 = 256   # 128
    n_neurons_5 = 128

    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Hidden layers
    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(input_data, W_hidden_1), bias_hidden_1))

    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    # Layer 5: Variables for hidden weights and biases
    W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
    bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))
    hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))

    # Output layer (must be transposed)
    W_out = tf.Variable(weight_initializer([n_neurons_5, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))
    out = tf.transpose(tf.add(tf.matmul(hidden_5, W_out), bias_out))

    return out


def train_mlp(epochs):
    # Preparing training and testing data
    data = load_stock_data()
    n = data.shape[0]
    p = data.shape[1]

    # split training and testing data
    train_start = 0
    train_end = int(np.floor(0.8 * n))
    test_start = train_end + 1
    test_end = n

    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    # rescale
    rescaler = MinMaxScaler()
    rescaler.fit(data_train)
    rescaler.transform(data_train)
    rescaler.transform(data_test)

    X_train = data_train[:, 1:]
    Y_train = data_train[:, 0]
    X_test = data_test[:, 1:]
    Y_test = data_test[:, 0]

    n_stocks = 500
    n_target = 1

    tf.reset_default_graph()

    # Placeholder
    X_input = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks], name="X_input")
    Y_input = tf.placeholder(dtype=tf.float32, shape=[None], name="Y_input")

    mlp_out = build_mlp(n_stocks, n_target, X_input)

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(mlp_out, Y_input))

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # Make Session
    session = tf.Session()

    # Run initializer
    session.run(tf.global_variables_initializer())

    # set batch size
    batch_size = DEFAULT_BATCH_SIZE

    for e in range(epochs):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(Y_train)))
        X_train = X_train[shuffle_indices]
        Y_train = Y_train[shuffle_indices]

        n_batch = int(len(Y_train) / batch_size)

        # Minibatch training
        for i in range(0, n_batch):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = Y_train[start:start + batch_size]

            # Run optimizer with batch
            session.run(opt, feed_dict={X_input: batch_x, Y_input: batch_y})

        mse_val = session.run(mse, feed_dict={X_input: X_test, Y_input: Y_test})
        print("At {}th epoch, mse for X_test/Y_test is {}".format(e, mse_val))


parser = argparse.ArgumentParser(description="Training TensorFlow MLP to predict SP500 Index")
parser.add_argument('-ep', type=int, default=10, help="Number of epochs")

if __name__ == "__main__":
    parsed_args = parser.parse_args()
    epochs = parsed_args.ep

    tf.logging.set_verbosity(tf.logging.INFO)
    
    train_mlp(epochs)