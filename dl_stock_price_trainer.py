import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

DEFAULT_STOCK_FILE = "data/data_stocks.csv"


def load_stock_data():
    data = pd.read_csv(DEFAULT_STOCK_FILE)

    # Drop date variable
    data = data.drop(['DATE'], 1)

    return data


def build_mlp(n_stocks, n_target, input):
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128

    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Hidden layers
    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(input, W_hidden_1), bias_hidden_1))

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

    # Output layer (must be transposed)
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

    return out


def main():
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

    n_stocks = 500
    n_target = 1

    # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    mlp_out = build_mlp(n_stocks, n_target, X)

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(mlp_out, Y))

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # Make Session
    session = tf.Session()

    # Run initializer
    session.run(tf.global_variables_initializer())

    # Number of epochs and batch size
    epochs = 10
    batch_size = 256

    for e in range(epochs):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        n_batch = len(y_train) / batch_size

        # Minibatch training
        for i in range(0, n_batch):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]

            # Run optimizer with batch
            session.run(opt, feed_dict={X: batch_x, Y: batch_y})


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    
    main()