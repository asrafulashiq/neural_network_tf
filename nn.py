import os

import numpy as np

import cv2
import tensorflow as tf
from tensorflow.python import debug as tf_debug

test_folder = 'test_data'
train_folder = 'train_data'

test_label = 'labels/test_label.txt'
train_label = 'labels/train_label.txt'


def create_data(folder, label_txt):
    # get all image filenames from folder
    filenames = sorted([
        os.path.join(folder, i) for i in os.listdir(folder)
        if i.endswith('jpg')
    ])
    labels = [int(i.strip()) for i in open(label_txt)]

    X_data = np.zeros((len(filenames), 784), dtype=np.float32)
    Y_data = np.zeros((len(filenames), 10), dtype=np.float32)

    for i in range(len(filenames)):
        filename = filenames[i]
        label = labels[i]
        im = cv2.imread(filename, 0)
        X_data[i, :] = im.flatten().astype(np.float32) / 255.
        Y_data[i, int(label)] = 1
    return X_data, Y_data


class NN:
    def __init__(self, lr=0.1, epoch=3, batch_size=50):
        self.learning_rate = lr
        self.epochs = epoch
        self.batch_size = batch_size
        # network parameter
        self.n_input = 784
        self.n_classes = 10
        self.n_hidden_1 = 100
        self.n_hidden_2 = 100
        
        self.x = tf.placeholder(tf.float32, [None, 784, 1])
        self.y = tf.placeholder(tf.float32, [None, 10, 1])

        self.weights = {
            'h1':
            tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1], stddev=.1)),
            'h2':
            tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], stddev=.1)),
            'out':
            tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes], stddev=.1))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden_1, 1])),
            'b2': tf.Variable(tf.zeros([self.n_hidden_2, 1])),
            'out': tf.Variable(tf.zeros([self.n_classes, 1]))
        }

    def soft_max(self, x):
        ex = tf.exp(x - tf.reduce_max(x))
        den = tf.reduce_sum(ex, 1)
        D = tf.tile(tf.expand_dims(den, 1), [1, n_classes, 1])
        return ex / D

    def expand(self):
        self.Wh1 = tf.tile(
            tf.expand_dims(self.weights['h1'], 0), [self.batch_size, 1, 1])
        self.Wh2 = tf.tile(
            tf.expand_dims(self.weights['h2'], 0), [self.batch_size, 1, 1])
        self.Wout = tf.tile(
            tf.expand_dims(self.weights['out'], 0), [self.batch_size, 1, 1])
        self.Bb1 = tf.tile(
            tf.expand_dims(self.biases['b1'], 0), [self.batch_size, 1, 1])
        self.Bb2 = tf.tile(
            tf.expand_dims(self.biases['b2'], 0), [self.batch_size, 1, 1])
        self.Bout = tf.tile(
            tf.expand_dims(self.biases['out'], 0), [self.batch_size, 1, 1])

    def forward(self):
        self.zh1 = tf.matmul(tf.transpose(self.Wh1, [0, 2, 1]), self.x) + self.Bb1
        self.h_1 = tf.nn.relu(self.zh1)
        self.zh2 = tf.matmul(tf.transpose(self.Wh2, [0, 2, 1]), h_1) + self.Bb2
        self.h_2 = tf.nn.relu(self.zh2)
        y_ = self.soft_max(
            tf.matmul(tf.transpose(self.Wout, [0, 2, 1]), h_2) + self.Bout)
        return y_

    def score_eval(self, y_):
        out['loss'] = loss_fn(self.y, y_)
        out['score'] = accuracy(y_, self.y)
        return out

    def loss_fn(self, out_layer):
        loss = 0.5 * \
            tf.reduce_sum(tf.squared_difference(out_layer, self.y)) / batch_size
        return loss

    def accuracy(self, pred):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy

    def backprop(self, y_):
        grad = {}
        grad['y'] = -(self.y - y_)

        I = tf.tile(
            tf.expand_dims(tf.eye(self.n_classes), 0), [self.batch_size, 1, 1])
        grad['z'] = tf.matmul(I - y_, tf.multiply(y_, grad['y']))
        grad['wout'] = tf.matmul(self.h_2, tf.transpose(grad['z'], [0, 2, 1]))
        grad['bout'] = grad['z']
        grad['h2'] = tf.matmul(self.Wout, grad['z'])

        fn = lambda x: tf.diag(tf.squeeze(tf.cast(x > 0, tf.float32)))

        grad['zh2'] = tf.matmul(tf.map_fn(fn, self.zh2), grad['h2'])
        grad['w2'] = tf.matmul(self.h_1, tf.transpose(grad['zh2'], [0, 2, 1]))
        grad['b2'] = grad['zh2']
        grad['h1'] = tf.matmul(self.Wh2, grad['zh2'])

        grad['zh1'] = tf.matmul(tf.map_fn(fn, self.zh1), grad['h1'])
        grad['w1'] = tf.matmul(self.x, tf.transpose(grad['zh1'], [0, 2, 1]))
        grad['b1'] = grad['zh1']

        grad['Wo'] = tf.reduce_sum(grad['wout'], 0) / batch_size
        grad['Bo'] = tf.reduce_sum(grad['bout'], 0) / batch_size
        grad['W2'] = tf.reduce_sum(grad['w2'], 0) / batch_size
        grad['B2'] = tf.reduce_sum(grad['b2'], 0) / batch_size
        grad['W1'] = tf.reduce_sum(grad['w1'], 0) / batch_size
        grad['B1'] = tf.reduce_sum(grad['b1'], 0) / batch_size

        train_op = [
            weights['h1'].assign(weights['h1'] - learning_rate * grad['W1']),
            weights['h2'].assign(weights['h2'] - learning_rate * grad['W2']),
            weights['out'].assign(weights['out'] - learning_rate * grad['Wo']),
            biases['b1'].assign(biases['b1'] - learning_rate * grad['B1']),
            biases['b2'].assign(biases['b2'] - learning_rate * grad['B2']),
            biases['out'].assign(biases['out'] - learning_rate * grad['Bo'])
        ]
        return train_op

    def predict(self):
        self.expand()
        ypred = self.forward()
        out = self.score_eval(ypred)
        return out

    def one_iter(self):
        self.expand()
        y_= self.forward()
        out = self.score_eval( y_)
        op = self.backprop( y_)
        return [op, out]

    def get_next(self, X_train, Y_train):
        for i in range(0, X_train.shape[0], self.batch_size):
            batch_x = np.expand_dims(X_train[i:i + self.batch_size, :], 2)
            batch_y = np.expand_dims(Y_train[i:i + self.batch_size, :], 2)
            yield (batch_x, batch_y)

    def fit(self, x, y):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                counter = 1
                for batch_x, batch_y in self.get_next(x, y):
                    _, output = session.run(
                        self.one_iter, feed_dict={
                            self.x: batch_x,
                            self.y: batch_y
                        })
                    print("epoch: {} batch: {}    Loss: {:.2f}, Score: {:.2f}".
                          format(epoch, counter, output['loss'],
                                 output['score'] * 100))
                    counter += 1
                print("---------\n" * 2)  # end of epoch
            print('Finished Training\n')  # end of training


# training data
x_train, y_train = create_data(train_folder, train_label)

net = NN()

net.fit(x_train, y_train)

