import os

import numpy as np

import cv2
import tensorflow as tf
from tensorflow.python import debug as tf_debug

test_folder = 'test_data'
train_folder = 'train_data'

test_label = 'labels/test_label.txt'
train_label = 'labels/train_label.txt'


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.reshape(image, [-1]) / 255.
    label = tf.one_hot(label, 10)
    return image, label


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


def load_data_iterator(dataset, batch_size, shuffle=False):
    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    iterator = dataset.make_one_shot_iterator()
    return iterator


# set hyper-parameters
learning_rate = 0.1
epochs = 3
batch_size = 100

# network parameters
n_input = 784
n_classes = 10
n_hidden_1 = 100
n_hidden_2 = 100

# import data
#training_dataset = create_data(train_folder, train_label)
#training_iterator = load_data_iterator(training_dataset, batch_size)
#print("Dataset Loaded")
X_train, Y_train = create_data(train_folder, train_label)

# store weights and bias

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=.1))
}

# biases = {
# 'b1': tf.Variable(tf.zeros([n_hidden_1])),
# 'b2': tf.Variable(tf.zeros([n_hidden_2])),
# 'out': tf.Variable(tf.zeros([n_classes]))
# }
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1, 1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2, 1])),
    'out': tf.Variable(tf.zeros([n_classes, 1]))
}

# forward propagate


def forward(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(
        tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.nn.softmax(
        tf.matmul(layer_2, weights['out']) + biases['out'])
    return out_layer

# define loss

def loss_fn(out_layer, y):
    loss = 0.5 * tf.reduce_sum(tf.squared_difference(out_layer, y)) / batch_size
    #loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
    return loss


def optimizer(loss, learning_rate):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_op


def accuracy(pred, Y):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy



def soft_max(x):
    ex = tf.exp(x - tf.reduce_max(x))
    den = tf.reduce_sum(ex, 1)
    D = tf.tile(tf.expand_dims(den, 1), [1, n_classes, 1])
    return ex / D

'''
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

out = forward(X)
loss = loss_fn(out, Y)
train_op = optimizer(loss, learning_rate)
score = accuracy(out, Y)
'''

''' input output '''
x = tf.placeholder(tf.float32, [None, 784, 1])
y = tf.placeholder(tf.float32, [None, 10, 1])

''' forward propagate '''
Wh1 = tf.tile(tf.expand_dims(weights['h1'], 0), [batch_size, 1, 1])
Wh2 = tf.tile(tf.expand_dims(weights['h2'], 0), [batch_size, 1, 1])
Wout = tf.tile(tf.expand_dims(weights['out'], 0), [batch_size, 1, 1])
Bb1 = tf.tile(tf.expand_dims(biases['b1'], 0), [batch_size, 1, 1])
Bb2 = tf.tile(tf.expand_dims(biases['b2'], 0), [batch_size, 1, 1])
Bout = tf.tile(tf.expand_dims(biases['out'], 0), [batch_size, 1, 1])

zh1 = tf.matmul(tf.transpose(Wh1, [0, 2, 1]), x) + Bb1
h_1 = tf.nn.relu(zh1)
zh2 = tf.matmul(tf.transpose(Wh2, [0, 2, 1]), h_1) + Bb2
h_2 = tf.nn.relu(zh2)
y_ = soft_max(tf.matmul(tf.transpose(Wout, [0, 2, 1]), h_2) + Bout)

loss = loss_fn(y, y_)
score = accuracy(y_, y)

out = {
    'y_': y_,
    'zh1': zh1,
    'zh2': zh2,
    'h1': h_1,
    'h2': h_2,
    'loss': loss,
    'score': score,
    'Wh1': Wh1, 'Bout':Bout, 'Wout':Wout,
    'Bb1': Bb1, 
    'w':weights, 'b':biases
}
''' back propagation '''
grad = {}
grad['y'] = -(y - y_)

I = tf.tile(tf.expand_dims(tf.eye(n_classes), 0), [batch_size, 1, 1])
grad['z'] = tf.matmul(I - y_, tf.multiply(y_, grad['y']))
grad['wout'] = tf.matmul(h_2, tf.transpose(grad['z'], [0, 2, 1]))
grad['bout'] = grad['z']
grad['h2'] = tf.matmul(Wout, grad['z'])


def fn(x):
    return tf.diag(tf.squeeze(tf.cast(x > 0, tf.float32)))


grad['zh2'] = tf.matmul(tf.map_fn(fn, zh2), grad['h2'])
grad['w2'] = tf.matmul(h_1, tf.transpose(grad['zh2'], [0, 2, 1]))
grad['b2'] = grad['zh2']
grad['h1'] = tf.matmul(Wh2, grad['zh2'])

grad['zh1'] = tf.matmul(tf.map_fn(fn, zh1), grad['h1'])
grad['w1'] = tf.matmul(x, tf.transpose(grad['zh1'], [0, 2, 1]))
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

debug = False

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    #session = tf_debug.LocalCLIDebugWrapperSession(session)

    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            batch_x = np.expand_dims(X_train[i:i + batch_size, :], 2)
            batch_y = np.expand_dims(Y_train[i:i + batch_size, :], 2)

            _, loss_, score_, g, o = session.run(
                [train_op, loss, score, grad, out],
                feed_dict={
                    x: batch_x,
                    y: batch_y
                })
            print("epoch: {} batch: {}    Loss: {:.2f}, Score: {:.2f}".format(
                epoch, i, loss_, score_ * 100))
            # print(bo)
            # break

print("Finished training\n\n\n")


# test
X_test, Y_test = create_data(test_folder, test_label)



