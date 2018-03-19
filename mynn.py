import os
import pickle

import numpy as np

import cv2
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import plotly
import plotly.plotly as py
from plotly.graph_objs import *

plotly.tools.set_credentials_file(username='asrafulashiq', api_key='vFaCiOh0zfJnFaA2mcnE')

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


# set hyper-parameters
learning_rate = 0.1
epochs = 5
batch_size = 50

# network parameters
n_input = 784
n_classes = 10
n_hidden_1 = 100
n_hidden_2 = 100

# store weights and bias

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=.1))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1, 1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2, 1])),
    'out': tf.Variable(tf.zeros([n_classes, 1]))
}


def loss_fn(out_layer, y):
    loss = 0.5 * \
        tf.reduce_sum(tf.squared_difference(out_layer, y)) / batch_size
    return loss


def accuracy(pred, Y):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


def soft_max(x):
    ex = tf.exp(x - tf.reduce_max(x))
    den = tf.reduce_sum(ex, 1)
    D = tf.tile(tf.expand_dims(den, 1), [1, n_classes, 1])
    return ex / D

def relu(x):
    return tf.maximum(x, 0)

def get_next(X, Y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        batch_x = np.expand_dims(X[i:i + batch_size, :], 2)
        batch_y = np.expand_dims(Y[i:i + batch_size, :], 2)
        yield batch_x, batch_y


def get_report(Ytrue, Ypred):
    N = Ytrue.shape[0]
    yt = np.zeros(N)
    yp = np.zeros(N)
    for i in range(N):
        yt[i] = np.argmax(Ytrue[i])
        yp[i] = np.argmax(Ypred[i])
    report = classification_report(yt, yp)
    cm = confusion_matrix(yt, yp)
    return report, cm

    

''' CREATE COMPUTATION GRAPH  '''
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
h_1 = relu(zh1)
zh2 = tf.matmul(tf.transpose(Wh2, [0, 2, 1]), h_1) + Bb2
h_2 = relu(zh2)
y_ = soft_max(tf.matmul(tf.transpose(Wout, [0, 2, 1]), h_2) + Bout)

loss = loss_fn(y, y_)

#tf.summary.scalar('loss', loss)

score = accuracy(y_, y)
out = {'loss': loss, 'score': score, 'pred':y_}
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
    biases['b1'].assign(biases['b1'] - learning_rate * grad['B1']),
    weights['h2'].assign(weights['h2'] - learning_rate * grad['W2']),
    biases['b2'].assign(biases['b2'] - learning_rate * grad['B2']),
    weights['out'].assign(weights['out'] - learning_rate * grad['Wo']),
    biases['out'].assign(biases['out'] - learning_rate * grad['Bo'])
]



''' Training '''
session = tf.Session()
#train_writer = tf.summary.FileWriter('./logs/1/train ', session.graph)
session.run(tf.global_variables_initializer())

X_train, Y_train = create_data(train_folder, train_label)

loss_array = []
score_array = []
for epoch in range(epochs):
    counter = 1
    for batch_x, batch_y in get_next(X_train, Y_train, batch_size):
        #merge = tf.summary.merge_all()
        op, o = session.run(
            [train_op, out], feed_dict={
                x: batch_x,
                y: batch_y
            })
        print(
            "epoch: {:<4d} batch: {:<5d} Loss:{:4.2f}   Score:{:4.2f}%".format(
                epoch, counter, o['loss'], o['score'] * 100))
        #train_writer.add_summary(summary, counter)
        counter += 1
        loss_array.append(o['loss'])
        score_array.append(o['score'])

    print('-----------\n')  # end of an epoch
print("Finished training\n\n\n")
loss_data = Scatter(y=loss_array)
py.plot([loss_data], filename = 'loss')
score_data = Scatter(y=score_array)
py.plot([score_data], filename = 'accuracy')


# save weight
Theta = op
filehandler = open("nn_parameters.txt", "wb")
pickle.dump(Theta, filehandler, protocol=2)
filehandler.close()



''' TEST '''
# test prediction graph
X_test, Y_test = create_data(test_folder, test_label)
batch_size = X_test.shape[0]

x = tf.placeholder(tf.float32, [None, 784, 1])
y = tf.placeholder(tf.float32, [None, 10, 1])

Wh1 = tf.tile(tf.expand_dims(weights['h1'], 0), [batch_size, 1, 1])
Wh2 = tf.tile(tf.expand_dims(weights['h2'], 0), [batch_size, 1, 1])
Wout = tf.tile(tf.expand_dims(weights['out'], 0), [batch_size, 1, 1])
Bb1 = tf.tile(tf.expand_dims(biases['b1'], 0), [batch_size, 1, 1])
Bb2 = tf.tile(tf.expand_dims(biases['b2'], 0), [batch_size, 1, 1])
Bout = tf.tile(tf.expand_dims(biases['out'], 0), [batch_size, 1, 1])

zh1 = tf.matmul(tf.transpose(Wh1, [0, 2, 1]), x) + Bb1
h_1 = relu(zh1)
zh2 = tf.matmul(tf.transpose(Wh2, [0, 2, 1]), h_1) + Bb2
h_2 = relu(zh2)
y_ = soft_max(tf.matmul(tf.transpose(Wout, [0, 2, 1]), h_2) + Bout)

loss = loss_fn(y, y_)
score = accuracy(y_, y)
out = {'loss': loss, 'score': score, 'pred': y_}

X_test = np.expand_dims(X_test, 2)
Y_test = np.expand_dims(Y_test, 2)

output = session.run(out, feed_dict={x: X_test, y: Y_test})
print('\nTEST\n---------------')
print('Evaluation:')

print('Accuracy : {:<5.2f}%'.format(output['score'] * 100))

print()

r, cm = get_report(Y_test, output['pred'])

print('Confusion matrix:')
print(cm)
print()
print('Classification Report:')
print(r)




