import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_uniform([784, 300], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([300], -1.0, 1.0))
L1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_uniform([300, 300], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([300], -1.0, 1.0))
L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_uniform([300, 10], -1.0, 1.0))
b3 = tf.Variable(tf.random_uniform([10], -1.0, 1.0))

logits = tf.matmul(L2, W3) + b3
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

ckpt_path = 'saved/model'
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    test_accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print("Test Accuracy :", sess.run(test_accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))


