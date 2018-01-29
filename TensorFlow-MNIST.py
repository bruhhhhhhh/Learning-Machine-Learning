import tensorflow as tf, numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

#model parameters
n_inputs = 784
n_nodes_hl1 = 526
n_nodes_hl2 = 268
n_classes = 10
batch_size = 10

#model creation
x = tf.placeholder('float', shape=(None, 784))
y = tf.placeholder('float')


net = x
net = tf.layers.dense(inputs=net, name='h0', units=n_nodes_hl1, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='h1', units=n_nodes_hl2, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='o', units=n_classes, activation=None)
output = net


def train_neural_network():
    global y
    global output
    prediction = output
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(cost)

    n_epochs =  4

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(n_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x1, y1 = mnist.train.next_batch(batch_size)
                
                _, c = sess.run([optimizer, cost], feed_dict={x:x1, y:y1})
                epoch_loss += c
            print('Epoch', epoch, ' completed. Loss = ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network()
