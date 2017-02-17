
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# number of final output classes of numbers 0 to 9, hence 10 classes
n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x,W):
	# strides=[1,1,1,1] refers to the 4 dimensions of the movement of the
	# conv filter, which in this case is [batch size step, row step, column step, color step]
	# see: http://stackoverflow.com/questions/34642595/tensorflow-strides-argument
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

	# for ksize see: http://stackoverflow.com/questions/38601452/the-usages-of-ksize-in-tf-nn-max-pool
def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')	


def convolutional_neural_network(x):
	# [5,5,1,32] means 5x5 conv, 1 input and 32 features output???
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    # reshape input image to 28x28. 
    # why -1? see: http://stackoverflow.com/questions/34642595/tensorflow-strides-argument
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # conv and pooling layer 1
    conv1 = conv2d(x,weights['W_conv1'])
    conv1 = maxpool2d(conv1)

    # conv and pooling layer 2
    conv2 = conv2d(conv1,weights['W_conv2'])
    conv2 = maxpool2d(conv2)

    # fully connected layer
    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    output = tf.matmul(fc, weights['out'])+biases['out']
    
    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

