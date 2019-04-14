# -*- coding: utf-8 -*-
"""
@author: zfgao
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference
from hyperprameter import *

BATCH_SIZE=FLAGS.batch_size
TRAINING_STEPS=FLAGS.global_step
LEARNING_RATE_BASE=FLAGS.LEARNING_RATE_BASE
LEARNING_RATE_DECAY=FLAGS.LEARNING_RATE_DECAY
REGULARIZER_RATE=FLAGS.REGULARIZER_RATE
MOVING_DECAY=0.99
#seed =12345
#tf.set_random_seed(seed)

def mnist(inp):
    x=tf.placeholder(tf.float32,[None,inference.input_node],name='x-input')
    y_=tf.placeholder(tf.float32,[None,inference.output_node],name='y-input')
    y=inference.inference(x)
    global_step=tf.Variable(0,trainable=False)
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    loss=tf.reduce_mean(ce)
    loss += tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * REGULARIZER_RATE

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,
                                             global_step,
                                             inp.train.num_examples/BATCH_SIZE,
                                             LEARNING_RATE_DECAY)
    train_steps=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        best_acc = 0
        for i in range(TRAINING_STEPS):
            xs,ys = inp.train.next_batch(BATCH_SIZE)
            _,step,lr = sess.run([train_steps,global_step,learning_rate],feed_dict={x:xs,y_:ys})
            if i%1000 == 0:
                accuracy_score = sess.run(accuracy, feed_dict={x:inp.test.images,y_:inp.test.labels})
                print('step={},lr={}'.format(step,lr))
                if best_acc< accuracy_score:
                    best_acc = accuracy_score
                print('Accuracy at step %s: %s' % (i, accuracy_score))
        accuracy_score=sess.run(accuracy,feed_dict={x:inp.test.images,y_:inp.test.labels})
        print("After %s trainning step(s),best accuracy=%g" %(step,best_acc))



def main(argv=None):
    inp=input_data.read_data_sets("./data/",validation_size=0,one_hot=True)
    mnist(inp)

if __name__=='__main__':
    tf.app.run()
