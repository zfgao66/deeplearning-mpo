#-*- coding: utf-8 -*-
"""
@author: zfgao

This network is normal lenet5 in MNIST dataset 
We set the lr=0.01 and use SGD optimizer
To run this program you can just input 
$ python train.py 
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference
from hyper_parameters import *

BATCH_SIZE=FLAGS.batch_size
TRAINING_STEPS=FLAGS.train_steps
lr = FLAGS.learning_rate

def mnist(inp):

    x=tf.placeholder(tf.float32,[None,inference.input_node],name='x-input')
    y_=tf.placeholder(tf.float32,[None,inference.output_node],name='y-input')

    y=inference.inference(x)
    global_step=tf.Variable(0,trainable=False)

    ce=tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
    loss=tf.reduce_mean(ce)
    tf.summary.scalar('loss',loss)

    train_steps=tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)

    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('./logs/test')                      
        for i in range(TRAINING_STEPS):
            xs,ys = inp.train.next_batch(BATCH_SIZE)
            _,loss_value,step,summary = sess.run([train_steps,loss,global_step,merged],feed_dict={x:xs,y_:ys})
            train_writer.add_summary(summary, i)
            best_pre = 0.0
            if i%1000 == 0:
                summary, accuracy_score = sess.run([merged, accuracy], feed_dict={x:inp.test.images,y_:inp.test.labels})
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, accuracy_score))
                if (best_pre <= accuracy_score):
                    best_pre = accuracy_score
        print("After %s trainning step(s),best accuracy=%g" %(step,best_pre))

def main(argv=None):
    inp=input_data.read_data_sets("./data/",one_hot=True)
    mnist(inp)
    
if __name__=='__main__':
    tf.app.run()
    
    


