# -*- coding: utf-8 -*-
"""
@author: zfgao
"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference
from hyper_parameters import *
#set the network paramters
BATCH_SIZE=FLAGS.batch_size
TRAINING_STEPS=FLAGS.global_step

def mnist(inp):
    x=tf.placeholder(tf.float32,[None,inference.input_node],name='x-input')
    y_=tf.placeholder(tf.float32,[None,inference.output_node],name='y-input')
    y=inference.inference(x)
    global_step=tf.Variable(0,trainable=False)
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    loss=tf.reduce_mean(ce)
    train_steps=tf.train.GradientDescentOptimizer(0.01).minimize(loss,global_step=global_step)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        writer=tf.summary.FileWriter("logs/", sess.graph)
        best_pred=0
        for i in range(TRAINING_STEPS):
            xs,ys = inp.train.next_batch(BATCH_SIZE)
            _,loss_value,step = sess.run([train_steps,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000 == 0:
                accuracy_score=sess.run(accuracy,feed_dict={x:inp.test.images,y_:inp.test.labels})
                if (best_pred<accuracy_score):
                    best_pred = accuracy_score
                print("After %s trainning step(s),test accuracy=%g" %(step,accuracy_score))
        print("After %s trainning step(s),best_pred=%g" %(step,best_pred))
def main(argv=None):
    inp=input_data.read_data_sets("./data/",one_hot=True)
    mnist(inp)
    
if __name__=='__main__':
    tf.app.run()
    
    


