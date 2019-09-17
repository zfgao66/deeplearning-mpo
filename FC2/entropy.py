# Calculate entanglement entropy from well-trained MPO tensors.
# Available for FC2-TTO net.

import tensorflow as tf
import os
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('tt_ranks_1',  10, 'ranks of the first tensor train')
os.environ["CUDA_VISIBLE_DEVICES"]=""
r_1 = FLAGS.tt_ranks_1
dir_log = './logs/tt_rank1_%d/epoch_295000.ckpt'%r_1
inp_mode =  [4,7,7,4]
out_mode =  [4,4,4,4]
mat_ranks =  [1,r_1,r_1,r_1,1]
mpo_mat = []
mpo_tensor_l = []
mpo_tensor_r = []
density = []
entropy = []
for i in range(4):
    mpo_mat.append(tf.get_variable('tt_scope_1/mat_core_%d'%(i+1),shape=[out_mode[i] * mat_ranks[i + 1], mat_ranks[i] * inp_mode[i]]))
saver = tf.train.Saver()

mpo_tensor_l.append(tf.reshape(tf.transpose(mpo_mat[0]), [-1,  mat_ranks[1]]))
temp = tf.reshape(mpo_mat[1],[out_mode[1],mat_ranks[2],mat_ranks[1],inp_mode[1]])
mpo_tensor_l.append(tf.reshape(tf.einsum('ij,lmjk->ilkm', mpo_tensor_l[0], temp), [-1, mat_ranks[2]]))
temp = tf.reshape(mpo_mat[2],[out_mode[2],mat_ranks[3],mat_ranks[2],inp_mode[2]])
mpo_tensor_l.append(tf.reshape(tf.einsum('ij,lmjk->ilkm', mpo_tensor_l[1], temp), [-1,mat_ranks[3]]))

mpo_tensor_r.append(tf.reshape(tf.transpose(mpo_mat[3]), [mat_ranks[3], -1]))
temp = tf.reshape(mpo_mat[2],[out_mode[2],mat_ranks[3],mat_ranks[2],inp_mode[2]])
mpo_tensor_r.append(tf.reshape(tf.einsum('ijkl,jm->kilm',temp, mpo_tensor_r[0]),[mat_ranks[2],-1]))
temp = tf.reshape(mpo_mat[1],[out_mode[1],mat_ranks[2],mat_ranks[1],inp_mode[1]])
mpo_tensor_r.append(tf.reshape(tf.einsum('ijkl,jm->kilm',temp, mpo_tensor_r[1]),[mat_ranks[1],-1]))



for i in range(3):
    density.append(tf.matmul(mpo_tensor_l[i], mpo_tensor_r[2-i]))
for i in range(3):
    s  = tf.svd(density[i],compute_uv=False)
    s = s * s
    s = s/tf.reduce_sum(s)
    num_nonzero = tf.count_nonzero(s)
    lamda = tf.zeros([tf.to_int32(num_nonzero),])
    lamda = s[:tf.to_int32(num_nonzero)]

    entropy.append(- tf.reduce_sum(lamda * tf.log(lamda)))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, dir_log)
    sess.run(entropy)
    print(entropy[0].eval(),entropy[1].eval(),entropy[2].eval())