import tensorflow as tf
import numpy as np
from .auxx import get_var_wrap

def tr(inp,
       inp_modes,
       out_modes,
       mat_ranks,
       cores_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
       cores_regularizer=None,
       biases_initializer=tf.zeros_initializer,
       biases_regularizer=None,
       trainable=True,
       cpu_variables=False,
       scope=None):
    """ tr-layer (tr-matrix by full tensor product)
    Args:
        inp: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes
        out_modes: output tensor modes
        mat_ranks: tr-matrix ranks
        cores_initializer: cores init function, could be a list of functions for specifying different function for each core
        cores_regularizer: cores regularizer function, could be a list of functions for specifying different function for each core
        biases_initializer: biases init function (if None then no biases will be used)
        biases_regularizer: biases regularizer function        
        trainable: trainable variables flag, bool
        cpu_variables: cpu variables flag, bool
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, prod(out_modes)]
    """
    with tf.variable_scope(scope):
        dim = inp_modes.size
        
        mat_cores = []
        
        for i in range(dim):
            if type(cores_initializer) == list:
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer
            
            if type(cores_regularizer) == list:
                creg = cores_regularizer[i]
            else:
                creg = cores_regularizer
            # mat_cores_i[(mi,ri+1),(ri,ni)]
            mat_cores.append(get_var_wrap('mat_core_%d' % (i + 1),
                                          shape=[out_modes[i] * mat_ranks[(i + 1)%dim], mat_ranks[i] * inp_modes[i]],
                                          initializer=cinit,
                                          regularizer=creg,
                                          trainable=trainable,
                                          cpu_variable=cpu_variables))
            
        # inp(b,n0,n1,...,nL-1)
        out = tf.reshape(inp, [-1, np.prod(inp_modes)]) # out[b,(n0,n1,...,nL-1)]
        out = tf.transpose(out, [1, 0]) # out[(n0,n1,...,nL-1),b]

        out = tf.reshape(out, [inp_modes[0], -1]) # out[n0,(n1,...,nL-1,b)]
        mat_cores_0 = tf.reshape(mat_cores[0], [-1, inp_modes[0]]) # mat_cores_0[(m0,r1,r0),n0]
        out = tf.matmul(mat_cores_0, out) # out[(m0,r1,r0),(n1,...,nL-1,b)]

        di = np.prod(inp_modes[1:]) # di = inp_modes[1]*inp_modes[2]*...*inp_modes[L-1]

        # out[m0,r1,r0,(n1,...,nL-1),b]
        out = tf.reshape(out, [out_modes[0], mat_ranks[1], mat_ranks[0], di, -1])
        out = tf.transpose(out, [1, 3, 2, 4, 0]) # out[r1,(n1,...,nL-1),r0,b,m0]
        
        for i in range(1, dim-1):
            # out[(ri,ni), (ni+1,...,nL-1,r0,b,m0,...,mi-1)]
            out = tf.reshape(out, [mat_ranks[i] * inp_modes[i], -1])
            # out[(mi,ri+1),(ni+1,...,nL-1,r0,b,m0,...,mi-1)] =
            # mat_cores_i[(mi,ri+1),(ri,ni)]*out[(ri,ni), (ni+1,...,nL-1,r0,b,m0,...,mi-1)]
            out = tf.matmul(mat_cores[i], out)
            # out[mi,(ri+1,ni+1,...,nL-1,r0,b,m0,...,mi-1)]
            out = tf.reshape(out, [out_modes[i], -1])
            # out[(ri+1,ni+1,...,nL-1,r0,b,m0,...,mi-1),mi]
            out = tf.transpose(out, [1, 0])        

        # after the loop: out[(rL-1,nL-1,r0,b,m0,...,mL-3),mL-2]
        # out[(rL-1,nL-1,r0),(b,m0,...,mL-3,mL-2)]
        out = tf.reshape(out, [mat_ranks[dim-1]*inp_modes[dim-1]*mat_ranks[0], -1])
        # mat_cores_L-1[mL-1,r0,(rL-1,nL-1)]
        mat_cores_L_1 = tf.reshape(mat_cores[dim-1], [out_modes[dim-1], mat_ranks[0], mat_ranks[dim-1]*inp_modes[dim-1]])
        mat_cores_L_1 = tf.transpose(mat_cores_L_1, [0, 2, 1]) # mat_cores_L-1[mL-1,(rL-1,nL-1),r0]
        # mat_cores_L-1[mL-1,(rL-1,nL-1,r0)]
        mat_cores_L_1 = tf.reshape(mat_cores_L_1, [out_modes[dim-1], mat_ranks[dim-1]*inp_modes[dim-1]*mat_ranks[0]])
        # out[mL-1,(b,m0,...,mL-3,mL-2)] =
        # mat_cores_L-1[mL-1,(rL-1,nL-1,r0)] * out[(rL-1,nL-1,r0),(b,m0,...,mL-3,mL-2)]
        out = tf.matmul(mat_cores_L_1, out)
        # out[(b,m0,...,mL-3,mL-2),mL-1]
        out = tf.transpose(out, [1, 0])

        if biases_initializer is not None:
            
            biases = get_var_wrap('biases',
                                  shape=[np.prod(out_modes)],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)
                                                                    
            out = tf.add(tf.reshape(out, [-1, np.prod(out_modes)]), biases, name="out")
        else:
            out = tf.reshape(out, [-1, np.prod(out_modes)], name="out")

    return out
