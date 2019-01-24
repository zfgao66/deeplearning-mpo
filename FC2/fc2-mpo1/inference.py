
import numpy as np
import tensorflow as tf
from hyper_parameters import *
import tt

r_1 = FLAGS.tt_ranks_1
r_2 = FLAGS.tt_ranks_2
#input_node=FLAGS.input_unit
#output_node=FLAGS.output_unit

#TTO_layer1
inp_modes1 =  [4,7,7,4]          
out_modes1 =  [4,4,4,4]       
mat_rank1  =  [1,r_1,r_1,r_1,1]

#TTO_layer2
inp_modes2 = [4,4,4,4]     
out_modes2 = [1,10,1,1]        
mat_rank2 =  [1,r_2,r_2,r_2,1]


def inference(inputs):
    inputs = tt.tto(inputs,
                    np.array(inp_modes1,dtype=np.int32),
                    np.array(out_modes1,dtype=np.int32),
                    np.array(mat_rank1,dtype=np.int32),
                    biases_initializer=None,
                    scope='tt_1')
    inputs = tf.nn.relu(inputs)
    inputs = tt.tto(inputs,
                    np.array(inp_modes2,dtype=np.int32),
                    np.array(out_modes2,dtype=np.int32),
                    np.array(mat_rank2,dtype=np.int32),
                    biases_initializer=None,
                    scope='tt_2')
    
    return inputs 
        
        
        
                        
    
