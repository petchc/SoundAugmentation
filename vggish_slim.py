import tensorflow as tf
import vggish_params
#from contextlib import contextmanager
#import astroid

slim = tf.contrib.slim

def define_vggish_slim(training=False):
 """
      Defines the VGGish TensorFlow model.
      All ops are created in the current default graph, under the scope 'vggish/'.
      The input is a placeholder named 'vggish/input_features' of type float32 and
      shape [batch_size, num_frames, num_bands] where batch_size is variable and
      num_frames and num_bands are constants, and [num_frames, num_bands] represents
      a log-mel-scale spectrogram patch covering num_bands frequency bands and
      num_frames time frames (where each frame step is usually 10ms). This is
      produced by computing the stabilized log(mel-spectrogram + params.LOG_OFFSET).
      The output is an op named 'vggish/embedding' which produces the activations of
      a 128-D embedding layer, which is usually the penultimate layer when used as
      part of a full model with a final classifier layer.
      Args:
        training: If true, all parameters are marked trainable.
      Returns:
        The op 'vggish/embeddings'.
 """
 # Defaults:
 # - All weights are initialized to N(0, INIT_STDDEV).
 # - All biases are initialized to 0.
 # - All activations are ReLU.
 # - All convolutions are 3x3 with stride 1 and SAME padding.
 # - All max-pools are 2x2 with stride 2 and SAME padding.
 with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(
                          stddev=vggish_params.INIT_STDDEV,seed=tf.compat.v1.set_random_seed(41)),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      trainable=training), \
       slim.arg_scope([slim.conv2d],
                      kernel_size=[3, 3], stride=1, padding='SAME'), \
       slim.arg_scope([slim.max_pool2d],
                      kernel_size=[2, 2], stride=2, padding='SAME'), \
       tf.compat.v1.variable_scope ('vggish'):
    # Input: a batch of 2-D log-mel-spectrogram patches.
    features = tf.compat.v1.placeholder(
        tf.float32, shape=(None, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS),
        name='input_features')
    # Reshape to 4-D so that we can convolve a batch with conv2d().
    net = tf.reshape(features, [-1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1])

    # The VGG stack of alternating convolutions and max-pools.
    net = slim.conv2d(net, 64, scope='conv1')
    net = slim.max_pool2d(net, scope='pool1')
    #net = slim.batch_norm(net)
    net = slim.conv2d(net, 128, scope='conv2')
    net = slim.max_pool2d(net, scope='pool2')
    #net = slim.batch_norm(net)
    net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
    net = slim.max_pool2d(net, scope='pool3')
    #net = slim.batch_norm(net)
    net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
    net = slim.max_pool2d(net, scope='pool4')
    #net = slim.batch_norm(net)

    # Flatten before entering fully-connected layers
    net = slim.flatten(net)
    net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
    # The embedding layer.

    net = slim.fully_connected(net, vggish_params.EMBEDDING_SIZE, scope='fc2')
     
    return tf.identity(net, name='embedding')

def load_vggish_slim_checkpoint(session, checkpoint_path):
  """Loads a pre-trained VGGish-compatible checkpoint.
  This function can be used as an initialization function (referred to as
  init_fn in TensorFlow documentation) which is called in a Session after
  initializating all variables. When used as an init_fn, this will load
  a pre-trained checkpoint that is compatible with the VGGish model
  definition. Only variables defined by VGGish will be loaded.
  Args:
    session: an active TensorFlow session.
    checkpoint_path: path to a file containing a checkpoint that is
      compatible with the VGGish model definition.
  """
  # Get the list of names of all VGGish variables that exist in
  # the checkpoint (i.e., all inference-mode VGGish variables).
  with tf.Graph().as_default():
    define_vggish_slim(training=False)
    vggish_var_names = [v.name for v in tf.compat.v1.global_variables()]

  # Get the list of all currently existing variables that match
  # the list of variable names we just computed.
  vggish_vars = [v for v in tf.compat.v1.global_variables() if v.name in vggish_var_names]

  # Use a Saver to restore just the variables selected above.
  saver = tf.compat.v1.train.Saver(vggish_vars, name='vggish_load_pretrained',
                         write_version=1)
  saver.restore(session, checkpoint_path)


def define_audio_slim(training=False,is_reuse=None):

 """
      Defines the VGGish TensorFlow model.
      All ops are created in the current default graph, under the scope 'vggish/'.
      The input is a placeholder named 'vggish/input_features' of type float32 and
      shape [batch_size, num_frames, num_bands] where batch_size is variable and
      num_frames and num_bands are constants, and [num_frames, num_bands] represents
      a log-mel-scale spectrogram patch covering num_bands frequency bands and
      num_frames time frames (where each frame step is usually 10ms). This is
      produced by computing the stabilized log(mel-spectrogram + params.LOG_OFFSET).
      The output is an op named 'vggish/embedding' which produces the activations of
      a 128-D embedding layer, which is usually the penultimate layer when used as
      part of a full model with a final classifier layer.
      Args:
        training: If true, all parameters are marked trainable.
      Returns:
        The op 'vggish/embeddings'.
 """
 # Defaults:
 # - All weights are initialized to N(0, INIT_STDDEV).
 # - All biases are initialized to 0.
 # - All activations are ReLU.
 # - All convolutions are 3x3 with stride 1 and SAME padding.
 # - All max-pools are 2x2 with stride 2 and SAME padding.

 with slim.arg_scope([slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(
                          stddev=vggish_params.INIT_STDDEV,seed=tf.compat.v1.set_random_seed(41)),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      reuse=is_reuse,
                      #activation_fn=tf.nn.tanh,
                      trainable=training), \
        tf.compat.v1.variable_scope('audio'):
        
    # Input: a batch of 2-D log-mel-spectrogram patches.
        vgg_input = tf.compat.v1.placeholder(
            tf.float32, shape=(None, vggish_params.EMBEDDING_SIZE),name='audio_input_features') 
        
        
        num_units = 1024
        fc = slim.fully_connected(vgg_input, num_units,scope='f1') 
        #net = slim.batch_norm(net,scope='norm1',is_training=training,reuse=is_reuse) 
        #net = slim.dropout(net, 0.5,scope='drop1',is_training=training,seed=1)
        
        #net = slim.fully_connected(net, num_units,scope='f2') 
        #net = slim.batch_norm(net,scope='norm2',is_training=training,reuse=is_reuse) 
        #net = slim.dropout(net, 0.5,scope='drop2',is_training=training,seed=1)
        
        
        #net = slim.fully_connected(net, num_units,scope='f3') 
        #net = slim.batch_norm(net,scope='norm3',is_training=training,reuse=is_reuse) 
        #fc = slim.dropout(net, 0.5,scope='drop3',is_training=training,seed=1)
    
        #fc = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
        
        
        
        
        logits = slim.fully_connected(
              fc, vggish_params._NUM_CLASSES, activation_fn=None, scope='logits')
           
        #logits = slim.fully_connected(
        #      fc, 3, activation_fn=None, scope='logits')

        return logits


    