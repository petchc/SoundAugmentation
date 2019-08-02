from __future__ import print_function

import soundfile as sf
#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
#import IPython.display as ipd  # To play sound in the notebook
import tensorflow as tf
import math
from datetime import datetime
import resampy
import time
#tf.enable_eager_execution()
from tempfile import TemporaryFile
import datetime
import sys
import six
from pathlib import Path
from scipy.io import wavfile
import torchnet.meter as meter
#from tqdm import tqdm_notebook as tqdm
import random as rd
import prepare_data
#import test_load_data
import vggish_slim
import vggish_params
import data_transformation
import Postprocessor

def test(filename):    
 
    n_y = 527
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        
        logits = vggish_slim.define_audio_slim(training=False)     
        #embeddings = define_vggish_slim(False)
       
        # Define a shallow classification model and associated training ops on top
        # of VGGish.
        
        with tf.variable_scope('mymodel'):
        
          # model for training
         
          predict = tf.sigmoid(logits, name='prediction')
            
          # Add training ops.
          with tf.variable_scope('train'):
            global_step = tf.Variable(
                0, name='global_step', trainable=False,  
                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                             tf.GraphKeys.GLOBAL_STEP])

            # Labels are assumed to be fed as a batch multi-hot vectors, with
            # a 1 in the position of each positive class label, and 0 elsewhere.
            labels = tf.placeholder(
                tf.float32, shape=(None, n_y ), name='labels')

            # Cross-entropy label loss.
            #xent = tf.nn.sigmoid_cross_entropy_with_logits(
            #    logits=logits, labels=labels, name='xent')
            #loss = tf.reduce_mean(xent, name='loss_op')
            #tf.summary.scalar('loss', loss)
                   
            #calculate the accuracy
            #correct_prediction = tf.equal(tf.round(predict), labels,name='test_eq')  
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='acc_op')
            
            # We use the same optimizer and hyperparameters as used to train VGGish.
            #optimizer = tf.train.AdamOptimizer(
            #    learning_rate=LEARNING_RATE,
            #    epsilon=ADAM_EPSILON)
            #optimizer.minimize(loss, global_step=global_step, name='train_op')

        # Initialize all variables in the model, and then load the pre-trained
        # VGGish checkpoint.
        sess.run(tf.global_variables_initializer())  
        #vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        
        
        saver = tf.train.Saver()
        
        #load_vggish_slim_checkpoint(sess, "./vggish_ckpt/vggish_model.ckpt")
        saver.restore(sess, "./tmp/model_epoch_8.ckpt")
        #save_path = saver.save(sess, "./tmp/model_epoch_%i.ckpt" % (epoch+1))
        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(
            'audio/audio_input_features:0')
        #labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        
        #global_step_tensor = sess.graph.get_tensor_by_name(
        #    'mymodel/train/global_step:0')
        
        #embedding_tensor = sess.graph.get_tensor_by_name(
        #    'vggish/embedding:0')
        
        #eq_tensor = sess.graph.get_tensor_by_name('mymodel/train/test_eq:0')
        
        #loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
    
        
        #train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')
        
        #accuracy_tensor = sess.graph.get_tensor_by_name('mymodel/train/acc_op:0')
    
        
        prediction_tensor = sess.graph.get_tensor_by_name('mymodel/prediction:0')
             
       
        graph = tf.Graph()
        with graph.as_default():
            vggish_slim.define_vggish_slim(training=False)

            #sess_config = tf.ConfigProto(allow_soft_placement=True)
            #sess_config.gpu_options.allow_growth = True
            sess_ext = tf.compat.v1.Session(graph=graph)
            vggish_slim.load_vggish_slim_checkpoint(sess_ext, "./vggish_ckpt/vggish_model.ckpt")

            # use the self.sess to init others
            
            input_tensor = graph.get_tensor_by_name('vggish/input_features:0')
            output_tensor = graph.get_tensor_by_name('vggish/embedding:0')
            pproc = Postprocessor.Postprocessor("./vggish_ckpt/vggish_pca_params.npz")    
            
            
            num_sound_train = 0
        #for epoch in tqdm(range(num_epochs)):
            
        print('\n###################')
        print('#  Testing loop  #')
        print('###################')
            
        data, sampleratde = sf.read(filename)
        #data, sampleratde = sf.read('C:\\Users\\pthad\\Desktop\\test3.wav')


        wave_array_example_pre = data_transformation.waveform_to_examples(data,sampleratde,display=0)
        
        [embedding_batch] = sess_ext.run([output_tensor],
                feed_dict={input_tensor: wave_array_example_pre})
        
        wave_arrays = pproc.postprocess(embedding_batch)

        pred_test_restore = sess.run(prediction_tensor, feed_dict={features_tensor: wave_arrays})
            
        
    return pred_test_restore

if __name__ == '__main__':
        csv_dir = '/export/home/2368985c/MSc_Project_Sound_Augmentation/csv_file/'
        column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["display_name"])
        label_columns = column_df.display_name.values

        #output_tensor_test = test('/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_eval_tmp/-/0/P/-0p7hKXZ1ww_30000_40000.flac')
        output_tensor_test = test('/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_eval_tmp/_/I/K/_IkLUOsNHAA_0_10000.flac')
        #print('target:',np.where(labels_train[0]==1))
        '''
        pred_result = np.zeros_like(output_tensor_test[9])
        for i in range(len(output_tensor_test)):
            pred_result += output_tensor_test[i]/len(output_tensor_test)
        print('predicted:',np.where(np.round(pred_result)==1))

        
        for i in range(len(output_tensor_test)):
            print('-------------------------------------------')
            print('sec:',i ,' is_[0]: ',np.where(np.round(output_tensor_test[i][433])==1))
            print('sec:',i,' is_[451]: ',np.where(np.round(output_tensor_test[i][451])==1))
            print('prob[0]',output_tensor_test[i][433])
            print('prob[451]',output_tensor_test[i][451])
            print('-------------------------------------------')
        '''


        #threshold = 0.4
        #print('target:',np.where(labels_train[0]==1))
        #pred_result = np.zeros_like(output_tensor_test[9])
        np.set_printoptions(precision=4)
        #top10_labels_index = output_tensor_test[i].argsort()[-10:][::-1]
        for i in range(len(output_tensor_test)):
            print('Second: ',i)
            print('Top 10 prob labels name : ',label_columns[output_tensor_test[i].argsort()[-10:][::-1]])
            print('Top 10 prob labels id   : ',output_tensor_test[i].argsort()[-10:][::-1])
            print('Raw probability values  : ',output_tensor_test[i][output_tensor_test[i].argsort()[-10:][::-1]])
            #print('Raw probability value: {:.3F}'.format(output_tensor_test[i][output_tensor_test[i].argsort()[-10:][::-1]]))
            print('-------------------------------------------')
        #print('predicted with threshold 0.4:',np.where(np.greater(pred_result,threshold).astype(float)==1.0))
        #print('sorted top 5:',pred_result.argsort()[-10:][::-1])   
        #print(pred_result[pred_result.argsort()[-10:][::-1]]) 

        #threshold = 0.3
        #print('target:',np.where(labels_train[0]==1))
        #pred_result = np.zeros_like(output_tensor_test[9])
        #for i in range(len(output_tensor_test)):
        #    pred_result += output_tensor_test[i]/len(output_tensor_test)
        #print('predicted with threshold 0.3:',np.where(np.greater(pred_result,threshold).astype(float)==1.0))
        

        #threshold = 0.1
        #print('target:',np.where(labels_train[0]==1))
        #pred_result = np.zeros_like(output_tensor_test[9])
        #for i in range(len(output_tensor_test)):
        #    pred_result += output_tensor_test[i]/len(output_tensor_test)
        #print('predicted with threshold 0.1:',np.where(np.greater(pred_result,threshold).astype(float)==1.0))    
        '''
        threshold = 0.4
        #print('target:',np.where(labels_train[0]==1))
        pred_result = np.zeros_like(output_tensor_test[9])
        for i in range(len(output_tensor_test)):
            pred_result += output_tensor_test[i]/len(output_tensor_test)
        print('predicted with threshold 0.4:',np.where(np.greater(pred_result,threshold).astype(float)==1.0))
        print('sorted top 5:',pred_result.argsort()[-10:][::-1])   
        print(pred_result[pred_result.argsort()[-10:][::-1]]) 

        threshold = 0.3
        #print('target:',np.where(labels_train[0]==1))
        #pred_result = np.zeros_like(output_tensor_test[9])
        #for i in range(len(output_tensor_test)):
        #    pred_result += output_tensor_test[i]/len(output_tensor_test)
        print('predicted with threshold 0.3:',np.where(np.greater(pred_result,threshold).astype(float)==1.0))
        

        threshold = 0.1
        #print('target:',np.where(labels_train[0]==1))
        #pred_result = np.zeros_like(output_tensor_test[9])
        #for i in range(len(output_tensor_test)):
        #    pred_result += output_tensor_test[i]/len(output_tensor_test)
        print('predicted with threshold 0.1:',np.where(np.greater(pred_result,threshold).astype(float)==1.0))
        '''
