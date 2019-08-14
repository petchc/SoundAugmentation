from __future__ import print_function

import soundfile as sf
import matplotlib.pyplot as plt
import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from sklearn import metrics
import math
from datetime import datetime
import resampy
import time
from tempfile import TemporaryFile
import datetime
import sys
import six
from pathlib import Path
from scipy.io import wavfile
from scipy import stats
import random as rd
import prepare_data
import vggish_slim
import vggish_params
import data_transformation
import Postprocessor

np.set_printoptions(threshold=sys.maxsize)
tf.compat.v1.set_random_seed(21)
slim = tf.contrib.slim

def train(X_train, Y_train, X_eval, Y_eval, checkpoint_dir,save_dir, num_epochs=1,minibatch_size=32,print_cost=True,augmentation=False):
   
    #print('!!!!!!!!!!!!!!!!!!!!!train augmentation: ',augmentation)
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # to keep results consistent (tensorflow seed)

    tf.compat.v1.set_random_seed(1)
    seed = 3

    loss_train=[]
    map_train=[]
    auc_train=[]
    d_prime_train=[]
    
    loss_eval_list=[]
    map_eval=[]
    auc_eval=[]
    d_prime_eval=[]
    
    m_train = X_train.shape[0]  
    n_y = Y_train.shape[1]
    
    m_eval = X_eval.shape[0]

    start = datetime.datetime.now()
    standard_normal = stats.norm()

    if(augmentation == True):
        prepare_data.create_folder('./audio_aug')
    
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        # Define VGGish.
       
        logits = vggish_slim.define_audio_slim(training=True,is_reuse=None)
        logits_eval = vggish_slim.define_audio_slim(training=False,is_reuse=True)

        # Define a shallow classification model and associated training ops on top
        # of VGGish.
        
        with tf.compat.v1.variable_scope('mymodel'):
          
          # model for training
          
          #predict = tf.sigmoid(logits, name='prediction')
          predict = tf.sigmoid(logits, name='prediction')  
          predict_eval = tf.sigmoid(logits_eval, name='prediction_eval')
            
          # Add training ops.
          with tf.compat.v1.variable_scope('train'):
            global_step = tf.Variable(
                0, name='global_step', trainable=False,  
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                             tf.compat.v1.GraphKeys.GLOBAL_STEP])

            # Labels are assumed to be fed as a batch multi-hot vectors, with
            # a 1 in the position of each positive class label, and 0 elsewhere.
            labels = tf.compat.v1.placeholder(
                tf.float32, shape=(None, n_y ), name='labels')

            # Cross-entropy label loss.
            xent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels, name='xent')
            loss = tf.reduce_mean(xent, name='loss_op')
            tf.compat.v1.summary.scalar('loss', loss)

            
            xent_eval = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_eval, labels=labels, name='xent_eval')
            loss_eval = tf.reduce_mean(xent_eval, name='loss_op_eval')
            tf.compat.v1.summary.scalar('loss_eval', loss_eval)

            # We use the same optimizer and hyperparameters as used to train VGGish.
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=vggish_params.LEARNING_RATE,
                epsilon=vggish_params.ADAM_EPSILON)
            optimizer.minimize(loss, global_step=global_step, name='train_op')


        # Initialize all variables in the model, and then load the pre-trained
       
        # VGGish checkpoint.
        sess.run(tf.compat.v1.global_variables_initializer())  
        saver = tf.compat.v1.train.Saver()
        
        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(
            'audio/audio_input_features:0')
        features_tensor_eval = sess.graph.get_tensor_by_name(
            'audio_1/audio_input_features:0')
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name(
            'mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        loss_tensor_eval = sess.graph.get_tensor_by_name('mymodel/train/loss_op_eval:0')
        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')
        prediction_tensor = sess.graph.get_tensor_by_name('mymodel/prediction:0')
        prediction_tensor_eval = sess.graph.get_tensor_by_name('mymodel/prediction_eval:0')  
        
        graph = tf.Graph()
        with graph.as_default():
            vggish_slim.define_vggish_slim(training=False)
            sess_ext = tf.compat.v1.Session(graph=graph)
            vggish_slim.load_vggish_slim_checkpoint(sess_ext, checkpoint_dir + "vggish_model.ckpt")
            input_tensor = graph.get_tensor_by_name('vggish/input_features:0')
            output_tensor = graph.get_tensor_by_name('vggish/embedding:0')
            pproc = Postprocessor.Postprocessor(checkpoint_dir + "vggish_pca_params.npz")
        
        for epoch in range(num_epochs):
            
            print('\n###################')
            print('#  Training loop  #')
            print('###################')
            
            avg_loss_train = 0.
            avg_map_train = 0.
            avg_auc_train = 0. 
            avg_d_prime_train = 0.
            step_train=0
            
            # number of minibatches of size minibatch_size in the train set
            num_minibatches_train = int(np.ceil(m_train / minibatch_size))
            seed = seed + 1
            minibatches_files_train = prepare_data.random_mini_batches_files(X_train, Y_train, minibatch_size, seed, shuffle=True)
            
            for minibatch_file in minibatches_files_train:
               
                # Select a minibatch
                (minibatch_X, minibatch_Y) =  prepare_data.get_train_data(minibatch_file,sess_ext,input_tensor,output_tensor,pproc,is_train=True,seed=seed,is_augment=augmentation)
                #print('mini_x',np.shape(minibatch_X))
                #print('mini_y',np.shape(minibatch_Y))
                
                num_steps_train ,step_loss_train,pred_tensor_train, _ = sess.run(
                  [global_step_tensor,loss_tensor,prediction_tensor, train_op],
                  feed_dict={features_tensor: minibatch_X, labels_tensor: minibatch_Y})   
                
                #print('after tf before print')

                step_auc_train = metrics.roc_auc_score(np.asarray(minibatch_Y), pred_tensor_train, average='micro')
                step_map_train = metrics.average_precision_score(np.asarray(minibatch_Y), pred_tensor_train, average='micro')
                step_d_prime_train = standard_normal.ppf(step_auc_train) * np.sqrt(2.0)
                avg_loss_train += step_loss_train / num_minibatches_train
                avg_map_train += step_map_train / num_minibatches_train
                avg_auc_train += step_auc_train / num_minibatches_train
                avg_d_prime_train += step_d_prime_train / num_minibatches_train

                if((step_train+1) % 10 == 0 or step_train+1==num_minibatches_train or step_train==0):
                    print("Epoch {}/{}, Step: {}/{} ,Loss: {:.5f}, MAP: {:.5f}, AUC: {:.5f}, d-prime: {:.5f}"  \
                    .format(epoch+1,num_epochs, \
                    step_train +1 ,num_minibatches_train,step_loss_train,step_map_train,step_auc_train,step_d_prime_train))                

                step_train += 1

                if(augmentation == True and vggish_params.AUGMENT_SAVE == False):
                    shutil.rmtree('./audio_aug')
                    prepare_data.create_folder('./audio_aug')
                    #print('delete file step')

                
            print('\n###################')
            print('# Validation loop #')
            print('###################')
            
            avg_loss_eval = 0.
            avg_map_eval = 0.
            avg_auc_eval = 0.
            avg_d_prime_eval = 0.
            
            num_minibatches_eval = int(np.ceil(m_eval / minibatch_size))

            minibatches_files_eval = prepare_data.random_mini_batches_files(X_eval, Y_eval, minibatch_size, seed, shuffle=False) 
            step_eval = 0
            
            for minibatch_file_eval in minibatches_files_eval:
                # Select a minibatch
                       
                (minibatch_X_eval, minibatch_Y_eval) =  prepare_data.get_train_data(minibatch_file_eval,sess_ext,input_tensor,output_tensor,pproc,is_train=False,seed=seed,is_augment=False)
 
                num_steps_eval ,step_loss_eval,pred_tensor_eval = sess.run(
                  [global_step_tensor, loss_tensor_eval,prediction_tensor_eval],
                  feed_dict={features_tensor_eval: minibatch_X_eval, labels_tensor: minibatch_Y_eval})  

                step_auc_eval = metrics.roc_auc_score(np.asarray(minibatch_Y_eval), pred_tensor_eval, average='micro')
                step_map_eval = metrics.average_precision_score(np.asarray(minibatch_Y_eval), pred_tensor_eval, average='micro')
                step_d_prime_eval = standard_normal.ppf(step_auc_eval) * np.sqrt(2.0)
                avg_loss_eval += step_loss_eval / num_minibatches_eval
                avg_map_eval += step_map_eval / num_minibatches_eval
                avg_auc_eval += step_auc_eval / num_minibatches_eval 
                avg_d_prime_eval += step_d_prime_eval / num_minibatches_eval
                
                if((step_eval+1) % 10 == 0 or step_eval+1==num_minibatches_eval or step_eval==0):
                    print("Epoch {}/{}, Step: {}/{} ,Loss: {:.5f}, mAP: {:.5F}, AUC: {:.5F}, d-prime: {:.5F}" \
                    .format(epoch+1,num_epochs, \
                    step_eval+1 ,num_minibatches_eval,step_loss_eval,step_map_eval,step_auc_eval,step_d_prime_eval))
                
                step_eval += 1
                  
            if print_cost == True:
                
                print("\n---Epoch %i Summary---" % (epoch+1)) 
                loss_train.append(avg_loss_train)
                map_train.append(avg_map_train)
                auc_train.append(avg_auc_train)
                d_prime_train.append(avg_d_prime_train)
                print("Training   : loss %f, mAP %f, AUC %f, d-prime %f" % (avg_loss_train,avg_map_train,avg_auc_train,avg_d_prime_train))

                
                loss_eval_list.append(avg_loss_eval)
                map_eval.append(avg_map_eval)
                auc_eval.append(avg_auc_eval)  
                d_prime_eval.append(avg_d_prime_eval)                                             
                print("Validation : loss %f, mAP %f, AUC %f, d-prime %f" % (avg_loss_eval,avg_map_eval,avg_auc_eval,avg_d_prime_eval))
                
                if((epoch+1)%2 == 0 ):
                    save_path = saver.save(sess, save_dir + "model_test_1_epoch_%i.ckpt" % (epoch+1))
                    print("Model saved in path: %s" % save_path)
                
        plt.plot(np.squeeze(loss_train), 'b', label='Training loss')
        plt.plot(np.squeeze(loss_eval_list), 'r', label='Validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('The loss of the training and validation dataset')  
        plt.legend()
        plt.xlim([0,9])
        locs, labels = plt.xticks()
        labels = [int(item)+1 for item in locs]
        plt.xticks(locs, labels)
        plt.savefig("./figures/Loss_per_epoch_test_1.png")
        plt.show() 
        plt.close()
        
        plt.plot(np.squeeze(map_train), 'b', label='Training mAP')
        plt.plot(np.squeeze(map_eval), 'r', label='Validation mAP')
        plt.ylabel('mAP')
        plt.xlabel('Epochs')
        plt.title('The mean average precision\nof the training and validation dataset')  
        plt.legend()
        plt.xlim([0,9])
        locs, labels = plt.xticks()
        labels = [int(item)+1 for item in locs]
        plt.xticks(locs, labels)
        plt.savefig("./figures/mAP_per_epoch_test_1.png")
        plt.show() 
        plt.close()
        
        plt.plot(np.squeeze(auc_train), 'b', label='Training AUC')
        plt.plot(np.squeeze(auc_eval), 'r', label='Validation AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epochs')
        plt.title('The area under the curve\nof the training and validation dataset')  
        plt.legend()
        plt.xlim([0,9])
        locs, labels = plt.xticks()
        labels = [int(item)+1 for item in locs]
        plt.xticks(locs, labels)        
        plt.savefig("./figures/AUC_per_epoch_test_1.png")
        plt.show()      
        plt.close()    

        plt.plot(np.squeeze(d_prime_train), 'b', label='Training d-prime')
        plt.plot(np.squeeze(d_prime_eval), 'r', label='Validation d-prime')
        plt.ylabel('d-prime')
        plt.xlabel('Epochs')
        plt.title('The d-prime of the training and validation dataset')  
        plt.legend()
        plt.xlim([0,9])
        locs, labels = plt.xticks()
        labels = [int(item)+1 for item in locs]
        plt.xticks(locs, labels)        
        plt.savefig("./figures/d_prime_per_epoch_test_1.png")
        plt.show()      
        plt.close()                                             
        
        end = datetime.datetime.now()
        elapsed = end - start
        print('Elapsed Time:',elapsed)

        print('\n###################')
        print('# Example testing #')
        print('###################')

        
        data, sampleratde = sf.read(Path(files_name_eval[0])) 

        wave_array_example_pre = data_transformation.waveform_to_examples(data,sampleratde,display=0)

                    
        [embedding_batch] = sess_ext.run([output_tensor],
                feed_dict={input_tensor: wave_array_example_pre})
        
        wave_arrays = pproc.postprocess(embedding_batch)

        pred_test = sess.run(prediction_tensor_eval, feed_dict={features_tensor_eval: wave_arrays})
        
        model_vars = tf.compat.v1.trainable_variables()
        print(slim.model_analyzer.analyze_vars(model_vars, print_info=True))

        return pred_test,sess.graph 

#testing loop by restore the model for the checkpoint
def test(X_test,Y_test,checkpoint_dir,checkpoint_path,label_columns_name,minibatch_size=32,print_cost=True):
    
    # to keep results consistent (tensorflow seed)
    tf.compat.v1.set_random_seed(1)
    seed = 3

    m_test = X_test.shape[0]  
    n_y = Y_test.shape[1]
    standard_normal = stats.norm()
    np.seterr(divide='ignore', invalid='ignore')

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:    

        # Define a shallow classification model and associated testing ops on top
        # of VGGish.
        
        logits_test = vggish_slim.define_audio_slim(training=False,is_reuse=None)
         
        with tf.compat.v1.variable_scope('mymodel'):

          predict_test = tf.sigmoid(logits_test, name='prediction_test')
                
          # Add testing ops.
          with tf.variable_scope('train'):
            global_step = tf.Variable(
                0, name='global_step', trainable=False,  
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                             tf.compat.v1.GraphKeys.GLOBAL_STEP])

            # Labels are assumed to be fed as a batch multi-hot vectors, with
            # a 1 in the position of each positive class label, and 0 elsewhere.
            labels = tf.compat.v1.placeholder(
                tf.float32, shape=(None, n_y ), name='labels')
            
            xent_test = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_test, labels=labels, name='xent_test')
            loss_test = tf.reduce_mean(xent_test, name='loss_op_test')
            tf.compat.v1.summary.scalar('loss_test', loss_test)
                
        # Initialize all variables in the model, and then load the pre-tested
        # VGGish checkpoint.
        
        sess.run(tf.compat.v1.global_variables_initializer())  
        
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, checkpoint_path)

        # Locate all the tensors and ops we need for the testing loop.

        features_tensor_test = sess.graph.get_tensor_by_name(
            'audio/audio_input_features:0')
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name(
            'mymodel/train/global_step:0')
        loss_tensor_test = sess.graph.get_tensor_by_name('mymodel/train/loss_op_test:0')
        prediction_tensor_test = sess.graph.get_tensor_by_name('mymodel/prediction_test:0')  
                          
        graph = tf.Graph()
        with graph.as_default():
            vggish_slim.define_vggish_slim(training=False)
            sess_ext = tf.compat.v1.Session(graph=graph)
            vggish_slim.load_vggish_slim_checkpoint(sess_ext, checkpoint_dir + "vggish_model.ckpt")
            input_tensor = graph.get_tensor_by_name('vggish/input_features:0')
            output_tensor = graph.get_tensor_by_name('vggish/embedding:0')
            pproc = Postprocessor.Postprocessor(checkpoint_dir + "vggish_pca_params.npz")
                   
        print('\n###################')
        print('#  Testing loop  #')
        print('###################')
            
        avg_loss_test = 0.
        avg_map_test= 0.
        avg_auc_test = 0. 
        avg_d_prime_test = 0.
        step_test = 0
        avg_map_class_test = np.zeros_like(Y_test[0])

        num_minibatches_test = int(np.ceil(m_test / minibatch_size))
        minibatches_files_test = prepare_data.random_mini_batches_files(X_test, Y_test, minibatch_size, seed, shuffle=False) 
            
        for minibatch_file_test in minibatches_files_test:
            # Select a minibatch
                    
            (minibatch_X_test, minibatch_Y_test) =  prepare_data.get_train_data(minibatch_file_test,sess_ext,input_tensor,output_tensor,pproc,is_train=False,seed=seed,is_augment=False)

            num_steps_test ,step_loss_test,pred_tensor_test = sess.run(
                [global_step_tensor, loss_tensor_test,prediction_tensor_test],
                feed_dict={features_tensor_test: minibatch_X_test, labels_tensor: minibatch_Y_test})   
                
            step_auc_test = metrics.roc_auc_score(np.asarray(minibatch_Y_test), pred_tensor_test, average='micro')
            step_map_test = metrics.average_precision_score(np.asarray(minibatch_Y_test), pred_tensor_test, average='micro')
            step_d_prime_test = standard_normal.ppf(step_auc_test) * np.sqrt(2.0)
            avg_loss_test += step_loss_test / num_minibatches_test            
            avg_map_test += step_map_test / num_minibatches_test
            avg_auc_test += step_auc_test / num_minibatches_test 
            avg_d_prime_test += step_d_prime_test / num_minibatches_test
            step_map_class_test = metrics.average_precision_score(np.asarray(minibatch_Y_test), pred_tensor_test, average=None)
            step_map_class_test[np.isnan(step_map_class_test)] = 0
            avg_map_class_test += step_map_class_test / num_minibatches_test
            
            if((step_test+1) % 10 == 0 or step_test+1==num_minibatches_test or step_test==0):
                    print("Step: {}/{} ,Loss: {:.5f}, mAP: {:.5F}, AUC: {:.5F}, d-prime: {:.5F}" \
                    .format(
                    step_test+1 ,num_minibatches_test,step_loss_test,step_map_test,step_auc_test,step_d_prime_test))
                
            step_test += 1
              
        if print_cost == True:
                     
            print('###################')
            print("------Summary------" )
            print('###################')       
            print("Testing : loss %f, mAP %f, AUC %f, d-prime %f" % (avg_loss_test,avg_map_test,avg_auc_test,avg_d_prime_test))
    
    sum_of_labels_test = Y_test.sum(axis=0)

    #top 10 classes
    #index = sum_of_labels_test.argsort()[-10:][::-1]
    index = avg_map_class_test.argsort()[-10:][::-1]

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.bar(label_columns_name[index],sum_of_labels_test[index],alpha=0.55,color='C0',label='Audio files')
    ax1.set_ylabel('Number of audio files')
    ax1.set_title('The number of audio files and mAP of top 10 classes')  
    ax1.set_xticklabels(labels=label_columns_name[index],rotation=35) 

    ax2 = ax1.twinx()
    ax2.stem(avg_map_class_test[index],linefmt='r-', markerfmt='ro',basefmt='k-',use_line_collection=True,label='mAP')
    ax2.set_ylabel('mAP')
    ax2.set_xlabel('Classes')
    ax2.set_ylim(bottom=0.,top=1.0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.tight_layout()
    plt.savefig("./figures/map_top_10_class_model_base_with_augmented_2.png")
    fig.show()      
    plt.close()

    print('---Top 10 result---')
    print(label_columns_name[index])
    print(sum_of_labels_test[index])
    print(avg_map_class_test[index])

    #last 10 classes 

    #index = sum_of_labels_test.argsort()[0:10][::-1]
    index = avg_map_class_test.argsort()[0:10][::-1]

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.bar(label_columns_name[index],sum_of_labels_test[index],alpha=0.55,color='C0',label='Audio files')
    ax1.set_ylabel('Number of audio files')
    ax1.set_title('The number of audio files and mAP of last 10 classes')  
    ax1.set_xticklabels(labels=label_columns_name[index],rotation=35) 

    ax2 = ax1.twinx()
    ax2.stem(avg_map_class_test[index],linefmt='r-', markerfmt='ro',basefmt='k-',use_line_collection=True,label='mAP')
    ax2.set_ylabel('mAP')
    ax2.set_xlabel('Classes')
    ax2.set_ylim(bottom=0.,top=1.0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.tight_layout()
    plt.savefig("./figures/map_last_10_class_model_base_with_augmented_2.png")
    fig.show() 
    plt.close()

    print('---Last 10 result---')
    print(label_columns_name[index])
    print(sum_of_labels_test[index])
    print(avg_map_class_test[index])

def inference(file_path,checkpoint_dir,checkpoint_path):
    
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:    

        # Define a shallow classification model and associated testing ops on top
        # of VGGish.
        
        logits_inf = vggish_slim.define_audio_slim(training=False,is_reuse=None)
         
        with tf.compat.v1.variable_scope('mymodel'):

          predict_inf = tf.sigmoid(logits_inf, name='prediction_inf')
                
          # Add testing ops.
          with tf.variable_scope('train'):
            global_step = tf.Variable(
                0, name='global_step', trainable=False,  
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                             tf.compat.v1.GraphKeys.GLOBAL_STEP])
                
        # Initialize all variables in the model, and then load the pre-tested
        # VGGish checkpoint.
        
        sess.run(tf.compat.v1.global_variables_initializer())  
        
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, checkpoint_path)

        # Locate all the tensors and ops we need for the testing loop.

        features_tensor_inf = sess.graph.get_tensor_by_name(
            'audio/audio_input_features:0')

        prediction_tensor_inf = sess.graph.get_tensor_by_name('mymodel/prediction_inf:0')  
         
        graph = tf.Graph()
        with graph.as_default():
            vggish_slim.define_vggish_slim(training=False)
            sess_ext = tf.compat.v1.Session(graph=graph)
            vggish_slim.load_vggish_slim_checkpoint(sess_ext, checkpoint_dir + "vggish_model.ckpt")
            input_tensor = graph.get_tensor_by_name('vggish/input_features:0')
            output_tensor = graph.get_tensor_by_name('vggish/embedding:0')
            pproc = Postprocessor.Postprocessor(checkpoint_dir + "vggish_pca_params.npz")
                   
        print('\n###################')
        print('#  Inference loop  #')
        print('###################')

        try:
            data, sampleratde = sf.read(Path(file_path))
            wave_array_example_pre = data_transformation.waveform_to_examples(data,sampleratde,display=0)
        
            [embedding_batch] = sess_ext.run([output_tensor],
                feed_dict={input_tensor: wave_array_example_pre})
        
            wave_arrays = pproc.postprocess(embedding_batch)

            pred_inf_restore = sess.run(prediction_tensor_inf, feed_dict={features_tensor_inf: wave_arrays})
            return pred_inf_restore 

        except:
            print('This program does not support the input file format or file does not found ')
       
          
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--csv_dir', type=str)
    parser_train.add_argument('--dataset_train_dir', type=str)
    #parser_train.add_argument('--dataset_eval_dir', type=str)
    parser_train.add_argument('--vggish_checkpoint_dir', type=str)
    parser_train.add_argument('--save_checkpoint_dir', type=str)
    parser_train.add_argument('--epoch',type=int)
    parser_train.add_argument('--batch_size',type=int)
    parser_train.add_argument('--augmentation',action='store_true')
    
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--csv_dir', type=str)
    parser_test.add_argument('--dataset_test_dir', type=str)
    parser_test.add_argument('--vggish_checkpoint_dir', type=str)
    parser_test.add_argument('--checkpoint_path', type=str)
    parser_test.add_argument('--batch_size',type=int)

    parser_test_augmented = subparsers.add_parser('test_augmented')
    parser_test_augmented.add_argument('--csv_dir', type=str)
    parser_test_augmented.add_argument('--dataset_test_dir', type=str)
    parser_test_augmented.add_argument('--vggish_checkpoint_dir', type=str)
    parser_test_augmented.add_argument('--checkpoint_path', type=str)
    parser_test_augmented.add_argument('--batch_size',type=int)

    parser_inf = subparsers.add_parser('inference')
    parser_inf.add_argument('--csv_dir', type=str)
    parser_inf.add_argument('--file_path', type=str)
    parser_inf.add_argument('--vggish_checkpoint_dir', type=str)
    parser_inf.add_argument('--checkpoint_path', type=str)

    args = parser.parse_args()
  
    
    if args.mode == "train":
        if args.augmentation == True:
            print('Training mode - Augmentation')
        elif args.augmentation == False:
            print('Training mode - Normal')

        files_name_train,labels_train,files_name_eval,labels_eval = prepare_data.get_filenames_and_labels(args)
        output_tensor_test,sess_graph = train(files_name_train, labels_train, files_name_eval, labels_eval, args.vggish_checkpoint_dir,
             args.save_checkpoint_dir,num_epochs=args.epoch,minibatch_size=args.batch_size,print_cost=True,augmentation=args.augmentation)   
        
        np.set_printoptions(precision=4)
        for i in range(len(output_tensor_test)):
            print('Second: ',i)
            print('Top 10 prob labels:    ',output_tensor_test[i].argsort()[-10:][::-1])
            print('Raw probability values: ',output_tensor_test[i][output_tensor_test[i].argsort()[-10:][::-1]])
            print('-------------------------------------------')

    elif args.mode == "test":
        print('Testing mode')
        files_name_test,labels_test,label_columns_name = prepare_data.get_filenames_and_labels_test(args)
        test(files_name_test, labels_test,args.vggish_checkpoint_dir,args.checkpoint_path,label_columns_name,minibatch_size=args.batch_size,print_cost=True)
    
    elif args.mode =="test_augmented":
        print('Testing mode with augmented data')    
        files_name_test,labels_test,label_columns_name = prepare_data.get_filenames_and_labels_test_augment(args)
        #print(files_name_test[0])
        #print(np.where(labels_test[0]==1))
        test(files_name_test, labels_test,args.vggish_checkpoint_dir,args.checkpoint_path,label_columns_name,minibatch_size=args.batch_size,print_cost=True)

    elif args.mode == "inference":
        print('Inference mode')
        
        label_columns = prepare_data.get_labels_indices(args.csv_dir)
        output_tensor_inf = inference(args.file_path,args.vggish_checkpoint_dir,args.checkpoint_path)

        np.set_printoptions(precision=4)
        try:
            for i in range(len(output_tensor_inf)):
                print('Second: ',i)
                print('Top 10 prob labels name : ',label_columns[output_tensor_inf[i].argsort()[-10:][::-1]])
                print('Top 10 prob labels id   : ',output_tensor_inf[i].argsort()[-10:][::-1])
                print('Raw probability values  : ',output_tensor_inf[i][output_tensor_inf[i].argsort()[-10:][::-1]])
                print('-------------------------------------------')
        except:
            pass

    else:
        print("Please complete the input parameter")

    
