import pandas as pd
import argparse
import numpy as np
import os
import math
import soundfile as sf
from pathlib import Path
import data_transformation
import random as rd
from random import shuffle
import ffmpeg

def split_and_label(rows_labels, label_mapping ,n_classes):
    
    # retrieves a list of all the relevant classes and split it into individual label.
    
    row_labels_list = []
    for row in rows_labels:
        row_labels = row.split(',')
        labels_array = np.zeros((n_classes))
        for label in row_labels:
            index = label_mapping[label]
            labels_array[index] = 1
        row_labels_list.append(labels_array)
    return row_labels_list

def map_dataset(df_csv,file_path,label_column_value,num_classes):

    
    # map filename to local path and encode the labels to one-hot encoding for the baseline model
       
    n_classes = num_classes 
    input_path = file_path 
    df_map = pd.DataFrame()
 
    df_map['Fname'] = input_path + df_csv["# YTID"].str.slice(0,1).str.upper() + '/' + df_csv["# YTID"].str.slice(1,2).str.upper() + \
        '/' + df_csv["# YTID"].str.slice(2,3).str.upper() + '/' + df_csv["# YTID"] + '_' + (df_csv["start_seconds"]*1000).astype(int).astype(str) \
        + '_' + (df_csv["end_seconds"]*1000).astype(int).astype(str) + '.flac'
    df_map["Label"] = df_csv.positive_labels
    label_columns = label_column_value
    label_mapping = dict((label, index) for index, label in enumerate(label_columns))
    for col in label_columns:
        df_map[col] = 0  
    df_map[label_columns] = split_and_label(df_map['Label'], label_mapping ,n_classes)

    return df_map

def map_dataset_augment(df_csv,file_path,label_column_value,num_classes):
    
    
    # map filename to local path and encode the labels to one-hot encoding for data augmentation model
      

    n_classes = num_classes 
    input_path = file_path 
    df_map = pd.DataFrame()
 
    df_map['Fname'] = input_path +  df_csv["# YTID"] + '_' + (df_csv["start_seconds"]*1000).astype(int).astype(str) \
        + '_' + (df_csv["end_seconds"]*1000).astype(int).astype(str) + '.flac'
    df_map["Label"] = df_csv.positive_labels
    label_columns = label_column_value
    label_mapping = dict((label, index) for index, label in enumerate(label_columns))
    for col in label_columns:
        df_map[col] = 0  
    df_map[label_columns] = split_and_label(df_map['Label'], label_mapping ,n_classes)

    return df_map

def get_labels_indices(csv_dir):

    
    # get all classes name 
    

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["display_name"])

    label_columns_name = column_df.display_name.values

    print('Preprocessing for dataset location and labels...')

    return label_columns_name

def get_filenames_and_labels_test(args):

    # get filenames with file location and the associated labels for test with original test set
    
    num_classes = 527

    csv_dir = args.csv_dir
    dataset_test_dir = args.dataset_test_dir

    df_test = pd.read_csv(Path(csv_dir + 'eval_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid","display_name"])

    label_columns = column_df.mid.values
    
    label_columns_name = column_df.display_name.values

    print('Preprocessing for dataset location and labels...')

    df_test_map = map_dataset(df_test,dataset_test_dir,column_df.mid.values,num_classes)

    files_name_test = df_test_map['Fname'].values
    labels_test = df_test_map.loc[:,label_columns].values

    return files_name_test,labels_test,label_columns_name

def get_filenames_and_labels_test_augment(args):

    # get filenames with file location and the associated labels for test with augmented test set
    
    num_classes = 527

    csv_dir = args.csv_dir
    dataset_test_dir = args.dataset_test_dir

    df_test = pd.read_csv(Path(csv_dir + 'eval_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid","display_name"])

    label_columns = column_df.mid.values
    
    label_columns_name = column_df.display_name.values

    print('Preprocessing for dataset location and labels...')

    df_test_map = map_dataset_augment(df_test,dataset_test_dir,column_df.mid.values,num_classes)

    files_name_test = df_test_map['Fname'].values
    labels_test = df_test_map.loc[:,label_columns].values

    return files_name_test,labels_test,label_columns_name

def get_filenames_and_labels(args):

    # get filenames with file location and the associated labels for training the model
    
    num_classes = 527

    csv_dir = args.csv_dir
    dataset_train_dir = args.dataset_train_dir

    df_train = pd.read_csv(Path(csv_dir + 'balanced_train_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid"])

    label_columns = column_df.mid.values
    
    print('Preprocessing for dataset location and labels...')

    df_train_map = map_dataset(df_train,dataset_train_dir,column_df.mid.values,num_classes)

    files_name_train_all = df_train_map['Fname'].values
    labels_train_all = df_train_map.loc[:,label_columns].values

    np.random.seed(1)
    permutation = list(np.random.permutation(files_name_train_all.shape[0]))
   
    shuffled_X = files_name_train_all[permutation]
    shuffled_Y = labels_train_all[permutation]

    files_name_train = shuffled_X[0:20000]
    labels_train =  shuffled_Y[0:20000]

    files_name_eval = shuffled_X[20000:]
    labels_eval = shuffled_Y[20000:]

    return files_name_train,labels_train,files_name_eval,labels_eval

def random_mini_batches_files(X, Y, mini_batch_size = 64, seed = 0 , shuffle=True):
    '''
        Creates a list of random minibatches from (X, Y)
        
        It will product list of filename as minibatch
        
            Args:
                X : list of filenames
                Y : ground true label
                mini_batch_size : size of the mini-batches
                seed : random seed to generate the different list of filename in each epoch.
                shuffle : shuffle the data or not

            Return:
                mini_batches : list of synchronous (mini_batch_X, mini_batch_Y)
    '''
    #ref https://www.kaggle.com/darienbm/wine-classification-using-tensorflow
    
    m = X.shape[0]
    mini_batches = []
    
    if(shuffle==True):
    # Step 1: Shuffle (X, Y)
        np.random.seed(seed)
        permutation = list(np.random.permutation(m))

        shuffled_X = X[permutation]
        shuffled_Y = Y[permutation, :]

    
    elif(shuffle==False):
        shuffled_X = X
        shuffled_Y = Y
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning -> step size 
    num_complete_minibatches = math.floor(m/mini_batch_size) 

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size : m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def data_augmentation(file_name,seed):

    '''
        Applying 7 random parameters from 5 filters and 3 type of audios to generate the artificial audio files 
        in order to training the augmentation model.
    '''

    input_file = file_name
    output_aug_dir = './audio_aug/'
    
    # random parameters
    rd.seed(seed)
    volume_rd = rd.randint(1, 10)
    threshold_rd = rd.uniform(0.5,0.8)
    ratio_rd = rd.randint(10, 20)
    attack_rd = rd.randint(1, 20)
    release_rd = rd.randint(10, 250)
    decays_rd =rd.uniform(0.5,1)
    format_type = rd.randint(0,2)

    # FLAC format
    if(format_type==0):
        file_name_aug = output_aug_dir + input_file.split('/')[-1]
        acodec_rd='flac'

    # AAC format    
    elif(format_type==1):
        file_name_aug = output_aug_dir + input_file.split('/')[-1].replace('000.flac','000.aac')
        acodec_rd='aac'

    # MP3 format    
    elif(format_type==2):
        file_name_aug = output_aug_dir + input_file.split('/')[-1].replace('000.flac','000.mp3')
        acodec_rd='mp3'  
    
    # augmentation pipeline
    ffmpeg_proc = (ffmpeg
            .input(input_file)  
            .filter(filter_name='volume',volume=volume_rd)
            .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
            .filter(filter_name='acompressor',threshold=threshold_rd,ratio=ratio_rd,attack=attack_rd,release=release_rd)
            .filter(filter_name='silenceremove',stop_periods=-1,stop_duration=2,stop_threshold='-50dB')
            .filter(filter_name='aecho',in_gain=1,out_gain=0.9,delays=500,decays=decays_rd)
            .output(file_name_aug,acodec=acodec_rd)
            .overwrite_output()
            .run(quiet=True)
            )
    
    # convert file types from AAC to FLAC due to SoundFile library does not support to read AAC format. 

    if(format_type==1):

        file_name_aug_flac = file_name_aug.replace('.aac','.flac')
        
        ffmpeg_proc = (ffmpeg
            .input(file_name_aug)  
            .output(file_name_aug_flac,acodec='flac')
            .overwrite_output()
            .run(quiet=True)
        )
        file_name_aug = file_name_aug_flac

    # convert file types from MP3 to FLAC due to SoundFile library does not support to read MP3 format.     
            
    if(format_type==2):

        file_name_aug_flac = file_name_aug.replace('.mp3','.flac')
        
        ffmpeg_proc = (ffmpeg
            .input(file_name_aug)  
            .output(file_name_aug_flac,acodec='flac')
            .overwrite_output()
            .run(quiet=True)
        )
        file_name_aug = file_name_aug_flac

    return file_name_aug

def get_train_data(minibatch_file,sess_ext,input_tensor,output_tensor,pproc,is_train,seed=0,is_augment=False):
    
    '''
        Read the list of file names from minibatch and perform preprocessing 
        then return as list of example frame and associated labels
    '''

    files_name,labels_name = minibatch_file  
    all_examples=[]
    all_labels=[]   
    no_file=0
   
    for file_name in files_name:

        try:    

            data, sampleratde = sf.read(Path(file_name))   
           
            wave_arrays_pre = data_transformation.waveform_to_examples(data,sampleratde,0)
            
            [embedding_batch] = sess_ext.run([output_tensor],
                feed_dict={input_tensor: wave_arrays_pre})
        
            wave_arrays = pproc.postprocess(embedding_batch)

            if(wave_arrays.shape[0]!=0):
                wave_labels = np.array([labels_name[no_file]] * wave_arrays.shape[0])      
                all_examples.append(wave_arrays)
                all_labels.append(wave_labels)

                '''
                    This is a data augmentation function to generate artificial audio file
                    for each file.
                '''

                if(is_augment == True):

                    file_name_aug = data_augmentation(file_name,seed)

                    data, sampleratde = sf.read(Path(file_name_aug))       
                    
                    wave_arrays_pre = data_transformation.waveform_to_examples(data,sampleratde,0)
  
                    [embedding_batch] = sess_ext.run([output_tensor],
                        feed_dict={input_tensor: wave_arrays_pre})
                
                    wave_arrays = pproc.postprocess(embedding_batch)

                    if(wave_arrays.shape[0]!=0):
                        wave_labels = np.array([labels_name[no_file]] * wave_arrays.shape[0])      
                        all_examples.append(wave_arrays)
                        all_labels.append(wave_labels)
                    
                    seed = seed+1        
                  
        except:
            pass
                       
        no_file = no_file+1
                
    all_examples = np.concatenate(all_examples)
    all_labels = np.concatenate(all_labels)

    if(is_train==True):
        labeled_examples = list(zip(all_examples, all_labels))
        shuffle(labeled_examples)

        #Separate and return the features and labels.
        
        features = [example for (example, _) in labeled_examples]
        labels = [label for (_, label) in labeled_examples]
        return (features, labels)
    
    elif(is_train==False):
        return (all_examples, all_labels)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na