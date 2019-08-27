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
import librosa

def split_and_label(rows_labels, label_mapping ,n_classes):
    '''
    Retrieves a list of all the relevant classes. This is necessary due to 
    the multi-labeling of the initial csv file.
    '''
    row_labels_list = []
    for row in rows_labels:
        row_labels = row.split(',')
        labels_array = np.zeros((n_classes))
        for label in row_labels:
            index = label_mapping[label]
            labels_array[index] = 1
        row_labels_list.append(labels_array)
    return row_labels_list



#Map filename to local path and make labels to one-hot encoding

def map_dataset(df_csv,file_path,label_column_value,num_classes):
    
    n_classes = num_classes # 527
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
    
    n_classes = num_classes # 527
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

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["display_name"])

    label_columns_name = column_df.display_name.values

    print('Preprocessing for dataset location and labels...')

    return label_columns_name

def get_filenames_and_labels_test(args):

    num_classes = 527

    csv_dir = args.csv_dir
    dataset_test_dir = args.dataset_test_dir

    df_test = pd.read_csv(Path(csv_dir + 'eval_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid","display_name"])

    # Retrieve list of labels
    label_columns = column_df.mid.values
    
    #column_df_test = pd.read_csv(Projec_Dir + '/class_labels_indices.csv',usecols=["display_name"])
    
    label_columns_name = column_df.display_name.values


    print('Preprocessing for dataset location and labels...')

    df_test_map = map_dataset(df_test,dataset_test_dir,column_df.mid.values,num_classes)

    files_name_test = df_test_map['Fname'].values
    labels_test = df_test_map.loc[:,label_columns].values

    return files_name_test,labels_test,label_columns_name

def get_filenames_and_labels_test_augment(args):

    num_classes = 527

    csv_dir = args.csv_dir
    dataset_test_dir = args.dataset_test_dir

    df_test = pd.read_csv(Path(csv_dir + 'eval_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid","display_name"])

    # Retrieve list of labels
    label_columns = column_df.mid.values
    
    #column_df_test = pd.read_csv(Projec_Dir + '/class_labels_indices.csv',usecols=["display_name"])
    
    label_columns_name = column_df.display_name.values


    print('Preprocessing for dataset location and labels...')

    df_test_map = map_dataset_augment(df_test,dataset_test_dir,column_df.mid.values,num_classes)

    files_name_test = df_test_map['Fname'].values
    labels_test = df_test_map.loc[:,label_columns].values

    return files_name_test,labels_test,label_columns_name

def get_filenames_and_labels(args):

    num_classes = 527

    csv_dir = args.csv_dir
    dataset_train_dir = args.dataset_train_dir
    #dataset_eval_dir = args.dataset_eval_dir

    #print(csv_dir)
    #print(dataset_train_dir)
    #print(dataset_eval_dir)

    df_train = pd.read_csv(Path(csv_dir + 'balanced_train_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    #df_eval = pd.read_csv(Path(csv_dir  + 'eval_segments.csv'),sep=',',skiprows=2,
    #            engine='python',quotechar = '"',skipinitialspace = True,)

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid"])

    # Retrieve list of labels
    label_columns = column_df.mid.values
    
    #print(df_train)
    #print(df_eval)
    print('Preprocessing for dataset location and labels...')

    df_train_map = map_dataset(df_train,dataset_train_dir,column_df.mid.values,num_classes)
    #df_eval_map = map_dataset(df_eval,dataset_eval_dir,column_df.mid.values,num_classes)
    
    #df_train_map = map_dataset(df_train,'C:\\Users\\pthad\\Downloads\\audio_train\\',column_df.mid.values,num_classes)
    #df_eval_map = map_dataset(df_eval,'C:\\Users\\pthad\\Downloads\\audio_eval\\',column_df.mid.values,num_classes)

    files_name_train_all = df_train_map['Fname'].values
    labels_train_all = df_train_map.loc[:,label_columns].values

    np.random.seed(1)
    permutation = list(np.random.permutation(files_name_train_all.shape[0]))

    
    #print(permutation)
    #print(np.shape(permutation))

    shuffled_X = files_name_train_all[permutation]
    shuffled_Y = labels_train_all[permutation]

    #files_name_eval = df_eval_map['Fname'].values
    #labels_eval= df_eval_map.loc[:,label_columns].values
    files_name_train = shuffled_X[0:20000]
    labels_train =  shuffled_Y[0:20000]

    files_name_eval = shuffled_X[20000:]
    labels_eval = shuffled_Y[20000:]


    return files_name_train,labels_train,files_name_eval,labels_eval

def random_mini_batches_files(X, Y, mini_batch_size = 64, seed = 0 , shuffle=True):
    """
    Creates a list of random minibatches from (X, Y)
    
    It will product list of filename as minibatch
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot),
         of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    # To make your "random" minibatches the same as ours

    # number of training examples
    m = X.shape[0]
    mini_batches = []
    
    if(shuffle==True):
        # Step 1: Shuffle (X, Y)
        np.random.seed(seed)
        permutation = list(np.random.permutation(m))
        #print(permutation)
        #print(np.shape(permutation))

        shuffled_X = X[permutation]
        shuffled_Y = Y[permutation, :]

        #print(np.shape(shuffled_X))
        #print(np.shape(shuffled_Y))
    
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

# Read the list of file names from minibatch and preprocessing then return as list of example frame and 
# associated labels

def data_augmentation(file_name,seed):
    #input_file = '/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_train_tmp/2/1/L/21L0pyts9WI_530000_540000.flac'
    input_file = file_name
    #print('input: ',input_file) 
    #print('seed aug: ',seed)
    
    output_aug_dir = './audio_aug/'
    
    #random parameters
    rd.seed(seed)
    volume_rd = rd.randint(1, 10)
    threshold_rd = rd.uniform(0.5,0.8)
    ratio_rd = rd.randint(10, 20)
    attack_rd = rd.randint(1, 20)
    release_rd = rd.randint(10, 250)
    decays_rd =rd.uniform(0.5,1)
    format_type = rd.randint(0,2)
    libro_parm = rd.uniform(0.8,1.5)


    #print(volume_rd)
    #print(threshold_rd)
    #print(ratio_rd)
    #print(attack_rd)
    #print(release_rd)
    #print(decays_rd)
    #print('format_type: ',format_type) 
    
    #FLAC format
    if(format_type==0):
        file_name_aug = output_aug_dir + input_file.split('/')[-1]
        acodec_rd='flac'
        #print('flac')

    #OGG format    
    elif(format_type==1):
        #file_name_aug = output_aug_dir + input_file.split('/')[-1].replace('000.flac','000_aac.aac')
        file_name_aug = output_aug_dir + input_file.split('/')[-1].replace('000.flac','000.aac')
        acodec_rd='aac'
        #print('aac')

    #MP3 format    
    elif(format_type==2):
        #file_name_aug = output_aug_dir + input_file.split('/')[-1].replace('000.flac','000_mp3.mp3')
        file_name_aug = output_aug_dir + input_file.split('/')[-1].replace('000.flac','000.mp3')
        acodec_rd='mp3'  
        #print('mp3')
 

    #file_name_aug = file_name_aug.replace('audio_train_tmp','audio_augment')    

    #print('step1 filename: ',file_name_aug)
    
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
    
    #librosa can read more format!
    #'''
    if(format_type==1):

        file_name_aug_flac = file_name_aug.replace('.aac','.flac')
        
        ffmpeg_proc = (ffmpeg
            .input(file_name_aug)  
            .output(file_name_aug_flac,acodec='flac')
            .overwrite_output()
            .run(quiet=True)
        )
        file_name_aug = file_name_aug_flac
        #print('in aac')
            
    if(format_type==2):

        file_name_aug_flac = file_name_aug.replace('.mp3','.flac')
        
        ffmpeg_proc = (ffmpeg
            .input(file_name_aug)  
            .output(file_name_aug_flac,acodec='flac')
            .overwrite_output()
            .run(quiet=True)
        )
        file_name_aug = file_name_aug_flac
        #print('in mp3')
    
    #print('final file name to return: ', file_name_aug)  
    #print('=========================') 
    #'''
    return file_name_aug

def get_train_data(minibatch_file,sess_ext,input_tensor,output_tensor,pproc,is_train,seed=0,is_augment=False):
    
    #print('in get train data')
    #print('is_augment: ',is_augment)
    files_name,labels_name = minibatch_file  
    all_examples=[]
    all_labels=[]   
    no_file=0
   
    for file_name in files_name:

        try:    
            #print(file_name) 
            data, sampleratde = sf.read(Path(file_name))  #change from soundfile to librosa      
            #data, sampleratde = librosa.load(Path(file_name),sr=16000,dtype=np.float64)     
             
            #print(file_name) 
             
            wave_arrays_pre = data_transformation.waveform_to_examples(data,sampleratde,0)
            
            [embedding_batch] = sess_ext.run([output_tensor],
                feed_dict={input_tensor: wave_arrays_pre})
        
            wave_arrays = pproc.postprocess(embedding_batch)

            if(wave_arrays.shape[0]!=0):
                wave_labels = np.array([labels_name[no_file]] * wave_arrays.shape[0])      
                all_examples.append(wave_arrays)
                all_labels.append(wave_labels)

                if(is_augment == True):
                    #print('in loop augment')
                    #add augmentation step here!!!
                    #print('seed: ',seed)
                    file_name_aug = data_augmentation(file_name,seed)
                    #print('file_name1:',file_name_aug)
                    #print('file_name_aug in get:', file_name_aug)
                    
                    data, sampleratde = sf.read(Path(file_name_aug))  #change from soundfile to librosa         
                    #data, sampleratde = librosa.load(Path(file_name_aug),sr=16000,dtype=np.float64) 
                    
                    #add more techniques
                    #data = librosa.effects.pitch_shift(data, 16000, n_steps=1.5) #0.8-1.5
                    #data = librosa.effects.time_stretch(data, 1.5) #0.8-1.5
                    
                    wave_arrays_pre = data_transformation.waveform_to_examples(data,sampleratde,0)
  
                    [embedding_batch] = sess_ext.run([output_tensor],
                        feed_dict={input_tensor: wave_arrays_pre})
                
                    wave_arrays = pproc.postprocess(embedding_batch)

                    if(wave_arrays.shape[0]!=0):
                        wave_labels = np.array([labels_name[no_file]] * wave_arrays.shape[0])      
                        all_examples.append(wave_arrays)
                        all_labels.append(wave_labels)
                        #print('!!!!!!!!!!!!!!!!!!!!!in loop aug!!!!!gen')
                        #print('file_name2:',file_name_aug)
                        #print(np.where(wave_labels[0]==1))
                        #print('file augmented = 1')
                    
                    seed = seed+1        

                  
        except:
            #print ("The referred audio file is not available")
            #print(file_name)
            pass
                       
        no_file = no_file+1
                
    all_examples = np.concatenate(all_examples)
    all_labels = np.concatenate(all_labels)

    if(is_train==True):
        #print('return is train')
        labeled_examples = list(zip(all_examples, all_labels))
        shuffle(labeled_examples)
        #Separate and return the features and labels.
        features = [example for (example, _) in labeled_examples]
        labels = [label for (_, label) in labeled_examples]
        return (features, labels)
    
    elif(is_train==False):
        return (all_examples, all_labels)

    #return (all_examples, all_labels)
        

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na