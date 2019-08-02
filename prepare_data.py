import pandas as pd
import argparse
import numpy as np
import os
import math
import soundfile as sf
from pathlib import Path
import data_transformation
from random import shuffle

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

def get_labels_indices(csv_dir):

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["display_name"])

    label_columns_name = column_df.display_name.values

    print('Preprocessing Data Location and Labels...')

    return label_columns_name

def get_filenames_and_labels_test(args):

    num_classes = 527

    csv_dir = args.csv_dir
    dataset_test_dir = args.dataset_test_dir

    df_test = pd.read_csv(Path(csv_dir + 'balanced_train_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid","display_name"])

    # Retrieve list of labels
    label_columns = column_df.mid.values
    
    #column_df_test = pd.read_csv(Projec_Dir + '/class_labels_indices.csv',usecols=["display_name"])
    
    label_columns_name = column_df.display_name.values


    print('Preprocessing Data Location and Labels...')

    df_test_map = map_dataset(df_test,dataset_test_dir,column_df.mid.values,num_classes)

    files_name_test = df_test_map['Fname'].values
    labels_test = df_test_map.loc[:,label_columns].values

    return files_name_test[20000:],labels_test[20000:],label_columns_name

def get_filenames_and_labels(args):

    num_classes = 527

    csv_dir = args.csv_dir
    dataset_train_dir = args.dataset_train_dir
    dataset_eval_dir = args.dataset_eval_dir


    print(csv_dir)
    print(dataset_train_dir)
    print(dataset_eval_dir)
    #dataset_train_dir = args.dataset_train_dir
    #dataset_eval_dir = args.dataset_eval_dir

    #Projec_Dir = 'C:/Users/pthad/Desktop/Final_Project/'

    df_train = pd.read_csv(Path(csv_dir + 'balanced_train_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    df_eval = pd.read_csv(Path(csv_dir  + 'eval_segments.csv'),sep=',',skiprows=2,
                engine='python',quotechar = '"',skipinitialspace = True,)

    column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid"])

    # Retrieve list of labels
    label_columns = column_df.mid.values
    
    #print(df_train)
    #print(df_eval)
    print('Preprocessing Data Location and Labels...')

    df_train_map = map_dataset(df_train,dataset_train_dir,column_df.mid.values,num_classes)
    df_eval_map = map_dataset(df_eval,dataset_eval_dir,column_df.mid.values,num_classes)
    
    #df_train_map = map_dataset(df_train,'C:\\Users\\pthad\\Downloads\\audio_train\\',column_df.mid.values,num_classes)
    #df_eval_map = map_dataset(df_eval,'C:\\Users\\pthad\\Downloads\\audio_eval\\',column_df.mid.values,num_classes)

    files_name_train = df_train_map['Fname'].values
    labels_train = df_train_map.loc[:,label_columns].values

    files_name_eval = df_eval_map['Fname'].values
    labels_eval= df_eval_map.loc[:,label_columns].values

    print(np.shape(files_name_train))
    print(np.shape(labels_train))
    print(np.shape(files_name_eval))
    print(np.shape(labels_eval))

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

def get_train_data(minibatch_file,sess_ext,input_tensor,output_tensor,pproc,is_train):
    
    files_name,labels_name = minibatch_file
    
    all_examples=[]
    all_labels=[]
    
    no_file=0
   
    for file_name in files_name:#minibatch_file[0]):
        #print(file_name)
        #file_name
        #print(file_name.split(''))
        try:    
        #   print("in loop")
        #   print(Path(file_name))
            data, sampleratde = sf.read(Path(file_name))          
            wave_arrays_pre = data_transformation.waveform_to_examples(data,sampleratde,0)
            
            [embedding_batch] = sess_ext.run([output_tensor],
                feed_dict={input_tensor: wave_arrays_pre})
        
        
            #pproc = Postprocessor("./vggish_ckpt/vggish_pca_params.npz")
            wave_arrays = pproc.postprocess(embedding_batch)


        #   print(np.shape(wave_arrays))
            if(wave_arrays.shape[0]!=0):
        #   print('in if')
                wave_labels = np.array([labels_name[no_file]] * wave_arrays.shape[0])      
                all_examples.append(wave_arrays)
                all_labels.append(wave_labels)
                  
        except:
            #print ("The referred audio file is not available")
            #print(file_name)
            pass
                       
        no_file = no_file+1
                
    all_examples = np.concatenate(all_examples)
    all_labels = np.concatenate(all_labels)

    if(is_train==True):
        #print('is_Train=True')
        labeled_examples = list(zip(all_examples, all_labels))
        shuffle(labeled_examples)
        #Separate and return the features and labels.
        features = [example for (example, _) in labeled_examples]
        labels = [label for (_, label) in labeled_examples]
        return (features, labels)
    
    elif(is_train==False):
        #print('is_Train=False')
        return (all_examples, all_labels)





    return (all_examples, all_labels)
        

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na