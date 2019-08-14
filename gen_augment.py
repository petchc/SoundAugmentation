from __future__ import print_function
import soundfile as sf
import shutil
import os
import numpy as np
from pathlib import Path
import pandas as pd
import time
import random as rd
import ffmpeg

def map_dataset(df_csv,file_path):
    
        #n_classes = num_classes # 527
        input_path = file_path 
        df_map = pd.DataFrame()
     
        df_map['Fname'] = input_path + df_csv["# YTID"].str.slice(0,1).str.upper() + '/' + df_csv["# YTID"].str.slice(1,2).str.upper() + \
            '/' + df_csv["# YTID"].str.slice(2,3).str.upper() + '/' + df_csv["# YTID"] + '_' + (df_csv["start_seconds"]*1000).astype(int).astype(str) \
            + '_' + (df_csv["end_seconds"]*1000).astype(int).astype(str) + '.flac'
    
        return df_map

def get_filenames_and_labels_test(csv_dir,dataset_test_dir):

   
        csv_dir = csv_dir
        dataset_test_dir = dataset_test_dir
    
        df_test = pd.read_csv(Path(csv_dir + 'eval_segments.csv'),sep=',',skiprows=2,
                    engine='python',quotechar = '"',skipinitialspace = True,)
    
        #column_df = pd.read_csv(Path(csv_dir + 'class_labels_indices.csv'),usecols=["mid","display_name"])
    
        print('Preprocessing for dataset location and labels...')
    
        df_test_map = map_dataset(df_test,dataset_test_dir)
    
        files_name_test = df_test_map['Fname'].values
        #labels_test = df_test_map.loc[:,label_columns].values
    
        return files_name_test

def create_folder(fd):
        if not os.path.exists(fd):
            os.makedirs(fd)

def get_augmentation_data(files_name_test):
    
    output_aug_dir = '/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_augment/'
    shutil.rmtree('/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_augment')
    create_folder(output_aug_dir)

    #i=0
    seed=1
  
    for file_name in files_name_test:
        rd.seed(seed)
        #print(file_name)
            
        augment_rd = rd.randint(0,2)
   
        #FLAC format
        if(augment_rd==0):
            file_name_aug = output_aug_dir + file_name.split('/')[-1]
            #acodec_rd='flac'
            try:
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=2)
                    .filter(filter_name='afftdn')
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.8,ratio=20,attack=20,release=200)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=50,decays=1)
                    .output(file_name_aug,acodec='flac')
                    .overwrite_output()
                    .run(quiet=True)
                    )
                #print('flac')
            except:
                #print('fnf')
                pass

        #OGG format    
        elif(augment_rd==1):
            #file_name_aug = output_aug_dir + input_file.split('/')[-1].replace('000.flac','000_aac.aac')
            file_name_aug = output_aug_dir + file_name.split('/')[-1].replace('000.flac','000.aac')
            #acodec_rd='aac'
            try:
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=4)
                    .filter(filter_name='afftdn')
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.7,ratio=15,attack=15,release=100)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=100,decays=1)
                    .output(file_name_aug,acodec='aac')
                    .overwrite_output()
                    .run(quiet=True)
                    )
                
                file_name_aug_flac = file_name_aug.replace('.aac','.flac')

                ffmpeg_proc = (ffmpeg
                    .input(file_name_aug)  
                    .output(file_name_aug_flac,acodec='flac')
                    .overwrite_output()
                    .run(quiet=True)
                )
                #file_name_aug = file_name_aug_flac
            
            except:
                #print('fnf')
                pass
            
            #print('aac')    

        #MP3 format    
        elif(augment_rd==2):
            try:
                #file_name_aug = output_aug_dir + input_file.split('/')[-1].replace('000.flac','000_mp3.mp3')
                file_name_aug = output_aug_dir + file_name.split('/')[-1].replace('000.flac','000.mp3')
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=6)
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.5,ratio=10,attack=10,release=50)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=200,decays=1)
                    .output(file_name_aug,acodec='mp3')
                    .overwrite_output()
                    .run(quiet=True) 
                    )  
                
                file_name_aug_flac = file_name_aug.replace('.mp3','.flac')

                ffmpeg_proc = (ffmpeg
                    .input(file_name_aug)  
                    .output(file_name_aug_flac,acodec='flac')
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                #print('mp3')
            except:
                #print('fnf')
                pass
            
        #i = i+1
        seed= seed+1
        #if i==40:
        #    break   
    #librosa can read more format!
    #'''
 

if __name__ == '__main__':
      
        files_name_test = get_filenames_and_labels_test('/export/home/2368985c/MSc_Project_Sound_Augmentation/csv_file/','/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_eval_tmp/')
        print('Generating augmented test set')
        get_augmentation_data(files_name_test)
        
