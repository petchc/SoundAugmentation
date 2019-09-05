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
        '''
        Create the mapping between filename and dataset location 
        using content in csv file and dataset directory path.

            Args:
                csv_dir : directory path of csv file
                file_path : directory path of evaluation dataset

            Return:
                list of mapping between filename and dataset location.

        '''   

        input_path = file_path 
        df_map = pd.DataFrame()
     
        df_map['Fname'] = input_path + df_csv["# YTID"].str.slice(0,1).str.upper() + '/' + df_csv["# YTID"].str.slice(1,2).str.upper() + \
            '/' + df_csv["# YTID"].str.slice(2,3).str.upper() + '/' + df_csv["# YTID"] + '_' + (df_csv["start_seconds"]*1000).astype(int).astype(str) \
            + '_' + (df_csv["end_seconds"]*1000).astype(int).astype(str) + '.flac'
    
        return df_map

def get_filenames_and_labels_test(csv_dir,dataset_test_dir):

        '''
        Reading the filename from csv file and mapping with the dataset location.

            Args:
                csv_dir : directory path of csv file
                dataset_test_dir : directory path of evaluation dataset

            Return:
                list of filenames with location path. 

        '''
        
        csv_dir = csv_dir
        dataset_test_dir = dataset_test_dir
    
        df_test = pd.read_csv(Path(csv_dir + 'eval_segments.csv'),sep=',',skiprows=2,
                    engine='python',quotechar = '"',skipinitialspace = True,)
    
        print('Preprocessing for dataset location and labels...')
    
        df_test_map = map_dataset(df_test,dataset_test_dir)
    
        files_name_test = df_test_map['Fname'].values
    
        return files_name_test

def create_folder(fd):
        if not os.path.exists(fd):
            os.makedirs(fd)

def get_augmentation_data(files_name_test):

    '''
     Generate augmented test set using 10 fixed FFmpeg pipeline to simulate 10 sources 
     which using to record audio files. Detail of each fileter please see Appendix B.

        Args:
            files_name_test : list of test files to perform the data augmentation.
        Output:
            augmented test set.
    '''
    
    output_aug_dir = '/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_augmented_10_pipelines/'

    #create folder if it does not exits
    create_folder(output_aug_dir)
    
    #clear the folder it is already exites
    shutil.rmtree('/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_augmented_10_pipelines')
    create_folder(output_aug_dir)

    seed=1
    
    i=0
    toteal_file = len(files_name_test)

    for file_name in files_name_test:
        
        # random pipeline for each audio file
        rd.seed(seed)
        augment_rd = rd.randint(0,9)  
   
        # Pipeline 1 
        if(augment_rd==0):
            file_name_aug = output_aug_dir + file_name.split('/')[-1]
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
            except:
                pass

        # Pipeline 2 
        if(augment_rd==1):
            file_name_aug = output_aug_dir + file_name.split('/')[-1]
            try:
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=4)
                    .filter(filter_name='afftdn')
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.7,ratio=15,attack=15,release=100)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=100,decays=1)
                    .output(file_name_aug,acodec='flac')
                    .overwrite_output()
                    .run(quiet=True)
                    )
            except:
                pass

        # Pipeline 3
        if(augment_rd==2):
            file_name_aug = output_aug_dir + file_name.split('/')[-1]
            try:
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=6)
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.5,ratio=10,attack=10,release=50)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=200,decays=1)
                    .output(file_name_aug,acodec='flac')
                    .overwrite_output()
                    .run(quiet=True)
                    )
            except:
                pass        
 

        # Pipeline 4
        elif(augment_rd==3):
            file_name_aug = output_aug_dir + file_name.split('/')[-1].replace('000.flac','000.aac')
            try:
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=2)
                    .filter(filter_name='afftdn')
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.8,ratio=20,attack=20,release=200)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=50,decays=1)
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
            
            except:
                pass
        # Pipeline 5
        elif(augment_rd==4):
            file_name_aug = output_aug_dir + file_name.split('/')[-1].replace('000.flac','000.aac')
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
            
            except:
                pass
        # Pipeline 6
        elif(augment_rd==5):
            file_name_aug = output_aug_dir + file_name.split('/')[-1].replace('000.flac','000.aac')
            try:
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=6)
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.5,ratio=10,attack=10,release=50)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=200,decays=1)
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
            
            except:
                pass             

        # Pipeline 7   
        elif(augment_rd==6):
            try:
                file_name_aug = output_aug_dir + file_name.split('/')[-1].replace('000.flac','000.mp3')
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=2)
                    .filter(filter_name='afftdn')
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.8,ratio=20,attack=20,release=200)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=50,decays=1)
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
            except:
                pass
        # Pipeline 8   
        elif(augment_rd==7):
            try:
                file_name_aug = output_aug_dir + file_name.split('/')[-1].replace('000.flac','000.mp3')
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=4)
                    .filter(filter_name='afftdn')
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.7,ratio=15,attack=15,release=100)
                    .filter(filter_name='aecho',in_gain=1,out_gain=0.8,delays=100,decays=1)
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
            except:
                pass
        # Pipeline 9   
        elif(augment_rd==8):
            try:
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
            except:
                pass

        # Pipeline 10   
        elif(augment_rd==9):
            try:
                file_name_aug = output_aug_dir + file_name.split('/')[-1].replace('000.flac','000.mp3')
                ffmpeg_proc = (ffmpeg
                    .input(file_name)  
                    .filter(filter_name='volume',volume=3)
                    .filter(filter_name='firequalizer',gain_entry='entry(125,0);entry(250,-5);entry(1000,-2.5);entry(6000,3);entry(7500,0)')
                    .filter(filter_name='acompressor',threshold=0.8,ratio=20,attack=20,release=200)
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
            except:
                pass
        
        
        print("Files: {}/{} ".format(i+1,toteal_file)) 

        seed= seed+1
        i=i+1
  

if __name__ == '__main__':
      
        files_name_test = get_filenames_and_labels_test('/export/home/2368985c/MSc_Project_Sound_Augmentation/csv_file/','/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_eval_tmp/')
        print('Generating augmented test set')
        get_augmentation_data(files_name_test)
        
