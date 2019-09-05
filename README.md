# Exploring data augmentation for sound classification
This project aims to study the opportunity to apply data augmentation in sound classification then identify difficulties and limitations of techniques

## Usage
- [main.py](https://github.com/petchc/SoundAugmentation/blob/master/main.py) : Main function to run the entire experiment.
- [prepare_data.py](https://github.com/petchc/SoundAugmentation/blob/master/prepare_data.py) : Create list of filenames and labels, Generate minibatches, Reading the audio file, apply feature extraction and perform data augmentation. 
- [data_transformation.py](https://github.com/petchc/SoundAugmentation/blob/master/data_transformation.py) : Transform the raw audio file to into input examples.
- [vggish_slim.py](https://github.com/petchc/SoundAugmentation/blob/master/vggish_slim.py) : Define VGGish architecture, classification and logits layer, and process the VGGish checkpoint loading.
- [vggish_params.py](https://github.com/petchc/SoundAugmentation/blob/master/vggish_params.py) : Program and VGGish parameters.
- [Postprocessor.py](https://github.com/petchc/SoundAugmentation/blob/master/Postprocessor.py) : Perform PCA process.
- [gen_augment.py](https://github.com/petchc/SoundAugmentation/blob/master/gen_augment.py) : Generate augmented test set by 10 fixed pipelines.

## Tools
- [TensorFlow: VGGish](https://github.com/tensorflow/models/tree/master/research/audioset)
- [Google AudioSet](https://research.google.com/audioset/index.html)
  - [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt)
  - [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz)
## Datasets
- [Balanced train segments](https://www.dropbox.com/sh/r547ggvdivljt32/AACQjpGsEpquDZqSlgCQOUc-a?dl=0&preview=audio.zip)
- [Evaluation segments](https://www.dropbox.com/sh/r547ggvdivljt32/AACQjpGsEpquDZqSlgCQOUc-a?dl=0&preview=eval_segments.zip)
- [Augmented test set](https://drive.google.com/file/d/1-MOR4V1H3C0rXdyn0KYM2Or2Su5ghof2/view?usp=sharing)
## CSV files
- [Balanced train segments](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv)
- [Evaluation segments](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv)
- [Class labels indices](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv)
## Training command
This is example of training command. You can change the directory to your own directory path. 
- Normal mode. 
```python3 main.py train --csv_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/csv_file/ --dataset_train_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_train_tmp/ --vggish_checkpoint_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/vggish_ckpt/ --save_checkpoint_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/tmp/ --epoch=500 --batch_size=64```
- Augmentation mode.
```python3 main.py train --csv_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/csv_file/ --dataset_train_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_train_tmp/ --vggish_checkpoint_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/vggish_ckpt/ --save_checkpoint_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/tmp/ --epoch=500 --batch_size=64 --augmentation```

## Test command
This is example of training command. You can change the directory to your own directory path and you need to specify the model checkpoint to evaluate.
- Test with original test set.
```python3 main.py test --csv_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/csv_file/ --dataset_test_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_eval_tmp/ --vggish_checkpoint_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/vggish_ckpt/ --checkpoint_path=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/tmp/model_base_conv_3_epoch_18.ckpt --batch_size=64```
- Test with augmented test set.
```python3 main.py test_augmented --csv_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/csv_file/ --dataset_test_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_augmented_10_pipelines/ --vggish_checkpoint_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/vggish_ckpt/ --checkpoint_path=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/tmp/model_aug_conv_3_epoch_18.ckpt --batch_size=64```

## Generate augmented test set
```python3 main.py gen_augment.py```

## Inference command
This is example of inference command for prediction the labels of audio file.
```python3 main.py inference --csv_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/csv_file/ --file_path=/export/home/2368985c/MSc_Project_Sound_Augmentation/audio_eval_tmp/-/0/P/-0p7hKXZ1ww_30000_40000.flac --vggish_checkpoint_dir=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/vggish_ckpt/ --checkpoint_path=/export/home/2368985c/MSc_Project_Sound_Augmentation/SoundAugmentation/tmp/model_aug_conv_3_epoch_18.ckpt```
