# Capstone: Singing Voice Style Transfer

## Introduction

The project tackles the problem of singing voice style transfer which involves swapping the vocal of one artist with another artist's vocal without changing the singing content. We provided a 2-step pipeline by first conducting source separation to extract only vocals outside of musical accompaniment, then performing voice conversion. Using the code of this project, you can train the model with new audio data (different artists) to experiment singing voice style transfer between multiple new artists beyond the ones used in the project (Justin Bieber, Taylor Swift, Ariana Grande). We use Spleeter with a U-Net model under the hood for source separation task and VAE-CycleGAN for voice conversion. 

## Project Structure

```
capstone-voice-style-transfer
│   README.md
│   requirements.txt    
│   main.ipynb
|
|____pretrained_models
|   |____2stems
|       |____checkpoint
|       |____model.data-00000-of-00001
|       |____model.index
|       |____model.meta
|
|____raw_audio
|   │____Ariana Grande
|       |____10 songs
|   |____Justin Bieber
|       |____10 songs
|   |____Taylor Swift
|       |____10 songs
|       
|____vocal_separation
|   │____Ariana Grande
|   |   |____music
|   |   |   |____accompaniment only of 10 songs
|   |   |
|   |   |____vocal
|   |       |___vocal only of 10 songs
|   |
|   │____Justin Bieber
|   |   |____music
|   |   |   |____accompaniment only of 10 songs
|   |   |
|   |   |____vocal
|   |       |___vocal only of 10 songs
|   |
|   │____Taylor Swift
|   |   |____music
|   |   |   |____accompaniment only of 10 songs
|   |   |
|   |   |____vocal
|   |       |___vocal only of 10 songs
|
|____voice_conversion
|   │____src
|   |   |____commandline.ipynb (where I run .py files)
|   |   |____data_proc.py
|   |   |____inference.py
|   |   |____models.py
|   |   |____params.py
|   |   |____preprocess.py
|   |   |____train.py
|   |   |____utils.py
|   |   |
|   |   |____database (preprocessed audio data for each artist under correct dir)
|   |   |   |____spkr_1 
|   |   |   |____spkr_2
|   |   |   |____spkr_3
|   |   |   |____pickle files of train, val, test data
|   |   |
|   |   |____out_infer (all inference results)
|   |   |   |____model_name
|   |   |   |   |____gen (generated wav)
|   |   |   |   |____plots (mel-spectrogram plots for input vs output)
|   |   |   |   |____ref (original wav on which voice style transfer is inferenced)
|   |   |
|   |   |____saved_models (trained models)
|   |   |   |____model_name
|   |   |   |   |____D1, D2, encoder, G1, G2 at specific epochs
|   |
|   │____test
|   |   |____non_ML_testing.py
|   |   |____melspectrogram
|   |   |   |____spectrogram plots from audio
|   |   |
|   |   |____reconstructed_vocal
|   |   |   |____wav files from spectrogram
|   |   |
|   |   |____vocal
|   |   |   |____original wav files
|   |
|   |____LICENSE (of the voice conversion model)
|   |____README.md
|   |____requirements.txt
```

## How To Use

First, collect songs you want to experiment with (mp3 or wav). Name the artist folder correspondingly and store data in the correct folder

To set up development:
```
pip3 install -r requirements.txt
```

To start preprocessing data and perform vocal separation, first adjust the global variables in capstone-voice-style-transfer/main.py as needed (sampling rate, segment length, artists included). Then run this:
```
python3 main.py
```

Navigate to Voice Conversion task and repo, using the inner README page to train, test split the data, train the model, hyperparameter tune and make inferences. Outputs and results are either logged or shown under ./voice_conversion/out_infer folder.

Using Google Colab is highly recommended to make use of CPU for efficient training.
```
To test data preprocessing within voice conversion model, navigate to ./voice_conversion. Make sure that ./voice_conversion and ./voice_conversion/src are within the sys.path for efficient imports. Under ./voice_conversion/test, run the following to produce Mel-spectrograms of an audio and reconstruct another audio from spectrograms:
```
python3 non_ML_testing.py
```