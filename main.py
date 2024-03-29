import os
from spleeter.separator import Separator
from pydub import AudioSegment
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import shutil
import random
# GLOBAL VARIABLES
segment_duration = 5 * 1000
cwd = os.getcwd()
artists = ['JustinBieber', 'ArianaGrande', 'TaylorSwift']
dir_base_for_database = './voice_conversion/src'
database_path = dir_base_for_database + "/database"
os.makedirs(database_path, exist_ok=True)
input = './raw_audio'
vocal = './vocal_separation'
os.makedirs(vocal, exist_ok=True)

def prepare_data_for_voice_conversion(database_path, 
                                      artists, 
                                      num_segments,
                                      vocal_directory=vocal, 
                                      segment_duration=segment_duration):
    """
    For every vocal piece, divide into small portions at configurable lengths for effective training
    Move vocal pieces into vocal_conversion model directory, classified into corresponding artists

    Parameters
    ----------
    database_path: string
        the directory where we store our 30-second audio pieces 
        for voice conversion models, categorized by artists
    artists: list[string]
        list of artists involved in the model
    vocal_directory: string
        the directory where all vocal separated pieces are stored
    num_songs: int
        number of songs per artist to train the model on 
        (for the sake of experimentally early training the model)
    segment_duration: int
        standard length of each audio piece fed into voice_conversion models
    
    Returns
    --------
    None
    """
    for i, artist in enumerate(artists):
        speaker = database_path + f"/spkr_{i + 1}"
        # make an artist directory under database directory for voice_conversion models
        os.makedirs(speaker, exist_ok=True)
        artist_vocal = os.path.join(vocal_directory, artist, 'vocals')
        # constrain number of songs trained on
        for f in os.listdir(artist_vocal):
            if f.lower().endswith(".wav"):
                # divide into smaller data chunks, standardize data length
                song = AudioSegment.from_file(os.path.join(artist_vocal, f), format="wav")
                # upper bound for starting point sampling
                max_start_point = len(song) - segment_duration
                for j in range(num_segments):
                    # randomly choose a start point
                    start_point = random.randint(0, max_start_point)
                    # sample a random 30-second portion of the vocal piece
                    segment = song[start_point:start_point + segment_duration]
                    # export into wav files to correct artist directory under voice_conversion
                    segment.export(os.path.join(speaker, f[:-4] + f"{j}_segment_{start_point}.wav"), format="wav")

def source_separation(artists, 
                      input_directory=input, 
                      output_directory=vocal):
    """"
    Takes some artists' audio pieces under an input directory
    Performs some source separation to get music and vocal out
    Move all vocal audio pieces to a different output directory, classifed into corresponding artists

    Parameters
    ----------
    artists: list[string]
        List of targeted artists involved in the model. 
        The names of artist will be included in audio pieces' name as well to ensure good labeling
    
    input_directory: string
        Directory where all raw songs are stored

    output_directory: string
        Directory where all vocal separated audio pieces are stored
    
    Returns
    -------
    None
    """
    # Initialize the Spleeter separator
    separator = Separator('spleeter:2stems')

    # Iterate through each artist
    for artist in artists:
        artist_input_directory = os.path.join(input_directory, artist)
        artist_output_directory = os.path.join(output_directory, artist)

        # Create the output directory for the current artist if it doesn't exist
        os.makedirs(artist_output_directory, exist_ok=True)
        
        artist_vocals_directory = os.path.join(artist_output_directory, 'vocals')
        artist_music_directory = os.path.join(artist_output_directory, 'music')

        # Create directories for vocals and music
        os.makedirs(artist_vocals_directory, exist_ok=True)
        os.makedirs(artist_music_directory, exist_ok=True)

        # Iterate through each WAV file in the artist's directory
        for filename in os.listdir(artist_input_directory):
            if filename.endswith('.wav'):
                input_file = os.path.join(artist_input_directory, filename)

                # Use Spleeter to separate vocals and music and save the results to a temporary directory
                separator.separate_to_file(input_file, artist_output_directory)
                print(f'Separated vocals and music for {artist} from {filename}')
                
                vocal = artist_output_directory + '/' + filename[:-4] + '/vocals.wav'
                music = artist_output_directory + '/' + filename[:-4] + '/accompaniment.wav'
                shutil.move(vocal, os.path.join(artist_vocals_directory, filename))
                shutil.move(music, os.path.join(artist_music_directory, filename))
        
                print('File move completed.')

    print('Separation completed.')

def convertWAV(artists, input_directory=input):
    """
    Converts the mp3 audio file into wav type for better audio quality
    Saves the wav file into the same artist-corresponding folder
    Removes the redundant mp3 file
    
    Parameters
    ----------
    artists: list[string]
        List of targeted artists involved in the model. 
        The names of artist will be included in audio pieces' name as well to ensure good labeling
    
    input_directory: string
        Directory where all raw songs are stored
    
    Returns
    -------
    None
    """
    for artist in artists:
        path = os.path.join(input_directory, artist)
        for f in os.listdir(path):
            # for each song in mp3 type
            if f.lower().endswith(".mp3"):
                output_file = f[:-4] + ".wav"
                if output_file not in os.listdir(path):
                    # convert into wav type, save to same directory, delete the old mp3 file
                    audio = AudioSegment.from_mp3(os.path.join(path, f))
                    audio.export(os.path.join(path, output_file), format="wav")
                    os.remove(os.path.join(path, f))
                    
convertWAV(artists)
source_separation(artists)
prepare_data_for_voice_conversion(database_path, artists, 30)