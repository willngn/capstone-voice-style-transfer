import sys
sys.path.append('/Users/ngttam3010/capstone-voice-style-transfer/voice_conversion')
sys.path.append('/Users/ngttam3010/capstone-voice-style-transfer/voice_conversion/src')
from src.utils import preprocess_wav, melspectrogram, reconstruct_waveform
import matplotlib.pyplot as plt
import os
import soundfile as sf
from src.params import sample_rate
import librosa.display

def vocalAudioToMelspectrogram(audio):
    """
    Produces a numerical mel-scaled spectrogram representation of an audio file
    Plots the mel-scaled spectrogram representation

    Parameters
    ----------
    audio: string
        the full path of the audio file we want to compute its spectrograms
    
    Returns
    -------
    spect: ndarray
        a numerical matrix of spectrograms to show the audio's frequencies changing over time
    """
    if audio[-3:] == 'wav':
        spect = melspectrogram(preprocess_wav(audio)) # the workflow in src/preprocess.py
        savePath = os.getcwd() + '/test/melspectrogram/' + audio.split('/')[-1][:-3]
        fig, ax = plt.subplots()
        img = librosa.display.specshow(spect, y_axis='log', x_axis='time', ax=ax)
        ax.set_title('Spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.savefig(savePath)
        return spect

def melspectrogramToAudio(spect, audio):
    """
    Reconstructs the wav audio file from its spectrogram representation and saves to a different folder

    Parameters
    ----------
    spect: ndarray
        a numerical matrix of spectrograms for the audio
    audio: string
        the audio file name represented by the spectrogram parameter
    
    Returns 
    -------
    None
    """
    wav = reconstruct_waveform(spect)
    audioPath = os.path.join(os.getcwd(), "test/reconstructed_vocal/" + audio)
    sf.write(audioPath, wav, sample_rate) # preserve the same sample rate in the voice conversion model

if __name__ == '__main__':
    # loop over all audio files we want to test
    for audio in os.listdir(os.path.join(os.getcwd(), "test/vocal")):
        # compute their spectrograms and convert back to wav audio files
        path = os.path.join(os.getcwd(), "test/vocal/") + audio
        spect = vocalAudioToMelspectrogram(path)
        melspectrogramToAudio(spect, audio)