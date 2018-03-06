import librosa
import numpy as np
import scipy.io.wavfile as wav

from extract_pyAA_features import get_features_from_wav

class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class LibrosaExtractor(FeatureExtractor):
    """
    Extracts features MFCC and FBANK features by frames for .wav file

    https://www.kaggle.com/ybonde/log-spectrogram-and-mfcc-filter-bank-example

    frame_sec: seconds in frame (0.5)

    LEN -- samples num in .wav file (176000)
    FRAME_SIZE -- samples num in frame (8000)
    FRAME_STEP -- samples num between successive frames (1600)

    num_of_frames = (LEN - FRAME_SIZE) / FRAME_STEP = 105

    len MFCC feature vector for each frame: 20
    len FBANK feature vector for each frame: 128

    So extract_features returns np.array with dtype=float and shape=(105, 148)
    """
    def __init__(self, frame_sec=0.5, frame_step=1600):
        self.frame_sec = frame_sec
        self.frame_step = 1600

    def extract_features(self, wav_path):
        rate, audio = wav.read(wav_path)
        LEN = audio.shape[0]
        FRAME_SIZE = int(rate * self.frame_sec)
        FRAME_STEP = self.frame_step

        mfcc = np.array([
            np.mean(
                librosa.feature.mfcc(
                    audio[i : i + FRAME_SIZE].astype(float),
                    rate).T,
                axis=0
                )
                        for i in range(0, LEN - FRAME_SIZE, FRAME_STEP)
                        ])

        fbank = np.array([
            np.mean(
                librosa.feature.melspectrogram(
                    audio[i : i + FRAME_SIZE].astype(float),
                    rate).T,
                axis=0
                )
                        for i in range(0, LEN - FRAME_SIZE, FRAME_STEP)
                        ])

        return np.concatenate((mfcc, fbank), axis=1)


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""
    def __init__(self, frame_sec=0.1):
        self.frame_sec = frame_sec

    def extract_features(self, wav_path):
        return get_features_from_wav(wav_path, self.frame_sec)
