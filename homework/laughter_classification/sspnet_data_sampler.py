import os
from os.path import join

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

from laughter_classification.utils import chunks, in_any, interv_to_range, get_sname
from laughter_prediction.feature_extractors import LibrosaExtractor

"""
Rewrote this class to get features instead of samples and frames with overlapping.
"""

class SSPNetDataSampler:
    """
    Class for loading and sampling audio data by frames for SSPNet Vocalization Corpus
    """

    @staticmethod
    def read_labels(labels_path):
        def_cols = ['Sample', 'original_spk', 'gender', 'original_time']
        label_cols = ["{}_{}".format(name, ind) for ind in range(6) for name in ('type_voc', 'start_voc', 'end_voc')]
        def_cols.extend(label_cols)
        labels = pd.read_csv(labels_path, names=def_cols, engine='python', skiprows=1)
        return labels

    def __init__(self, corpus_root):
        self.sample_rate = 16000
        self.duration = 11
        self.default_len = self.sample_rate * self.duration
        self.data_dir = join(corpus_root, "data")
        labels_path = join(corpus_root, "labels.txt")
        self.labels = self.read_labels(labels_path)

    def _interval_generator(self, incidents):
        for itype, start, end in chunks(incidents, 3):
            if itype == 'laughter':
                yield int(start * self.sample_rate), int(end * self.sample_rate)

    def get_labels_for_file(self, wav_path, frame_sec):
        sname = get_sname(wav_path)
        sample = self.labels[self.labels.Sample == sname]

        incidents = sample.loc[:, 'type_voc_0':'end_voc_5']
        incidents = incidents.dropna(axis=1, how='all')
        incidents = incidents.values[0]

        rate, audio = wav.read(wav_path)

        laught_by_sample = np.zeros((self.default_len, ), dtype=bool)
        for start, end in self._interval_generator(incidents):
            laught_by_sample[start:end+1] = 1

        FRAME_SIZE = int(frame_sec * self.sample_rate)
        FRAME_STEP = int(FRAME_SIZE / 5)

        is_laughter = np.array([
            laught_by_sample[i : i + FRAME_SIZE].sum() > FRAME_SIZE / 2
                   for i in range(0, self.default_len - FRAME_SIZE, FRAME_STEP)
                        ])

        df = pd.DataFrame({'IS_LAUGHTER': is_laughter,
                           'SNAME': sname})
        return df

    def df_from_file(self, wav_path, frame_sec):
        """
        Returns sampled data by path to audio file
        :param wav_path: string, .wav file path
        :param frame_sec: int, length of each frame in sec
        :return: pandas.DataFrame with sampled audio
        """

        extr = LibrosaExtractor()
        features = extr.extract_features(wav_path)
        labels = self.get_labels_for_file(wav_path, frame_sec)
        return pd.concat([pd.DataFrame(features), labels], axis=1)

    def get_valid_wav_paths(self):
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            fullpaths = [join(dirpath, fn) for fn in filenames]
            return [path for path in fullpaths if len(wav.read(path)[1]) == self.default_len]

    def create_sampled_df(self, frame_sec, naudio=None, save_path=None, force_save=False):
        """
        Returns sampled data for whole corpus
        :param frame_sec: int, length of each frame in sec
        :param naudio: int, number of audios to parse, if not defined parses all
        :param save_path: string, path to save parsed corpus
        :param force_save: boolean, if you want to override file with same name
        :return:
        """
        fullpaths = self.get_valid_wav_paths()[:naudio]
        dataframes = [self.df_from_file(wav_path, frame_sec) for wav_path in fullpaths]
        df = pd.concat(dataframes)

        colnames = ["V{}".format(i) for i in range(df.shape[1] - 2)]
        colnames.append("IS_LAUGHTER")
        colnames.append("SNAME")
        df.columns = colnames

        if save_path is not None:
            if not os.path.isfile(save_path) or force_save:
                print("saving df: ", save_path)
                df.to_csv(save_path, index=False)

        return df
