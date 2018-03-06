import argparse
import json

import numpy as np

from feature_extractors import LibrosaExtractor
from predictors import RnnPredictor


def predicted_to_intervals(pred_classes):
    RATE = 16000
    FRAME_SEC = 0.5
    FRAME_SIZE = int(RATE * FRAME_SEC)
    FRAME_STEP = int(FRAME_SIZE / 5)

    intervals = []
    start = -1

    for i, res in enumerate(pred_classes):
        if res:
            if start == -1:
                start = i * FRAME_STEP
        else:
            if start != -1:
                end = (i - 1) * FRAME_STEP + FRAME_SIZE
                intervals.append((start / float(RATE), end / float(RATE)))
                start = -1

    return intervals


def main():
    parser = argparse.ArgumentParser(description='Script for prediction laughter intervals for .wav file')
    parser.add_argument('--wav_path', type=str, help='Path to .wav file')

    args = parser.parse_args()

    extr = LibrosaExtractor()
    features = extr.extract_features(args.wav_path)

    model = RnnPredictor()
    pred_classes = model.predict(features[np.newaxis, :, :])[0]
    intervals = predicted_to_intervals(pred_classes)

    print("Target intervals")
    print(intervals)


if __name__ == '__main__':
    main()
