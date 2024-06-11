import glob
import os
import shutil
import zipfile

import gdown
import librosa
import numpy as np
import pandas as pd
import parselmouth
from praatio.utilities import textgrid_io
from tqdm.auto import tqdm
from unidecode import unidecode
import soundfile as sf
from acoustic_features import extract_feature_means


# --------------------------------------------------------------------------- #
# Parse TextGrid
# Read the TextGrid file, keep the interval that text != ""
# and get the label from that interval, also try to store a set of possible phonemes in labels
data = []

sr = 16000


def process_textgrid_and_audio(textgrid_file):
    try:
        labels = []
        # Open TextGrid file using Parselmouth
        textgrid_data = parselmouth.Data.read(textgrid_file)
        textgrid_data.save_as_text_file("textgrid.txt")
        text = open("textgrid.txt", "r", encoding="utf-16").read()

        textgrid_data = textgrid_io.parseTextgridStr(
            text,
            includeEmptyIntervals=False,
        )

        entries = []
        for tier in textgrid_data["tiers"]:
            for entry in tier["entries"]:
                entries.append(entry)

        if len(entries) > 0:
            for i in range(0, len(entries)):
                labels.append(
                    {
                        "file_name": textgrid_file,
                        "start": entries[i].start,
                        "end": entries[i].end,
                        "phoneme": entries[i].label,

                    }
                )

        # Get the first 'start' value
        first_start = float(labels[0]['start'])

        # Get the last 'end' value
        last_end = float(labels[-1]['end'])

        # Concatenate 'phoneme' into a single string
        phoneme_string = list(item['phoneme'] for item in labels)

        # Load the audio file
        audio_file = labels[0]['file_name'].replace('.TextGrid', '.wav')

        # Load the audio
        y, sr = librosa.load(audio_file, sr=16000)

        # Calculate sample indices for t1 and t2
        start_idx = int(first_start * sr)
        end_idx = int(last_end * sr)

        # Extract the desired segment
        desired_segment = y[start_idx:end_idx]

        sf.write(
            labels[0]['file_name'].replace('.TextGrid', '_test.wav'), desired_segment, sr
        )
        data.append(
            {
                "file_name": str(textgrid_file).replace('.TextGrid', '_test.wav'),
                "phoneme": phoneme_string,
            }
        )

    except Exception as e:
        labels = []

        textgrid_data = parselmouth.Data.read(textgrid_file)
        textgrid_data.save_as_text_file("textgrid.txt")
        text = open("textgrid.txt", "r", encoding="ISO-8859-1").read()
        text = unidecode(text, "utf-8")
        text = text.replace("\x00", "")

        textgrid_data = textgrid_io.parseTextgridStr(
            text,
            includeEmptyIntervals=False,
        )

        entries = []
        for tier in textgrid_data["tiers"]:
            for entry in tier["entries"]:
                entries.append(entry)

        if len(entries) > 0:
            for i in range(0, len(entries)):
                labels.append(
                    {
                        "file_name": str(textgrid_file),
                        "start": entries[i].start,
                        "end": entries[i].end,
                        "phoneme": entries[i].label,

                    }
                )

        # Get the first 'start' value
        first_start = float(labels[0]['start'])

        # Get the last 'end' value
        last_end = float(labels[-1]['end'])

        # Concatenate 'phoneme' into a single string
        phoneme_string = list(item['phoneme'] for item in labels)

        # Load the audio file
        audio_file = labels[0]['file_name'].replace('.TextGrid', '.wav')

        # Load the audio
        y, sr = librosa.load(audio_file, sr=16000)

        # Calculate sample indices for t1 and t2
        start_idx = int(first_start * sr)
        end_idx = int(last_end * sr)

        # Extract the desired segment
        desired_segment = y[start_idx:end_idx]

        sf.write(
            labels[0]['file_name'].replace('.TextGrid', '_test.wav'), desired_segment, sr
        )
        data.append(
            {
                "file_name": str(textgrid_file).replace('.TextGrid', '_test.wav'),
                "phoneme": phoneme_string,
            }
        )


folder_path = "bahnaric/dataset/"
file_extension = ".TextGrid"


# Traverse the directory
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file ends with ".TextGrid"
        if file.endswith(file_extension):
            # If so, print the absolute path of the file
            file_path = os.path.join(root, file)
            process_textgrid_and_audio(file_path)

data = pd.DataFrame(data)
data.to_csv("experiment_data.csv", index=False, encoding="utf-8")
print("here")