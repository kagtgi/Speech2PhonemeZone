import glob
import multiprocessing
import multiprocessing as mp
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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists("bahnaric"):
    shutil.rmtree("bahnaric")
    
if os.path.exists("Am vi Ba Na"):
    shutil.rmtree("Am vi Ba Na")
# Unzip the ZIP file
with zipfile.ZipFile("Am vi Ba Na-20240202T004404Z-001.zip", "r") as zip_file:
    zip_file.extractall()
shutil.move("Am vi Ba Na", "bahnaric/dataset/raw")

print(" check 1 ")
# --------------------------------------------------------------------------- #
# Parse TextGrid
# Read the TextGrid file, keep the interval that text != ""
# and get the label from that interval, also try to store a set of possible phonemes in labels
data = []

sr=16000
def process_textgrid_and_audio(textgrid_file):
    """
    Extracts text from a TextGrid file and cuts a corresponding audio segment from a WAV file.
    Args:
        textgrid_file (str): Path to the TextGrid file.
        wav_file (str): Path to the WAV file.
    Returns:
        None (prints results to console).
    """

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
        print("preprocessing..." )

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
        phoneme_string = ''.join(item['phoneme'] for item in labels)

        # Load the audio file
        audio_file = labels[0]['file_name'].replace('.TextGrid', '.wav')
       
        # Load the audio
        y, sr = librosa.load(audio_file, sr=16000)

        # Calculate sample indices for t1 and t2
        start_idx = int(first_start * sr)
        end_idx = int(last_end * sr)

        # Extract the desired segment
        desired_segment = y[start_idx:end_idx]

        # Calculate padding lengths based on desired padding duration
        padding_length = int(1 * sr // 2)

        # Create padding arrays
        padding_before = np.zeros(padding_length)
        padding_after = np.zeros(padding_length)

        # Combine the desired segment, padding, and check for overflow
        padded_segment = np.concatenate((padding_before, desired_segment, padding_after))
        if padded_segment.shape[0] > y.shape[0]:
            padded_segment = padded_segment[:y.shape[0]]  # Truncate if overflow occurs

        
        sf.write(
            labels[0]['file_name'].replace('.TextGrid', '_cut.wav'), padded_segment, sr
            )
        data.append(
            {
                "file_name": textgrid_file,
                "start": first_start,
                "end": last_end,
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
        print("preprocessing..." )

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
        phoneme_string = ''.join(item['phoneme'] for item in labels)

        # Load the audio file
        audio_file = labels[0]['file_name'].replace('.TextGrid', '.wav')
       
        # Load the audio
        y, sr = librosa.load(audio_file, sr=16000)

        # Calculate sample indices for t1 and t2
        start_idx = int(first_start * sr)
        end_idx = int(last_end * sr)

        # Extract the desired segment
        desired_segment = y[start_idx:end_idx]

        # Calculate padding lengths based on desired padding duration
        padding_length = int(1 * sr // 2)

        # Create padding arrays
        padding_before = np.zeros(padding_length)
        padding_after = np.zeros(padding_length)

        # Combine the desired segment, padding, and check for overflow
        padded_segment = np.concatenate((padding_before, desired_segment, padding_after))
        if padded_segment.shape[0] > y.shape[0]:
            padded_segment = padded_segment[:y.shape[0]]  # Truncate if overflow occurs

        
        sf.write(
            
            labels[0]['file_name'].replace('.TextGrid', '_cut.wav'), padded_segment, sr
            )
        data.append(
            {
                "file_name": str(textgrid_file),
                "start": first_start,
                "end": last_end,
                "phoneme": phoneme_string,
            }
        )



def _par_features_generator(file_name: str):
    signal, sr = librosa.load(file_name, sr=16000)

    feature_dfs = []
    for k in range(5):
        # Define window size
        k = 75 + (1 + k) * 10
        assert k % 2 == 1, "k must be odd"

        # Break audio into frames
        frame_length = int(sr * 0.005)  # 5ms
        hop_length = int(sr * 0.001)  # 1ms
        frames = librosa.util.frame(
            signal, frame_length=frame_length, hop_length=hop_length
        )

        # Pad frames at the beginning and end
        padding = (k - 1) // 2
        padded_frames = np.pad(frames, ((0, 0), (padding, padding)), mode="edge")

        # Calculate features on sliding window of k frames
        features = []
        for i in range(padding, len(padded_frames[0]) - padding):
            window = padded_frames[:, i - padding : i + padding + 1]
            feature = extract_feature_means(signal=window.flatten(), sr=sr)
            features.append(feature)

        features = pd.concat(features, axis=0)
        features.columns = [f"{col}_w{str(k).zfill(3)}" for col in features.columns]
        feature_dfs.append(features)

    features = pd.concat(feature_dfs, axis=1)
    features.to_parquet(
        os.path.join(
            "bahnaric/features",
            file_name.split('\\')[-1].replace("wav", "parquet"),
        )
    )


if not os.path.exists("bahnaric/features"):
    # Create the folder
    os.makedirs("bahnaric/features")
folder_path = "bahnaric/dataset/raw"
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
data.to_csv("preprocess.csv", index=False, encoding="utf-8")
print("here!!!")

folder_path = "bahnaric/dataset/raw"
file_extension = "_cut.wav"

# Traverse the directory
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file ends with ".TextGrid"
        if file.endswith(file_extension):
            # If so, print the absolute path of the file
            
            file_path = os.path.join(root, file)
            _par_features_generator(file_path)

print("end")




