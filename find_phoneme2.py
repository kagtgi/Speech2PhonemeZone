import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import patches

sys.path.append('..')
import libfmp.b
import libfmp.c3
import libfmp.c7
import pandas as pd

def get_wav_length(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Trim leading and trailing silence
    y_trimmed, _ = librosa.effects.trim(y)
    
    # Calculate the duration of the trimmed audio file in seconds
    duration = librosa.get_duration(y=y_trimmed, sr=sr)
    return duration

# Load the dataset
data = pd.read_csv('test_data.csv')
mean_length_df = pd.read_csv('mean_lengths.csv')


# Open a text file to write the results
with open('TestResult.txt', 'w', encoding='utf-8') as result_file:
    # Iterate over each row in the dataset
    for index, row in data.iterrows():
        input_file = row['file_name']
        given_phoneme = eval(row['phoneme'])
        input_length = get_wav_length(input_file)
        result_file.write("Full Length: " + str(input_length) + "\n")
        result_file.write("Input File: " + input_file + "\n")
        mean_length_list = [mean_length_df[mean_length_df['Phoneme'] == x] for x in given_phoneme]
        
        # Filter out any empty DataFrames
        mean_length_list = [x for x in mean_length_list if not x.empty]
        
        # Calculate sum of mean lengths
        sum_mean_lengths = sum([x['Mean_Wav_Length'].values[0] for x in mean_length_list])
        
        # Print time part for each phoneme
        for i in range(len(given_phoneme)):
            try:
                # Check if mean_length_list[i] is empty
                if not mean_length_list[i].empty:
                    result_file.write("Phoneme: " + str(given_phoneme[i]) + "\n")
                    result_file.write("Time part: " + str(mean_length_list[i]['Mean_Wav_Length'].values[0] * input_length / sum_mean_lengths) + "\n")
            except IndexError:
                result_file.write("IndexError occurred. Skipping this phoneme.\n")
                continue
