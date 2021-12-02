import tensorflow.keras as keras
from inputGen import midiToPianoRoll, createSeqTestInputs, seqToPianoRoll, pianoRollToMidi

import numpy as np
import time
import random
import os

pwd = os.getcwd()

# CONFIG VARIABLES
DATASET_FILE_PATH = pwd + "\\music_dataset"
X_SEQ_LENGTH = 50


def obtainMidiFiles(dataDir):
    midi_files = []
    for path, subdirs, files in os.walk(dataDir):
        for file in files:
            if file[-3:] == "mid":
                filePath = os.path.join(path, file)
                # print(filePath)
                midi_files.append(filePath)

    return midi_files

# Music generator class which allows us
# 1. To take test_input_music and generate music for the given test_input_music
# 2. Saves the generated music file in designated FilePath using naming convention as a string composed of year, month, date and time

class MusicGenerator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

    def run(self, x_test):
        generatedFile = 'LSTM_gen_music-' + time.strftime("%Y_%m_%d_%H_%M")
        generatedFilePath = pwd + "\\output\\" + generatedFile + ".mid"
        print(generatedFilePath)

        for i, song in enumerate(x_test):
            print("Test Shape:", song.shape)
            genOutput = self.model.predict(song)
            print("Seq_Output:", np.array(genOutput.shape))
            genPianoRoll = seqToPianoRoll(genOutput)
            print("Generated PianoRoll:", np.array(genPianoRoll.shape))
            pianoRollToMidi(genPianoRoll, generatedFilePath)


if __name__ == '__main__':

    midiFiles = obtainMidiFiles(DATASET_FILE_PATH)

    fileIdx = random.randint(0, len(midiFiles) - 1)

    # Choosing a random midi file from the dataset for testing
    testMidiFile = midiFiles[fileIdx]

    print(testMidiFile)
    # Creating test inputs for model prediction
    testPianoRoll = midiToPianoRoll(testMidiFile)
    testPianoRollList = [testPianoRoll]
    x_test = createSeqTestInputs(testPianoRollList, X_SEQ_LENGTH)

    # Inputting model to be used
    MODEL_NAME = input("Enter model name:")
    MODEL_PATH = pwd + "\\models\\" + MODEL_NAME + ".h5"
    print(MODEL_PATH)

    if os.path.exists(MODEL_PATH):
        print("Model Exist!")
        mg = MusicGenerator(MODEL_PATH)
        mg.run(x_test)
    else:
        print("No such model exist!")
