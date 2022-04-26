from mido import MidiFile, Message, MetaMessage, MidiTrack
from mido.midifiles.tracks import MidiTrack
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import os

import time

pwd = os.getcwd()

# CONFIG VARIABLES
TIME_PER_TIME_SLICE = 0.02  # No. of seconds to be in a single column
MICROSECONDS_PER_MINUTE = 60000000  # 60sec * 100 * 100
HIGHEST_NOTE = 105  # Highest note considered is A7
LOWEST_NOTE = 21  # Lowest note considered is A0
INPUT_DIM = HIGHEST_NOTE - LOWEST_NOTE + 1  # 85


# Converting a midi file into its piano roll form
def midiToPianoRoll(filePath):
    midiData = MidiFile(filePath, clip=True)
    # try:
    #     midiData = MidiFile(filePath, clip=True)
    # except:
    #     print("Error: In filepath ", filePath)
    #     return

    RESOLUTION = midiData.ticks_per_beat  # Resolution is the ticks per beat

    # gives a list of all tempo events(tempo here is in microseconds per beat and not in BPM)
    tempo_events = [x for track in midiData.tracks for x in track if str(
        x.type) == 'set_tempo']

    # Obtaining tempo in beats per minute format
    if(len(tempo_events) != 0):
        TEMPO = MICROSECONDS_PER_MINUTE/tempo_events[0].tempo
    else:
        TEMPO = 120

    TEMPO_IN_SECONDS = TEMPO/60  # tempo_in_seconds is the beats per second

    # Number of ticks we are considering for every column
    TICKS_PER_TIME_SLICE = RESOLUTION * TEMPO_IN_SECONDS * TIME_PER_TIME_SLICE

    totalTicks = 0

    # Obtaining the maximum total ticks in every track by adding the ticks in every event in a track
    for track in midiData.tracks:
        sum = 0
        for e in track:
            if str(e.type) == 'note_on' or str(e.type) == 'note_off' or str(e.type) == 'end_of_track':
                sum += e.time
        #print('Sum: ', sum)
        if sum > totalTicks:
            totalTicks = sum

    # Time Slice is the Number of columns to be added into the piano roll
    timeSlice = int(math.ceil(totalTicks/TICKS_PER_TIME_SLICE))

    # PainoRoll of dim (No. of notes X Time Slice)
    pianoRoll = np.zeros((INPUT_DIM, timeSlice), dtype=int)

    noteStates = {}  # Note State Dictionary is used to map a note which was pressed with the column when it was pressed

    for track in midiData.tracks:
        tickPoint = 0  # Tick point variable is used to point at a particular tick as when a note is pressed or released
        for event in track:
            # Condition when the note is pressed
            if str(event.type) == 'note_on' and event.velocity > 0:
                # Adding the time to the tick point since now it points after event.time
                tickPoint += event.time
                # Obtaining the column index
                colIdx = int(tickPoint/TICKS_PER_TIME_SLICE)

                if event.note <= HIGHEST_NOTE and event.note >= LOWEST_NOTE:
                    rowIdx = event.note - LOWEST_NOTE  # Obtaining the row index
                    pianoRoll[rowIdx][colIdx] = 1
                    noteStates[rowIdx] = colIdx

            # Condition when the note is released or not pressed
            elif str(event.type) == 'note_off' or (str(event.type) == 'note_on' and event.velocity == 0):
                # Adding the time to the tick point since now it points after event.time
                tickPoint += event.time
                # Obtaining the column index
                colIdx = int(tickPoint/TICKS_PER_TIME_SLICE)
                rowIdx = event.note - LOWEST_NOTE  # Obtaining the row index

                # This condition is used to assign the value 1 from the start column when
                # the note was pressed to the end column when the note was released
                if rowIdx in noteStates:
                    # StartColIdx is the initial colIdx when the note was pressed
                    startColIdx = noteStates[rowIdx]
                    pianoRoll[rowIdx][startColIdx: colIdx] = 1
                    del noteStates[rowIdx]

    ######################################## DEBUG ########################################

    # print(midiData.tracks[0])
    # print("Resolution(TPB): ", RESOLUTION)
    # print("Tempo Event List: ", tempo_events)
    # print("Tempo(BPM): ", TEMPO)
    # print("Ticks per time slice: ", TICKS_PER_TIME_SLICE)
    # print("Total Ticks: ", totalTicks)
    # print("Time Slice(No. of Columns): ", timeSlice)
    # print("Input dimension(No. of Rows): ", INPUT_DIM)

    #######################################################################################

    return pianoRoll.T


# Converting a piano roll format into a midi file format
def pianoRollToMidi(pianoRoll, filePath):
    # All these values are chosen arbitarily and can be changed to get optimal results
    TICKS_PER_TIME_SLICE = 1
    TEMPO = 1 / TIME_PER_TIME_SLICE
    RESOLUTION = 60 * TICKS_PER_TIME_SLICE
    VELOCITY = 90

    mid = MidiFile(ticks_per_beat=int(RESOLUTION))
    track = MidiTrack()
    mid.tracks.append(track)
    # Here tempo is converted from BPM to Microseconds per beat and added as an event
    track.append(MetaMessage('set_tempo', tempo=int(
        MICROSECONDS_PER_MINUTE/TEMPO), time=0))

    prevTimeSlice = np.zeros(INPUT_DIM)
    prevTimeSliceIdx = 0

    for timeSliceIdx, timeSlice in enumerate(np.concatenate((pianoRoll, np.zeros((1, INPUT_DIM))), axis=0)):
        # print(timeSliceIdx)
        timeSliceChange = timeSlice - prevTimeSlice

        for noteIdx, noteChange in enumerate(timeSliceChange):
            if noteChange == 1:
                noteEvent = Message('note_on', time=(
                    timeSliceIdx - prevTimeSliceIdx) * TICKS_PER_TIME_SLICE, velocity=VELOCITY, note=noteIdx + LOWEST_NOTE)
                track.append(noteEvent)
                prevTimeSliceIdx = timeSliceIdx

            if noteChange == -1:
                noteEvent = Message('note_off', time=(
                    timeSliceIdx - prevTimeSliceIdx) * TICKS_PER_TIME_SLICE, velocity=VELOCITY, note=noteIdx + LOWEST_NOTE)
                track.append(noteEvent)
                prevTimeSliceIdx = timeSliceIdx

        prevTimeSlice = timeSlice

    track.append(MetaMessage('end_of_track', time=1))
    mid.save(filePath)


# Obtaining data from dataset
def getData(dataDir):
    pianoRollData = []
    for path, subdirs, files in os.walk(dataDir):
        for file in files:
            if file[-3:] == "mid":
                filePath = os.path.join(path, file)
                # print(filePath)
                pianoRoll = midiToPianoRoll(filePath)
                pianoRollData.append(pianoRoll)

    return pianoRollData


# Sequence data for training the model
def createSeqTrainInputs(pianoRolls_data, xSeqLength, ySeqLength):
    x = []
    y = []

    for pianoRoll in pianoRolls_data:
        pos = 0
        while pos + xSeqLength + ySeqLength < pianoRoll.shape[0]:
            x.append(pianoRoll[pos: pos + xSeqLength])
            y.append(pianoRoll[pos + xSeqLength: pos +
                               xSeqLength + ySeqLength])
            pos += xSeqLength

    X = np.array(x)
    Y = np.array(y)

    # random_X, random_Y = shuffle(X, Y)

    return X, Y


# Sequence data for testing the model (Only input sequence provided)
def createSeqTestInputs(pianoRolls_data, xSeqLength):
    x_test = []

    for pianoRoll in pianoRolls_data:
        x = []
        pos = 0
        while pos + xSeqLength < pianoRoll.shape[0]:
            x.append(pianoRoll[pos: pos + xSeqLength])
            pos += xSeqLength

        x_test.append(np.array(x))

    return np.array(x_test)


# Sequence to Piano Roll conversion
def seqToPianoRoll(output, threshold=0.1):
    pianoRoll = []
    for seqOut in output:
        for timeSlice in seqOut:
            idx = [i for i, t in enumerate(timeSlice) if t > threshold]
            pianoRollSlice = np.zeros(timeSlice.shape)
            pianoRollSlice[idx] = 1
            pianoRoll.append(pianoRollSlice)

    return np.array(pianoRoll)


# Function to plot the piano roll
def plotPianoRoll(pianoRoll):
    plt.plot(range(pianoRoll.shape[0]), np.multiply(np.where(pianoRoll > 0, 1, math.nan), range(
        pianoRoll.shape[1])), marker='_', markersize=0.25)
    plt.title("Piano Roll of the Midi File")
    plt.ylabel("Notes")
    plt.xlabel("Time Slice")
    plt.show()


if __name__ == "__main__":
    path = pwd + '\\music_dataset\\scales\\scale_a_major.mid'
    newPath = pwd + '\\music_dataset\\sigurour_skuli\\8.mid'
    print(path)

    pr = midiToPianoRoll(newPath)
    # print(pr)

    generatedFile = 'Test' + time.strftime("%Y_%m_%d_%H_%M")
    generatedFilePath = pwd + "\\output\\" + generatedFile + ".mid"
    print(generatedFilePath)
    pianoRollToMidi(pr, generatedFilePath)
    plotPianoRoll(pr)
