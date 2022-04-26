from mido import MidiFile, Message, MetaMessage, MidiTrack
from mido.midifiles.tracks import MidiTrack
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import os

import time

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
