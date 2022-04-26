from mido import MidiFile, Message, MetaMessage, MidiTrack
from mido.midifiles.tracks import MidiTrack
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import os

import time

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
