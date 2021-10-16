import math
import pyaudio

PyAudio = pyaudio.PyAudio  # initialize pyaudio

BITRATE = 44100
AMPLITUDE = 127


def createNote(FREQUENCY, LENGTH, RESTLENGTH, BITRATE):
    _WAVEDATA = ''

    if FREQUENCY > BITRATE:
        BITRATE = FREQUENCY+100

    NUMBEROFFRAMES = int(BITRATE * LENGTH)
    RESTFRAMES = int(BITRATE * RESTLENGTH)

    if NUMBEROFFRAMES == 0:
        NUMBEROFFRAMES = 1

    CURRENT_AMPLITUDE = AMPLITUDE
    AMPLITUDE_DECREASE = AMPLITUDE / NUMBEROFFRAMES

    # generating waves
    for x in range(NUMBEROFFRAMES):
        _WAVEDATA += chr(int(CURRENT_AMPLITUDE *
                             math.sin((2 * math.pi * FREQUENCY * x) / BITRATE)) + 128)
        CURRENT_AMPLITUDE -= AMPLITUDE_DECREASE

    # generating rest frames
    for x in range(RESTFRAMES):
        _WAVEDATA += chr(128)

    return _WAVEDATA


def play(BITRATE, WAVEDATA):
    p = PyAudio()

    stream = p.open(format=p.get_format_from_width(1),
                    channels=1,
                    rate=BITRATE,
                    output=True)

    for data in WAVEDATA:
        stream.write(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


WAVEDATA = []

WAVEDATA.append(createNote(261.63, 0.5, 0.05, BITRATE))  # C4
WAVEDATA.append(createNote(261.63, 0.5, 0.05, BITRATE))  # C4
WAVEDATA.append(createNote(392, 0.5, 0.05, BITRATE))  # G4
WAVEDATA.append(createNote(392, 0.5, 0.05, BITRATE))  # G4
WAVEDATA.append(createNote(440, 0.5, 0.05, BITRATE))  # A4
WAVEDATA.append(createNote(440, 0.5, 0.05, BITRATE))  # A4
WAVEDATA.append(createNote(392, 1, 0.05, BITRATE))  # G4

WAVEDATA.append(createNote(349.23, 0.5, 0.05, BITRATE))  # F4
WAVEDATA.append(createNote(349.23, 0.5, 0.05, BITRATE))  # F4
WAVEDATA.append(createNote(329.63, 0.5, 0.05, BITRATE))  # E4
WAVEDATA.append(createNote(329.63, 0.5, 0.05, BITRATE))  # E4
WAVEDATA.append(createNote(293.66, 0.5, 0.05, BITRATE))  # D4
WAVEDATA.append(createNote(293.66, 0.5, 0.05, BITRATE))  # D4
WAVEDATA.append(createNote(261.63, 1, 0.05, BITRATE))  # C4

play(BITRATE, WAVEDATA)
