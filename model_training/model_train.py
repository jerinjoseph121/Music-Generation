from socket import INADDR_ALLHOSTS_GROUP
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, RepeatVector, TimeDistributed, BatchNormalization, Input
from keras.callbacks import EarlyStopping, History
from inputGen import INPUT_DIM, getData, createSeqTrainInputs
import matplotlib.pyplot as plt
import numpy as np
import os

pwd = os.getcwd()

# CONFIG VARIABLES
DATASET_FILE_PATH = pwd + "\\music_dataset\\sigurour_skuli"
NUM_UNITS = [512, 256, 128, 64]
EPOCHS = 500
# BATCH_SIZE = 64
BATCH_SIZE = INPUT_DIM
OUTPUT_DIM = INPUT_DIM  # 85
LEARNING_RATE = 0.01
X_SEQ_LENGTH = 50
Y_SEQ_LENGTH = 50
LOSS_FUNCTION = 'categorical_crossentropy'
OPTIMIZER = keras.optimizers.Adam()


# Building the model
def build_model1(x_train):
    model = Sequential()

    # Encoder
    model.add(LSTM(NUM_UNITS[3], batch_input_shape=(
        None, X_SEQ_LENGTH, INPUT_DIM), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(NUM_UNITS[3]))

    # Decoder
    model.add(RepeatVector(Y_SEQ_LENGTH))

    num_layers = 2

    for i in range(num_layers):
        model.add(LSTM(NUM_UNITS[3], return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(OUTPUT_DIM, activation='softmax')))

    model.build(x_train.shape)
    model.summary()

    # Compiling the model
    model.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER)

    return model


def build_model2(x_train):

    model = Sequential()

    # Phase 1
    model.add(LSTM(NUM_UNITS[0], batch_input_shape=(
        None, X_SEQ_LENGTH, INPUT_DIM), return_sequences=True))
    model.add(Dropout(0.3))

    # Phase 2
    model.add(Bidirectional(LSTM(NUM_UNITS[0], return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Phase 3
    model.add(TimeDistributed(Dense(NUM_UNITS[1], activation='relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Final Phase
    model.add(TimeDistributed(Dense(OUTPUT_DIM, activation='softmax')))

    model.build(x_train.shape)
    model.summary()

    # Compiling the model
    model.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER, metrics=['accuracy'])

    return model


# Training the model
def train(modelPath):
    # Preparing the data
    pianoRollData = getData(DATASET_FILE_PATH)
    print("Dataset files:", len(pianoRollData))
    x_train, y_train = createSeqTrainInputs(
        pianoRollData, X_SEQ_LENGTH, Y_SEQ_LENGTH)
    x_train = x_train.astype(bool)
    y_train = y_train.astype(bool)

    print("X-Train:", np.array(x_train.shape))
    print("Y-Train:", np.array(y_train.shape))

    # Buiilding the model
    print("Model Building")
    model = build_model2(x_train)

    print("Training the new model")
    print("Model training started")
    print("#################################################")

    earlystop = EarlyStopping(
        monitor='loss', patience=10, min_delta=0.01, verbose=0, mode='auto')
    history = History()

    hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                     epochs=EPOCHS, callbacks=[earlystop, history])

    # Saving the model
    model.save(modelPath)

    return hist


if __name__ == "__main__":
    # Inputting model name
    MODEL_NAME = input("Enter model name:")
    MODEL_PATH = pwd + "\\models\\" + MODEL_NAME + ".h5"
    # print(MODEL_PATH)

    # model = keras.models.load_model(MODEL_PATH)

    # print(model.history)
    modelHistory = train(MODEL_PATH)

    plt.plot(modelHistory.history['loss'])
    plt.title("Loss Function of LSTM Model")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    PLOT_PATH = pwd + "\\models\\loss_functions\\" + MODEL_NAME + ".jpg"
    # print(PLOT_PATH)

    plt.savefig(PLOT_PATH)

    plt.show()
