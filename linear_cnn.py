import argparse
from email.policy import default
import pickle
import h5py
import os
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Activation, add, Conv2D, Dense, Dropout, Flatten, Input, ZeroPadding2D)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def positive_int(value):
    try:
        parsed = int(value)
        if not parsed > 0:
            raise ValueError()
        return parsed
    except ValueError:
        raise argparse.ArgumentTypeError('value must be an positive integer')


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e', '--epochs',
        nargs='?',
        type=positive_int,
        action='store',
        default=10,
        help='number of training epochs'
    )
    parser.add_argument(
        '-t', '--train',
        metavar='TRAIN',
        type=str,
        dest='train',
        default='./tested_synthetic_test.h5',  # './artificial.h5',
        help='path to the HDF5 file with the training data'
    )
    parser.add_argument(
        '-m', '--model',
        metavar='MODEL',
        type=str,
        dest='model',
        default='./latest_model_DenseReg.keras',
        help='path where to store the model'
    )
    parser.add_argument(
        '-s', '--saveflag',

        metavar='SAVEFLAG',
        type=bool,
        dest='save_flag',
        default=True,
        help='Flag to determine whether to enable checkpointing and writing to disk'
    )
    parser.add_argument(
        '-d', '--data',

        metavar='DATA',
        type=str,
        dest='data',
        default='matrix_of_64',
        help='--'
    )
    parser.add_argument(
        '-l', '--labels',

        metavar='LABEL',
        type=str,
        dest='labels',
        default='labels_for_64',
        help='--'
    )
    return parser.parse_args()


def load_data(path):
    with h5py.File(path, 'r') as handle:
        labels = np.array(handle['labels_for_64'])
        data = np.array(handle['matrix_of_64'])

        return data, labels


def preprocess(data, labels):
    # simply add an additional dimension for the channels for data
    return np.expand_dims(data, axis=3), np.moveaxis(labels, 0, -1)

# change parameters to take num outputs rather than labels
def build_model(input_shape, num_outputs=1):
    input_img = Input(shape=input_shape)

    # first bottleneck unit
    bn_1 = BatchNormalization()(input_img)
    activation_1 = Activation('selu')(bn_1)
    conv_1 = Conv2D(32, kernel_size=(5, 5,), padding='same', kernel_regularizer=l2(0.02))(activation_1)

    bn_2 = BatchNormalization()(conv_1)
    activation_2 = Activation('selu')(bn_2)
    conv_2 = Conv2D(128, kernel_size=(3, 3,), padding='same', kernel_regularizer=l2(0.02))(activation_2)

    merged = add([input_img, conv_2])

    # corner detection
    bn_3 = BatchNormalization()(merged)
    padding = ZeroPadding2D(padding=(0, 3))(bn_3)
    conv_3 = Conv2D(32, kernel_size=(21, 7,), padding='valid', activation='tanh')(padding)
    conv_4 = Conv2D(128, kernel_size=(1, 3,), padding='same', activation='tanh')(conv_3)

    # fully-connected predictor
    flat = Flatten()(conv_4)
    classify = Dense(512, activation='sigmoid')(flat)
    dropout = Dropout(0.1)(classify)

    # change activation to linear for regression
    result = Dense(1, activation='linear')(dropout)

    # change metric to mae from accuracy, because accuracy is for classification
    model = Model(inputs=input_img, outputs=result)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def train_network(model, data, labels, model_file, epochs, save_flag):
    #plot_model(model, to_file='{}.png'.format(model_file), show_shapes=True)

    if (save_flag):
        checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=True, save_best_only=True, save_weights_only=False, mode='auto')
        training = model.fit(data, labels, epochs=epochs, batch_size=8, validation_split=1.0/5.0, callbacks=[checkpoint])
    else:
        training = model.fit(data, labels, epochs=epochs, batch_size=8, validation_split=1.0/5.0)

    if (save_flag):
        with open('{}.history'.format(model_file), 'wb') as handle:
            pickle.dump(training.history, handle)

def evaluate_model(model, X_test, y_test):
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MAE: {test_mae}")

if __name__ == '__main__':
    arguments = parse_cli()
    print("complete (1)")
    data, labels = preprocess(*load_data(arguments.train)) #, arguments.data, arguments.labels
    print(f"Data shape is: {data.shape}")
    print(f"Labels shape is: {labels.shape}")

    data = np.transpose(data, (2,0,1,3))
    labels = labels.reshape(-1, 1)
    print(f"Data shape is: {data.shape}")
    print(f"Labels shape is: {labels.shape}")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = build_model(data.shape[1:],num_outputs=1)
    train_network(model, X_train, y_train, arguments.model, epochs=20, arguments.save_flag)
    evaluate_model(model, X_test, y_test)

    model_file = arguments.model
    with open('{}.history'.format(model_file), 'rb') as handle:
        history = pickle.load(handle)

    # Plot training & validation Mean Absolute Error
    plt.figure()
    plt.plot(history['mae'])
    plt.plot(history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig("DenseReg_mae.png")

    # Plot training & validation Loss (MSE)
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss (MSE)')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig("DenseReg_mse.png")
