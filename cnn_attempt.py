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
        default=1,
        help='number of training epochs'
    )
    parser.add_argument(
        '-t', '--train',
        metavar='TRAIN',
        type=str,
        dest='train',
        default='./synthetic_data.h5',  # './artificial.h5',
        help='path to the HDF5 file with the training data'
    )
    parser.add_argument(
        '-m', '--model',
        metavar='MODEL',
        type=str,
        dest='model',
        default='./latest_model.keras',
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
        default='matrix_of_50',
        help='--'
    )
    parser.add_argument(
        '-l', '--labels',

        metavar='LABEL',
        type=str,
        dest='labels',
        default='labels_for_50',
        help='--'
    )
    return parser.parse_args()



def load_data(path, data_name, label_name):
    # Load Data
    with h5py.File(path, 'r') as handle:
        data = np.array(handle[data_name])
        labels = np.array(handle[label_name])
        print(data.shape)
        print(labels.shape)
        return data, labels

def preprocess(data, labels):
    # Expand dimensions of data to include channel
    # (num_samples, height, width, 1)
    data_expanded = np.expand_dims(data.T, axis=-1)

    # Move axis for labels (if needed)
    # Should aready be fine (1000,) so leaving this for now
    labels_moved = np.moveaxis(labels, 0, -1)  # Change the shape of labels if required
    print(f'Preprocessing data shape: {data_expanded.shape}')
    print(f'Preprocessing labels shape: {labels_moved.shape}')
    return data_expanded, labels_moved


def build_model(input_shape):
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

    # had to change this. We only want one output, the predicted block size
    result = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=input_img, outputs=result)
    model.compile(optimizer=optimizers.Nadam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae', 'mse'])

    return model

def train_network(model, data, labels, model_file, epochs, save_flag):
    #plot_model(model, to_file='{}.png'.format(model_file), show_shapes=True)

    if (save_flag):
        checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=True, save_best_only=True, save_weights_only=False, mode='auto')
        training = model.fit(data, labels, epochs=epochs, batch_size=8, validation_split=1.0/5.0, class_weight={0: 0.1, 1: 0.9}, callbacks=[checkpoint])
    else:
        training = model.fit(data, labels, epochs=epochs, batch_size=8, validation_split=1.0/5.0, class_weight={0: 0.1, 1: 0.9})

    if (save_flag):
        with open('{}.history'.format(model_file), 'wb') as handle:
            pickle.dump(training.history, handle)


def evaluate_model(model, X_test, y_test):
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MAE: {test_mae}")
    print(f"Test MSE: {test_mse}")

if __name__ == '__main__':
    args = parse_cli()
    data, labels = preprocess(*load_data(args.train, args.data, args.labels))
    print(data.shape)  # Shape of the data
    print(labels.shape)  # Shape of the labels
    model = build_model((data.shape[1], data.shape[2], 1))
    model.summary()
    X_train, X_test, y_train, y_test = train_test_split(data, labels.T, test_size=0.2, random_state=42)
    train_network(model, X_train, y_train, args.model, epochs=5, save_flag=True)
    evaluate_model(model, X_test, y_test)

