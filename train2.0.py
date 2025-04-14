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


def load_data(path):
    with h5py.File(path, 'r') as handle:
        labels = np.array(handle['labels_for_50'])
        data = np.array(handle['matrix_of_50'])

        return data, labels


def preprocess(data, labels):
    # simply add an additional dimension for the channels for data
    return np.expand_dims(data, axis=3), np.moveaxis(labels, 0, -1)


def build_model(input_shape, labels):
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
    classify = Dense(10, activation='sigmoid')(flat)
    dropout = Dropout(0.1)(classify)

    result = Dense(labels.shape[1], activation='sigmoid')(dropout)

    model = Model(inputs=input_img, outputs=result)
    model.compile(optimizer=optimizers.Nadam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_model_lr(input_shape, labels):
    input_img = Input(shape=input_shape)

    # first bottleneck unit
    #bn_1 = BatchNormalization()(input_img)
    #activation_1 = Activation('selu')(input_img)
    conv_1 = Conv2D(32, kernel_size=(5, 5,), padding='same', kernel_regularizer=l2(0.02))(input_img)

    #bn_2 = BatchNormalization()(conv_1)
    activation_2 = Activation('selu')(conv_1)
    conv_2 = Conv2D(128, kernel_size=(3, 3,), padding='same', kernel_regularizer=l2(0.02))(activation_2)

    #merged = add([input_img, conv_2])

    # corner detection
    #bn_3 = BatchNormalization()(merged)
    #padding = ZeroPadding2D(padding=(0, 3))(bn_3)
    #conv_3 = Conv2D(32, kernel_size=(21, 7,), padding='valid', activation='tanh')(conv_2)
    #conv_4 = Conv2D(128, kernel_size=(1, 3,), padding='same', activation='tanh')(conv_3)

    # fully-connected predictor
    flat = Flatten()(conv_2)
    classify = Dense(10, activation='sigmoid')(flat)
    #dropout = Dropout(0.1)(classify)

    result = Dense(labels.shape[1], activation='sigmoid')(classify)

    model = Model(inputs=input_img, outputs=result)
    model.compile(optimizer=optimizers.Nadam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_network(model, data, labels, model_file, epochs, save_flag):
    # plot_model(model, to_file='{}.png'.format(model_file), show_shapes=True)

    if (save_flag):
        checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=True, save_best_only=True,
                                     save_weights_only=False, mode='auto')
        training = model.fit(data,
                             labels,
                             epochs=epochs,
                             batch_size=8,
                             validation_split=0.2,
                             class_weight={0: 0.1, 1: 0.9},
                             callbacks=[checkpoint])
    else:
        training = model.fit(data,
                             labels,
                             epochs=epochs,
                             batch_size=8,
                             validation_split=0.2,
                             class_weight={0: 0.1, 1: 0.9}
                             )

    if (save_flag):
        with open('{}.history'.format(model_file), 'wb') as handle:
            pickle.dump(training.history, handle)


if __name__ == '__main__':
    arguments = parse_cli()
    print("complete (1)")
    data, labels = preprocess(*load_data(arguments.train)) #, arguments.data, arguments.labels
    print("complete (2)")
    labelsSimple = labels[:, 0]
    print("complete (3)")
    labelsSimple = np.expand_dims(labelsSimple, axis=1)  # Reshape into (n_samples, 1)
    print("complete (4)")
    model = build_model(data.shape[1:],labelsSimple)
    print("complete (5)")
    train_network(model, data, labelsSimple, arguments.model, arguments.epochs, arguments.save_flag)
    print("complete (6)")

