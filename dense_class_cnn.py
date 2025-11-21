import argparse
from email.policy import default
import pickle
import h5py
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import keras_tuner
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Activation, add, Conv2D, Dense, Dropout, Flatten, Input, ZeroPadding2D)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
#from tensorflow.keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import np_utils
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

#this line is to ignore GPU stuff
tf.config.set_visible_devices([], 'GPU')
FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NUM_CLASSES      = len(FRACTION_CLASSES)


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
        default='./size_128_amount_3000.h5',  # './artificial.h5',
        help='path to the HDF5 file with the training data'
    )
    parser.add_argument(
        '-m', '--model',
        metavar='MODEL',
        type=str,
        dest='model',
        default='./latest_model_DenseClass.h5',
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
        default='matrix_of_128',
        help='--'
    )
    parser.add_argument(
        '-l', '--labels',

        metavar='LABEL',
        type=str,
        dest='labels',
        default='labels_for_128',
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


def build_model(hp, input_shape):
    input_img = Input(shape=input_shape)
    x = input_img

    # bottleneck blocks
    num_blocks = hp.Int("num_blocks", min_value=1, max_value=3)

    for i in range(num_blocks):
        filters_1 = hp.Choice(f"filters1_block{i}", [16, 32, 64])
        filters_2 = hp.Choice(f"filters2_block{i}", [32, 64, 128])
        kernel_size_1 = hp.Choice(f"kernel1_block{i}", [3, 5])
        kernel_size_2 = hp.Choice(f"kernel2_block{i}", [3, 4])
        l2_reg = hp.Float(f"l2_block{i}", 0.001, 0.03, step=0.005)

        bn1 = BatchNormalization()(x)
        act1 = Activation('selu')(bn1)
        conv1 = Conv2D(filters_1, kernel_size=(kernel_size_1, kernel_size_1),
                       padding='same', kernel_regularizer=l2(l2_reg))(act1)

        bn2 = BatchNormalization()(conv1)
        act2 = Activation('selu')(bn2)
        conv2 = Conv2D(filters_2, kernel_size=(kernel_size_2, kernel_size_2),
                       padding='same', kernel_regularizer=l2(l2_reg))(act2)

        if x.shape[-1] != conv2.shape[-1]:
            x = Conv2D(conv2.shape[-1], kernel_size=(1, 1), padding='same')(x)

        x = add([x, conv2])  # now shapes will match

    # corner detection block
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding=(0, 3))(x)
    x = Conv2D(hp.Choice("corner_filters1", [16, 32]), (21, 7), activation='tanh')(x)
    x = Conv2D(hp.Choice("corner_filters2", [32, 64]), (1, 3), padding='same', activation='tanh')(x)

    # fully connected head
    x = Flatten()(x)
    x = Dense(hp.Choice("dense_units", [128, 256, 512]), activation='sigmoid')(x)
    x = Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1))(x)

    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=output)

    model.compile(
        optimizer=optimizers.Nadam(
            learning_rate=hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_network(model, data, labels, model_file, epochs, save_flag):
    #plot_model(model, to_file='{}.png'.format(model_file), show_shapes=True)

    if (save_flag):
        checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=True, save_best_only=True, save_weights_only=False, mode='auto')
        training = model.fit(data, labels, epochs=epochs, batch_size=10, validation_split=1.0/5.0, callbacks=[checkpoint])
    else:
        training = model.fit(data, labels, epochs=epochs, batch_size=10, validation_split=1.0/5.0)

    if (save_flag):
        with open('{}.history'.format(model_file), 'wb') as handle:
            pickle.dump(training.history, handle)


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test Accuracy: {test_acc}")

if __name__ == '__main__':
    args = parse_cli()
    # load data
    data, labels = preprocess(*load_data(args.train, args.data, args.labels))
    print(f"Args.train is {args.train}")
    #directory = args.train
    #file_names = [f.name for f in os.scandir(directory) if f.is_file()]
    #print(f"File names: {file_names}")
    print(f"Data Shape:{data.shape}")  # Shape of the data
    print(f"Labels Shape:{labels.shape}")  # Shape of the labels
    # label mapping
    # add code here to pull sizes from directory
    print(f"labels before indexing: {labels}")
    matrix_size = data.shape[1]

    block_classes = [int(fraction * matrix_size) for fraction in FRACTION_CLASSES]
    print(block_classes)
    labels_idx = {j:i for i, j in enumerate(block_classes)}
    print(labels_idx)
    labels = np.array([labels_idx[val] for val in labels])
    print(f"labels after indexing: {labels}")
    #split into train/test
    X_train, X_test, y_train, y_test = train_test_split(data, labels.T, test_size=0.2, random_state=42)

    # one-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    input_shape = (data.shape[1], data.shape[2], 1)
    print(f"Input Shape: {input_shape}")

    tuner = RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_logs',
        project_name='dense_cnn'
    )
    #early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    #tuner.search(X_train, y_train, epochs=8, validation_split=0.2, callbacks=[early_stop])
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(f"Best hps:{best_hps.values}")
    best_model = build_model(best_hps, input_shape)
    best_model.summary()
    # build and train model
    #model = build_model((data.shape[1], data.shape[2], 1))
    #model.summary()

    train_network(best_model, X_train, y_train, args.model, epochs=20, save_flag=True)
    evaluate_model(best_model, X_test, y_test)

    model_file = args.model
    with open('{}.history'.format(model_file), 'rb') as handle:
        history = pickle.load(handle)

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("denseClass_TrainAcc.png")

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("denseClass_trainLoss.png")
# later need to convert back to block sizes
# pred_classes = model.predict(X_test).argmax(axis=1)
# pred_block_sizes = [block_sizes[i] for i in pred_classes]
