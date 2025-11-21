import argparse
from email.policy import default
import pickle
#import h5py
import os
import tensorflow as tf
import sklearn
from numpy.matrixlib.defmatrix import matrix
from sklearn.preprocessing import LabelEncoder
#import keras_tuner
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
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import np_utils
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.preprocessing import image

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
        default=10,
        help='number of training epochs'
    )
    parser.add_argument(
        '-t', '--train',
        metavar='TRAIN',
        type=str,
        dest='train',
        default='png_dataset128',  # './artificial.h5',
        help='path to the HDF5 file with the training data'
    )
    parser.add_argument(
        '-m', '--model',
        metavar='MODEL',
        type=str,
        dest='model',
        default='./latest_model_PngClass.h5',
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
        '-T', '--target',
        nargs='?',
        type=positive_int,
        default=500,
        help='Target size'
    )
    return parser.parse_args()

def load_data(path, target_size):
    data = []
    labels = []
    class_path = os.path.join(path, 'classes.txt')
    with open(class_path, 'r') as f:
        class_list = [int(line.strip()) for line in f]
        #print(f"Class list: {class_list}")
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        #print(f"Current entry: {entry}")
        if os.path.isdir(entry_path) and entry.startswith('label_'):
            #print(f"Label entry: {entry_path}")
            label = int(entry.split('_')[1])
            for matrix in os.listdir(os.path.join(path, entry)):
                if matrix.endswith(".png"):
                    #print(f"Matrix: {matrix}")
                    img = image.load_img(os.path.join(path, entry, matrix), color_mode='grayscale', target_size=target_size)
                    img_array = image.img_to_array(img) / 255
                    data.append(img_array)
                    labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels, class_list

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

        if x.shape[-1] != filters_2:
            x = Conv2D(filters_2, kernel_size=(1, 1), padding='same')(x)
        x = add([x, conv2])  # skip connection

    # corner detection block
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding=(0, 3))(x)
    x = Conv2D(hp.Choice("corner_filters1", [16, 32]), (21, 7), activation='tanh')(x)
    x = Conv2D(hp.Choice("corner_filters2", [32, 64]), (1, 3), padding='same', activation='tanh')(x)

    # fully connected head
    x = Flatten()(x)
    x = Dense(hp.Choice("dense_units", [128, 256, 512]), activation='sigmoid')(x)
    x = Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1))(x)

#### change the number before activation depending on # of classes  A
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
    #verbose false or truee
    if (save_flag):
        checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=False, save_best_only=True, save_weights_only=False, mode='auto')
        training = model.fit(data, labels, epochs=epochs, batch_size=10, validation_split=1.0/5.0,verbose=0, callbacks=[checkpoint])
    else:
        training = model.fit(data, labels, epochs=epochs, batch_size=10, validation_split=1.0/5.0, verbose=0)

    if (save_flag):
        with open('{}.history'.format(model_file), 'wb') as handle:
            pickle.dump(training.history, handle)


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Categorical Cross-entropy Loss (MSE): {test_loss}")
    print(f"Test Accuracy: {test_acc}")

if __name__ == '__main__':
    args = parse_cli()
    # load data
    data, labels, block_classes = load_data(args.train, (args.target, args.target)) #can specify which size PNG
    #print(f"Args.train is {args.train}")
    #directory = args.train
    #file_names = [f.name for f in os.scandir(directory) if f.is_file()]
    #print(f"File names: {file_names}")
    #print(f"Data Shape:{data.shape}")  # Shape of the data
    #print(f"Labels Shape:{labels.shape}")  # Shape of the labels
    #print(f"Labels are: {labels}")
    #print(f"Class List:{block_classes}")
    # label mapping
    # add code here to pull sizes from directory

    matrix_size = data.shape[1]

    labels_idx = {j:i for i, j in enumerate(block_classes)}
    #print(labels_idx)
    labels = np.array([labels_idx[val] for val in labels])
    #print(f"Labels after mapping: {labels}")
    #split into train/test
    X_train, X_test, y_train, y_test = train_test_split(data, labels.T, test_size=0.2, random_state=42)

    # one-hot encode labels
    # later, add variable that holds # of classes based on labels in folder
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    # input shape
    input_shape = (data.shape[1], data.shape[2], 1)
    #print(f"Input Shape: {input_shape}")
    # keras tuner
    tuner = RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_logs',
        project_name='png_proj'
    )

    # find best model/hps
    #early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    #tuner.search(X_train, y_train, epochs=8, validation_split=0.2, callbacks=[early_stop])
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("Chosen learning rate:", best_hps.get("learning_rate"))
    print(f"Best hps:{best_hps.values}")

    best_model = build_model(best_hps, input_shape)
    best_model.summary()

    # train
    train_network(best_model, X_train, y_train, args.model, args.epochs, save_flag=True)
    
    # evaluate
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
    plt.savefig("pngClass_trainAcc.png")
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("pngClass_trainLoss.png")


