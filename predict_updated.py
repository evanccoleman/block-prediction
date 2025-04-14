import argparse
import tensorflow as tf
import h5py
import numpy as np


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model',
        metavar='MODEL',
        type=str,
        default='./latest_model.keras',
        dest='model_file',
        help='path to the model checkpoint file'
    )
    parser.add_argument(
        '-f', '--file',
        metavar='FILE',
        type=str,
        default='./synthetic_data.h5',
        dest='predict_file',
        help='path to the HDF5 file with the prediction data'
    )

    return parser.parse_args()


def load_data(path):
    with h5py.File(path, 'r') as handle:
        data = np.array(handle['matrix_of_hard_64'])
        labels = np.array(handle['labels_for_hard_64'])

        return data, labels


def load_model(model_file):
    return tf.keras.models.load_model(model_file)

def preprocess(data, labels):
    # simply add an additional dimension for the channels for data
    # swap axis of the label set
    return np.expand_dims(data, axis=3), np.moveaxis(labels, 0, -1)


def predict(data, model):
    predictions = model.predict(data, batch_size=1, verbose=True)
    predicted_indices = np.argmax(predictions, axis=1)
    index_to_label = {0: 2, 1: 4, 2: 8, 3: 16}
    predicted_block_sizes = np.array([index_to_label[i] for i in predicted_indices])
    return predicted_block_sizes

def store(prediction, path):
    prediction_dataset = 'predictionset'
    with h5py.File(path, 'r+') as handle:
        if prediction_dataset in handle:
            del handle[prediction_dataset]
        handle[prediction_dataset] = prediction


if __name__ == '__main__':
    arguments = parse_cli()

    data, labels = preprocess(*load_data(arguments.predict_file))
    model = load_model(arguments.model_file)
    prediction = predict(data, model)
    store(prediction, arguments.predict_file)
