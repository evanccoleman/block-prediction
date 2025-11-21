import argparse
import h5py
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

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
        '-f', '--file',
        metavar='FILE',
        type=str,
        dest='predict_file',
        default='png_dataset128',  # './artificial.h5',
        help='path to the file with the prediction data'
    )
    parser.add_argument(
        '-m', '--model',
        metavar='MODEL',
        type=str,
        dest='model',
        default='./latest_model_PngClass.h5',
        help='path to model'
    )

    parser.add_argument(
        '-T', '--target',
        nargs='?',
        type=positive_int,
        default=1000,
        help='Target size'
    )
    return parser.parse_args()

def load_data(path, target_size):
    data = []
    labels = []
    class_path = os.path.join(path, 'classes.txt')
    with open(class_path, 'r') as f:
        class_list = [int(line.strip()) for line in f]
        print(f"Class list: {class_list}")
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        print(f"Current entry: {entry}")
        if os.path.isdir(entry_path) and entry.startswith('label_'):
            print(f"Label entry: {entry_path}")
            label = int(entry.split('_')[1])
            for matrix in os.listdir(os.path.join(path, entry)):
                if matrix.endswith(".png"):
                    print(f"Matrix: {matrix}")
                    img = image.load_img(os.path.join(path, entry, matrix), color_mode='grayscale', target_size=target_size)
                    img_array = image.img_to_array(img) / 255
                    data.append(img_array)
                    labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels, class_list

def load_model(model_file):
    return tf.keras.models.load_model(model_file)

def predict(data,model):
    return model.predict(data, batch_size=1, verbose=True)

if __name__ == '__main__':
    args = parse_cli()
    # load data
    data, labels, block_classes = load_data(args.train, (args.target, args.target)) #can specify which size PNG
    model = load_model(args.model_file)
    matrix_size = data.shape[1]
    labels_idx = {j:i for i, j in enumerate(block_classes)}
    print(labels_idx)
    labels = np.array([labels_idx[val] for val in labels])
    print(labels)
    X_test, y_test = data, labels.T

    # one-hot encode labels
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    
    y_pred_probs = predict(X_test, model) 
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(report)
