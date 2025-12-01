import argparse
import json
import os
import pickle
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, mixed_precision
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras_tuner as kt

# --- OPTIMIZATION 1: GPU Memory Growth ---
# Prevents TF from hogging all VRAM and crashing on allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Memory Growth Enabled on {len(gpus)} GPUs")
    except RuntimeError as e:
        print(e)

# --- OPTIMIZATION 2: Mixed Precision ---
# Uses float16 for calculations (faster, less VRAM) on RTX cards
# try:
#     mixed_precision.set_global_policy('mixed_float16')
#     print("Mixed Precision (float16) Enabled")
# except Exception as e:
#     print("Could not enable mixed precision:", e)

FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CLASS_MAP = {k: v for v, k in enumerate(FRACTION_CLASSES)}


def parse_cli():
    parser = argparse.ArgumentParser(description="Unified Matrix CNN")

    # Inputs
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--mode', type=str, choices=['classification', 'regression'], default='regression')
    parser.add_argument('--input_type', type=str, choices=['png', 'dense'], default='png')
    parser.add_argument('--reg_label', type=str, default='solver')

    # NEW: Target Size for Resizing
    parser.add_argument('--target_size', type=int, default=128, help='Resize inputs to this dimension (saves VRAM)')

    # Outputs
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save all artifacts')
    parser.add_argument('--run_name', type=str, default='experiment', help='Prefix for saved files')

    # Hyperparams
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tune', action='store_true')

    return parser.parse_args()


class MatrixGenerator(Sequence):
    def __init__(self, metadata_list, data_dir, input_type, mode, target_size=128, batch_size=32, shuffle=True,
                 reg_label_type='solver'):
        self.metadata = metadata_list
        self.data_dir = data_dir
        self.input_type = input_type
        self.mode = mode
        self.target_size = target_size  # Store target size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reg_label_type = reg_label_type
        self.indexes = np.arange(len(self.metadata))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.metadata) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_meta = [self.metadata[k] for k in indexes]
        X, y = self.__data_generation(batch_meta)
        return X, y

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

    def __data_generation(self, batch_meta):
        X = []
        y = []
        for entry in batch_meta:
            # X Generation
            if self.input_type == 'png':
                img_path = os.path.join(self.data_dir, 'images', entry['files']['image'])
                # OPTIMIZATION 3: Load directly at target size
                img = load_img(img_path, color_mode='grayscale', target_size=(self.target_size, self.target_size))
                X.append(img_to_array(img) / 255.0)

            elif self.input_type == 'dense':
                mat_filename = entry['files']['matrix']
                if mat_filename is None:
                    raise ValueError(f"Sample {entry.get('matrix_id')} missing raw matrix file.")

                mat_path = os.path.join(self.data_dir, 'matrices', mat_filename)
                sparse_mat = sp.load_npz(mat_path)
                dense_mat = sparse_mat.todense()

                # Resize Dense Matrix if needed?
                # Ideally dense matrices are passed to CV2 or TF.image.resize
                # For now, assuming dense matrices match target or CNN handles shape adaptation
                # But adding channel dim is required
                dense_mat = np.expand_dims(dense_mat, axis=-1)

                # If dense matrix is 500x500 but we want 128x128, we must resize
                if dense_mat.shape[0] != self.target_size:
                    dense_mat = tf.image.resize(dense_mat, [self.target_size, self.target_size]).numpy()

                X.append(dense_mat)

            # Y Generation
            labels = entry['labels']
            if self.mode == 'classification':
                val = labels.get('class_optimal_time')
                idx = CLASS_MAP.get(val, 0)
                y.append(to_categorical(idx, num_classes=len(FRACTION_CLASSES)))
            elif self.mode == 'regression':
                key = 'regression_ground_truth' if self.reg_label_type == 'physics' else 'regression_interpolated_optimal'
                y.append(labels.get(key))

        return np.array(X), np.array(y)


def build_model_tuner(hp, input_shape, mode):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    num_blocks = hp.Int("num_blocks", 1, 3)
    for i in range(num_blocks):
        filters_1 = hp.Choice(f"f1_{i}", [16, 32, 64])
        filters_2 = hp.Choice(f"f2_{i}", [32, 64, 128])

        residual = x
        if x.shape[-1] != filters_2:
            residual = layers.Conv2D(filters_2, 1, padding='same')(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('selu')(x)
        x = layers.Conv2D(filters_1, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('selu')(x)
        x = layers.Conv2D(filters_2, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.add([x, residual])

    x = layers.BatchNormalization()(x)
    x = layers.ZeroPadding2D((0, 3))(x)
    x = layers.Conv2D(32, (21, 7), activation='tanh')(x)
    x = layers.Conv2D(64, (1, 3), padding='same', activation='tanh')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(hp.Choice("dense_units", [128, 256]), activation='sigmoid')(x)
    x = layers.Dropout(hp.Float("dropout", 0.1, 0.5))(x)

    if mode == 'classification':
        # float32 is required for Softmax output even in Mixed Precision
        outputs = layers.Dense(len(FRACTION_CLASSES), activation='softmax', dtype='float32')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        # float32 is required for Regression output
        outputs = layers.Dense(1, activation='linear', dtype='float32')(x)
        loss = 'mean_squared_error'
        metrics = ['mae']

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Nadam(learning_rate=1e-3), loss=loss, metrics=metrics)
    return model


def load_dataset_metadata(data_dir):
    meta_dir = os.path.join(data_dir, "metadata")
    all_metadata = []
    if not os.path.exists(meta_dir):
        print(f"ERROR: Metadata directory not found: {meta_dir}")
        return []

    files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    print(f"Scanning {len(files)} metadata files...")
    for f in files:
        try:
            with open(os.path.join(meta_dir, f), 'r') as jf:
                meta = json.load(jf)
                if 'labels' in meta: all_metadata.append(meta)
        except:
            continue
    return all_metadata


def main():
    args = parse_cli()
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = load_dataset_metadata(args.data_dir)
    if not metadata: return
    train_meta, test_meta = train_test_split(metadata, test_size=0.2, random_state=42)

    print(f"Training on {len(train_meta)} samples, Testing on {len(test_meta)} samples.")

    # Peek at shape (Passing target_size)
    temp_gen = MatrixGenerator([train_meta[0]], args.data_dir, args.input_type, args.mode,
                               target_size=args.target_size, batch_size=1)
    try:
        X_sample, y_sample = temp_gen[0]
        input_shape = X_sample.shape[1:]
        print(f"Detected Input Shape: {input_shape}")
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    train_gen = MatrixGenerator(train_meta, args.data_dir, args.input_type, args.mode,
                                target_size=args.target_size, batch_size=args.batch_size, reg_label_type=args.reg_label)
    test_gen = MatrixGenerator(test_meta, args.data_dir, args.input_type, args.mode,
                               target_size=args.target_size, batch_size=args.batch_size, shuffle=False,
                               reg_label_type=args.reg_label)

    hp = kt.HyperParameters()
    hp.Fixed("num_blocks", 2)
    model = build_model_tuner(hp, input_shape, args.mode)

    model_path = os.path.join(args.output_dir, f"{args.run_name}_model.h5")
    checkpoint = callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(train_gen, validation_data=test_gen, epochs=args.epochs, callbacks=[checkpoint, early_stop])

    hist_path = os.path.join(args.output_dir, f"{args.run_name}_history.pkl")
    with open(hist_path, 'wb') as f:
        pickle.dump(history.history, f)

    loss, metric = model.evaluate(test_gen)

    results = {
        "run_name": args.run_name,
        "mode": args.mode,
        "input_type": args.input_type,
        "test_loss": loss,
        "test_metric": metric,
        "metric_name": model.metrics_names[1],
        "epochs_trained": len(history.history['loss'])
    }
    json_path = os.path.join(args.output_dir, f"{args.run_name}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Artifacts saved to {args.output_dir}")


if __name__ == '__main__':
    main()

