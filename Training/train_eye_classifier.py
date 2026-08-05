"""
SleepAway — Eye State Classifier Training
==========================================
Trains a tiny CNN to classify cropped eye images as OPEN or CLOSED.

RUN THIS ON YOUR PC OR GOOGLE COLAB — NOT ON THE PI.
The Pi only runs the exported .tflite file.

Dataset: MRL Eye Dataset (84,898 infrared eye images)
Download: https://mrl.cs.vsb.cz/eyedataset.html
     or:  https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset

Expected folder layout after you extract the dataset:

    dataset/
      open/
        s0001_00001_0_0_1_0_0_01.png
        ...
      closed/
        s0001_00002_0_0_0_0_0_01.png
        ...

If your download has all images in one flat folder, run
sort_mrl_dataset() below first — MRL encodes the label in the filename.
"""

import os
import glob
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset"        # folder containing open/ and closed/
IMG_SIZE      = 64               # 64x64 input. Use 32 for even faster.
BATCH_SIZE    = 128
EPOCHS        = 25
MODEL_OUT     = "eye_state.tflite"
SEED          = 42

# ─────────────────────────────────────────────────────────────────
# OPTIONAL — sort a flat MRL dump into open/ and closed/ folders
# ─────────────────────────────────────────────────────────────────
def sort_mrl_dataset(flat_dir, out_dir="dataset"):
    """
    MRL filenames encode metadata separated by underscores:
      subjectID_imageID_gender_glasses_eyeState_reflections_lighting_sensorID

    Index 4 is eye state:  0 = closed, 1 = open
    """
    os.makedirs(f"{out_dir}/open",   exist_ok=True)
    os.makedirs(f"{out_dir}/closed", exist_ok=True)

    files = glob.glob(os.path.join(flat_dir, "**", "*.png"), recursive=True)
    files += glob.glob(os.path.join(flat_dir, "**", "*.jpg"), recursive=True)
    print(f"Found {len(files)} images to sort")

    n_open = n_closed = 0
    for f in files:
        parts = os.path.basename(f).split("_")
        if len(parts) < 5:
            continue
        eye_state = parts[4]
        if eye_state == "1":
            shutil.copy(f, f"{out_dir}/open/");   n_open   += 1
        elif eye_state == "0":
            shutil.copy(f, f"{out_dir}/closed/"); n_closed += 1

    print(f"Sorted -> open: {n_open}, closed: {n_closed}")


# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
def load_datasets():
    print("Loading dataset...")

    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",      # KEY: grayscale unifies day + IR night
        label_mode="binary",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary",
    )

    print("Class names:", train_ds.class_names)   # ['closed', 'open']

    # Cache + prefetch for speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────
# MODEL — deliberately tiny so it flies on a Pi 5 CPU
# ─────────────────────────────────────────────────────────────────
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Normalise 0-255 -> 0-1
        layers.Rescaling(1.0 / 255),

        # Augmentation — only active during training.
        # These simulate real dashboard-mount variation:
        #   rotation  = head tilt
        #   zoom      = driver sitting closer/further
        #   translate = camera mount not perfectly centred
        #   contrast  = day vs IR night brightness swing
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.25, value_range=(0.0, 1.0)),

        # Block 1
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),   # 1 = open, 0 = closed
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


# ─────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────
def train():
    train_ds, val_ds = load_datasets()
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Final evaluation
    results = model.evaluate(val_ds, return_dict=True)
    print("\n--- Final validation metrics ---")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return model, val_ds


# ─────────────────────────────────────────────────────────────────
# EXPORT TO TFLITE (quantised — ~4x smaller, ~2-3x faster on Pi)
# ─────────────────────────────────────────────────────────────────
def export_tflite(model, val_ds):
    print("\nConverting to TFLite...")

    # Representative dataset lets the converter calibrate int8 ranges
    def representative_data_gen():
        for images, _ in val_ds.take(50):
            for i in range(images.shape[0]):
                yield [tf.cast(images[i:i+1], tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()

    with open(MODEL_OUT, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"Saved {MODEL_OUT}  ({size_kb:.1f} KB)")
    print("Copy this file to your Pi alongside the inference script.")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # If your dataset is a flat folder, uncomment this once:
    # sort_mrl_dataset("mrlEyes_2018_01", "dataset")

    model, val_ds = train()
    export_tflite(model, val_ds)

    # Also save the full Keras model in case you want to retrain later
    model.save("eye_state_full.keras")
    print("Done.")
