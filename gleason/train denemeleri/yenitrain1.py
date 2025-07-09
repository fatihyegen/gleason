
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

import sys
sys.path.append('/content/drive/MyDrive/')
from unet import build_unet
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    # Eğitim verileri
    train_img_dir = os.path.join(path, "train", "images")
    train_mask_dir = os.path.join(path, "train", "masks")

    train_x = sorted(glob(os.path.join(train_img_dir, "*")))
    train_y = sorted(glob(os.path.join(train_mask_dir, "*")))

    # Test verileri
    test_img_dir = os.path.join(path, "test", "images")
    test_mask_dir = os.path.join(path, "test", "masks")

    test_x = sorted(glob(os.path.join(test_img_dir, "*")))
    test_y = sorted(glob(os.path.join(test_mask_dir, "*")))

    # İsteğe bağlı: valid set için train'den ayır (örnek: %20'si valid)
    from sklearn.model_selection import train_test_split
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
def get_colormap():
    # Sınıf isimleri
    classes = [
        "Benign",
        "Gleason_3",
        "Gleason_4",
        "Gleason_5"
    ]

    # Her sınıfa karşılık gelen RGB renk (isteğe göre değiştirilebilir)
    colormap = [
        [0, 0, 0],        # Benign - Siyah
        [0, 255, 0],      # Gleason_3 - Yeşil
        [255, 165, 0],    # Gleason_4 - Turuncu
        [255, 0, 0]       # Gleason_5 - Kırmızı
    ]

    return classes, colormap
def read_image_mask(x, y):
    """ Reading """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    y = cv2.imread(y, cv2.IMREAD_COLOR)
    assert x.shape == y.shape

    """ Resizing """
    x = cv2.resize(x, (IMG_W, IMG_H))
    y = cv2.resize(y, (IMG_W, IMG_H))

    """ Image processing """
    x = x / 255.0
    x = x.astype(np.float32)

    """ Mask processing """
    output = []
    for color in COLORMAP:
        cmap = np.all(np.equal(y, color), axis=-1)
        output.append(cmap)
    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)

    return x, output
def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        image, mask = read_image_mask(x, y)
        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
    image.set_shape([IMG_H, IMG_W, 3])
    mask.set_shape([IMG_H, IMG_W, NUM_CLASSES])

    return image, mask
def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    IMG_H = 320
    IMG_W = 416
    NUM_CLASSES = 4
    input_shape = (IMG_H, IMG_W, 3)

    batch_size = 16
    lr = 1e-4
    num_epochs = 10
    dataset_path = "/content/drive/MyDrive/dataset"

    model_path = os.path.join("files", "yenimodel.h5")
    csv_path = os.path.join("files", "data.csv")

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Process the colormap """
    CLASSES, COLORMAP = get_colormap()

    """ Dataset Pipeline """
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet(input_shape, NUM_CLASSES)
    # model.load_weights(model_path)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )
    # model.summary()

    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=num_epochs,
        callbacks=callbacks
    )