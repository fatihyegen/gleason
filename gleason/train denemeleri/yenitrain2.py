from google.colab import drive
drive.mount("/content/drive")
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
sys.path.append("/content/drive/MyDrive")
from unet import build_unet
from sklearn.utils import shuffle
IMG_H = 512
IMG_W = 512
NUM_CLASSES = 4
num_epochs = 10



global CLASSES
global COLORMAP
""" Klasör oluşturma fonksiyonu """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def load_dataset(train_img_dir, train_mask_dir, split=0.2):
    train_x = sorted(glob(os.path.join(train_img_dir, "*")))
    train_y = sorted(glob(os.path.join(train_mask_dir, "*")))

    assert len(train_x) == len(train_y), "Train görüntü ve maskeleri eşleşmiyor!"

    # Split into train/valid
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=split, random_state=42
    )

    train_x, train_y = shuffle(train_x, train_y, random_state=42)
    train_x = train_x[:1000]
    train_y = train_y[:1000]

    valid_x = valid_x[:250]
    valid_y = valid_y[:250]

    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)}")
    print("")

    return (train_x, train_y), (valid_x, valid_y)
def get_colormap(dataset_path=None):
    """
    Sabit sınıfları (CLASSES) ve bu sınıflara karşılık gelen renk haritasını (COLORMAP) döndürür.
    dataset_path parametresi kullanılmıyor ama arayüz uyumluluğu için bırakıldı.
    """

    COLORMAP = [
        [0, 255, 0],      # 0 - Benign (green)
        [0, 0, 255],      # 1 - Gleason_3 (blue)
        [255, 255, 0],    # 2 - Gleason_4 (yellow)
        [255, 0, 0],      # 3 - Gleason_5 (red)
    ]

    CLASSES = [
        "Benign",
        "Gleason_3",
        "Gleason_4",
        "Gleason_5",
    ]

    return CLASSES, COLORMAP
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
def tf_dataset(x, y, batch=4):
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

    """ Directory for save files """
    create_dir("files")

    """ Hyperparameters """
    IMG_H = 512
    IMG_W = 512
    NUM_CLASSES = 4
    input_shape = (IMG_H, IMG_W, 3)

    batch_size = 8
    lr = 1e-4
    num_epochs = 10
model_path = "files/modeldeneme2305.h5"
csv_path ="/content/drive/MyDrive/files/log.csv"
image_dir = "/content/drive/MyDrive/prostate_data/original_images/original_images"
mask_dir = "/content/drive/MyDrive/prostate_data/mask_images"

image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))  # <-- JPG uzantısı!
mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

matched_image_paths = []
matched_mask_paths = []

for mask_path in mask_paths:
    mask_filename = os.path.basename(mask_path)
    image_filename = mask_filename.replace("mask_", "").replace(".png", ".jpg")
    image_path = os.path.join(image_dir, image_filename)

    if os.path.exists(image_path):
        matched_image_paths.append(image_path)
        matched_mask_paths.append(mask_path)
    else:
        print(f"Uyarı: {image_filename} bulunamadı!")

print(f"Eşleşen görüntü sayısı: {len(matched_image_paths)}")
(train_x, train_y), (valid_x, valid_y) = load_dataset(image_dir, mask_dir)
print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)}")
CLASSES, COLORMAP = get_colormap()
batch_size=8
train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
input_shape = (IMG_H, IMG_W, 3)

model = build_unet(input_shape, NUM_CLASSES)
    # model.load_weights(model_path)
lr = 1e-4  # veya istediğin başka bir öğrenme oranı
model.compile(
loss="categorical_crossentropy",
optimizer=tf.keras.optimizers.Adam(lr)
    )
    # model.summary()


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