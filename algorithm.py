import re
import os
import numpy as np
import pandas as pd

from tensorflow.keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
from PIL import Image



            

def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
    return df


def class_id_to_label(id):
    label_map = {1: 'glass', 2: 'paper', 3: 'cardboard', 4: 'plastic', 5: 'metal', 6: 'trash'}
    return label_map[id]


IMAGES_DIR = 'C:\\Users\\Vipul\\Downloads\\81794_189983_bundle_archive\\Garbage classification\\Garbage Classification'
    
train_csv = 'C:\\Users\\Vipul\\Downloads\\81794_189983_bundle_archive\\one-indexed-files-notrash_train.txt'
val_csv  = 'C:\\Users\\Vipul\\Downloads\\81794_189983_bundle_archive\\one-indexed-files-notrash_val.txt'
test_csv  = 'C:\\Users\\Vipul\\Downloads\\81794_189983_bundle_archive\\one-indexed-files-notrash_test.txt'

df_train = pd.read_csv(train_csv, sep=' ', header=None, names=['rel_path', 'label'])
df_valid = pd.read_csv(val_csv,   sep=' ', header=None, names=['rel_path', 'label'])
df_test  = pd.read_csv(test_csv,   sep=' ', header=None, names=['rel_path', 'label'])




df_train = add_class_name_prefix(df_train, 'rel_path')
df_valid = add_class_name_prefix(df_valid, 'rel_path')
df_test  = add_class_name_prefix(df_test,  'rel_path')


df_train['label'] = df_train['label'].apply(class_id_to_label)
df_valid['label'] = df_valid['label'].apply(class_id_to_label)
df_test['label']  = df_test['label'].apply(class_id_to_label)

print(f'Found {len(df_train)} training, {len(df_valid)} validation and {len(df_test)} samples.')


datagen = ImageDataGenerator()

datagen_train = datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=IMAGES_DIR,
    x_col='rel_path',
    y_col='label',
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=7,
)
datagen_valid = datagen.flow_from_dataframe(
    dataframe=df_valid,
    directory=IMAGES_DIR,
    x_col='rel_path',
    y_col='label',
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=7,
)
def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False)

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
        
    return model

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

net = build_model(num_classes = 6)
net.compile(optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.categorical_accuracy])
history = net.fit_generator(
    generator=datagen_train,
    validation_data=datagen_valid,
    epochs=30,
    validation_freq = 1,
    callbacks=[early_stop]
)

model_json = net.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
net.save_weights("model.h5")
print("Saved model to disk")
    
    
