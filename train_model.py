# =========================================
# FRUITS & VEGETABLES DISEASE DETECTION
# MODEL TRAINING
# =========================================

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    Conv2D,

    MaxPooling2D,

    Flatten,

    Dense,

    Dropout

)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (

    EarlyStopping,

    ModelCheckpoint

)

import matplotlib.pyplot as plt

import os

# =========================================
# IMAGE SETTINGS
# =========================================

IMAGE_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 20

DATASET_PATH = 'dataset'

MODEL_PATH = 'model/fruits_vegetables_disease_model.h5'

# =========================================
# CREATE MODEL DIRECTORY
# =========================================

if not os.path.exists('model'):

    os.makedirs('model')

# =========================================
# DATA AUGMENTATION
# =========================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=25,

    zoom_range=0.25,

    horizontal_flip=True,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2

)

# =========================================
# TRAINING DATASET
# =========================================

train_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMAGE_SIZE, IMAGE_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='training'

)

# =========================================
# VALIDATION DATASET
# =========================================

validation_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMAGE_SIZE, IMAGE_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='validation'

)

# =========================================
# DISPLAY CLASS LABELS
# =========================================

print('\nCLASS LABELS:\n')

print(train_generator.class_indices)

# =========================================
# CNN MODEL
# =========================================

model = Sequential()

# =========================================
# CONVOLUTION BLOCK 1
# =========================================

model.add(

    Conv2D(

        32,

        (3, 3),

        activation='relu',

        input_shape=(128, 128, 3)

    )

)

model.add(

    MaxPooling2D(

        pool_size=(2, 2)

    )

)

# =========================================
# CONVOLUTION BLOCK 2
# =========================================

model.add(

    Conv2D(

        64,

        (3, 3),

        activation='relu'

    )

)

model.add(

    MaxPooling2D(

        pool_size=(2, 2)

    )

)

# =========================================
# CONVOLUTION BLOCK 3
# =========================================

model.add(

    Conv2D(

        128,

        (3, 3),

        activation='relu'

    )

)

model.add(

    MaxPooling2D(

        pool_size=(2, 2)

    )

)

# =========================================
# FLATTEN LAYER
# =========================================

model.add(Flatten())

# =========================================
# DENSE LAYERS
# =========================================

model.add(

    Dense(

        256,

        activation='relu'

    )

)

model.add(

    Dropout(0.5)

)

# =========================================
# OUTPUT LAYER
# =========================================

model.add(

    Dense(

        train_generator.num_classes,

        activation='softmax'

    )

)

# =========================================
# COMPILE MODEL
# =========================================

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

# =========================================
# MODEL SUMMARY
# =========================================

model.summary()

# =========================================
# CALLBACKS
# =========================================

early_stopping = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor='val_accuracy',

    save_best_only=True,

    verbose=1

)

# =========================================
# TRAIN MODEL
# =========================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[

        early_stopping,

        checkpoint

    ]

)

# =========================================
# SAVE FINAL MODEL
# =========================================

model.save(MODEL_PATH)

print('\nMODEL SAVED SUCCESSFULLY')

# =========================================
# ACCURACY GRAPH
# =========================================

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title(

    'Fruits & Vegetables Disease Detection Accuracy'

)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend([

    'Train',

    'Validation'

])

plt.show()

# =========================================
# LOSS GRAPH
# =========================================

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title(

    'Fruits & Vegetables Disease Detection Loss'

)

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend([

    'Train',

    'Validation'

])

plt.show()