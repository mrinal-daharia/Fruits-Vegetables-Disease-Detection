# =========================================
# FRUITS & VEGETABLES DISEASE DETECTION
# PREDICTION SYSTEM
# =========================================

import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing import image

# =========================================
# LOAD TRAINED MODEL
# =========================================

model = tf.keras.models.load_model(

    'model/fruits_vegetables_disease_model.h5'

)

# =========================================
# CLASS LABELS
# =========================================

class_names = [

    'Apple__Healthy',
    'Apple__Rotten',

    'Banana__Healthy',
    'Banana__Rotten',

    'Bellpepper__Healthy',
    'Bellpepper__Rotten',

    'Carrot__Healthy',
    'Carrot__Rotten',

    'Cucumber__Healthy',
    'Cucumber__Rotten',

    'Grape__Healthy',
    'Grape__Rotten',

    'Guava__Healthy',
    'Guava__Rotten',

    'Jujube__Healthy',
    'Jujube__Rotten',

    'Mango__Healthy',
    'Mango__Rotten',

    'Orange__Healthy',
    'Orange__Rotten',

    'Pomegranate__Healthy',
    'Pomegranate__Rotten',

    'Potato__Healthy',
    'Potato__Rotten',

    'Strawberry__Healthy',
    'Strawberry__Rotten',

    'Tomato__Healthy',
    'Tomato__Rotten'

]

# =========================================
# PREDICTION FUNCTION
# =========================================

def predictDisease(img_path):

    # LOAD IMAGE

    img = image.load_img(

        img_path,

        target_size=(128, 128)

    )

    # CONVERT IMAGE TO ARRAY

    img_array = image.img_to_array(img)

    # EXPAND DIMENSIONS

    img_array = np.expand_dims(

        img_array,

        axis=0

    )

    # NORMALIZE IMAGE

    img_array = img_array / 255.0

    # MODEL PREDICTION

    prediction = model.predict(img_array)

    # GET PREDICTED CLASS INDEX

    predicted_class = np.argmax(prediction)

    # GET CONFIDENCE SCORE

    confidence = np.max(prediction)

    # GET CLASS LABEL

    disease_name = class_names[predicted_class]

    # FORMAT LABEL

    disease_name = disease_name.replace(

        '__',

        ' - '

    )

    # RETURN RESULT

    return disease_name, confidence