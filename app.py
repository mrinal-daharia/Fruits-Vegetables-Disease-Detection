# =========================================
# FRUITS & VEGETABLES DISEASE DETECTION
# FLASK APPLICATION
# =========================================

from flask import (

    Flask,

    render_template,

    request,

    send_from_directory

)

import os

from predict import predictDisease

# =========================================
# FLASK APP CONFIGURATION
# =========================================

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =========================================
# CREATE UPLOADS FOLDER
# =========================================

if not os.path.exists(UPLOAD_FOLDER):

    os.makedirs(UPLOAD_FOLDER)

# =========================================
# HOME PAGE
# =========================================

@app.route('/')

def home():

    return render_template('index.html')

# =========================================
# DISPLAY UPLOADED IMAGES
# =========================================

@app.route('/uploads/<filename>')

def uploaded_file(filename):

    return send_from_directory(

        app.config['UPLOAD_FOLDER'],

        filename

    )

# =========================================
# DISEASE PREDICTION
# =========================================

@app.route('/predict', methods=['POST'])

def predict():

    # CHECK IMAGE EXISTS

    if 'image' not in request.files:

        return 'No Image Uploaded'

    file = request.files['image']

    # CHECK FILE SELECTED

    if file.filename == '':

        return 'No Selected Image'

    # SAVE IMAGE

    file_path = os.path.join(

        app.config['UPLOAD_FOLDER'],

        file.filename

    )

    file.save(file_path)

    # PREDICT DISEASE

    disease, confidence = predictDisease(file_path)

    # CONVERT CONFIDENCE TO PERCENTAGE

    confidence = round(confidence * 100, 2)

    # SHOW RESULT PAGE

    return render_template(

        'result.html',

        disease=disease,

        confidence=confidence,

        image_path=file.filename

    )

# =========================================
# RUN APPLICATION
# =========================================

if __name__ == '__main__':

    app.run(

        debug=True

    )