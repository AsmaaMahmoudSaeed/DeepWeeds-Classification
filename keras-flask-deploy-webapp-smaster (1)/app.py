import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = r'C:\Users\ASMAA\ITI\Graduation Project\DeepWeeds-master\models\resnet.hdf5'

# Load your own trained model


model = keras.models.load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model=model):
    img = img.resize((224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255.)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='tf')
    print(x.shape)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("image.png")
        image_width, image_height = 224, 224
        img = image.load_img("image.png", target_size=(image_width, image_height))
        # Make prediction
        classes = {'Chinee Apple':0,'Lantana':1,'Parkinsonia':2,'Parthenium':3,'Prickly Acacia':4,'Rubber Vine':5,'Siam Weed':6,'Snake Weed':7,'Negatives':8}
        new_dict = {value:key for (key,value) in classes.items()}
        preds = new_dict[np.argmax(model_predict(img))]

        # Process your result for human
        pred_proba = 100.   # Max probability
        result = str(preds)              # Convert to string
        # result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
