# Install Flask (if not already installed)
# pip install Flask

from flask import Flask, request, jsonify
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
from json import dumps
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the trained model
model = load_model("./trainedModel/")

# Define a function to preprocess an image for prediction
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the image from the user
        image = request.files['image']
        
        # Ensure the image file is valid
        if image:
            # Preprocess the image
            img = preprocess_image(Image.open(image))

            # Make a prediction
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            class_probabilities = prediction.tolist()[0]
            if (predicted_class==0):
                class_id = "Daisy"
            elif (predicted_class==1):
                class_id = "Dandelion"
            elif (predicted_class==2):
                class_id = "Rose"
            elif (predicted_class==3):
                class_id = "Sunflower"
            else:
                class_id = "Tulip"
            # Provide the result as JSON
            response = {
                'class_id': class_id,
                'probability': np.max(class_probabilities)
            }
            return jsonify(response)
        else:
            return jsonify({'error': 'Invalid image file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
