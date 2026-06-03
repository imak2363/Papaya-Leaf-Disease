from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import base64

app = Flask(__name__)
CORS(app)

# Set a secret key for session data
app.secret_key = os.urandom(24)

# --- Configuration ---
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
MODEL_PATH = 'C:/Users/DELL/Documents/API/New folder/trained_densenet201.h5'
CLASS_NAMES = [
    'Anthracnose', 'Bacterial Spot', 'Healthy Leaf', 'Leaf Curl',
    'Mealybug', 'Mite Disease', 'Mosaic', 'Ring Spot'
]

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# --- Load the Model ---
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- Check allowed file types ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Preprocess the image ---
def preprocess_image(image_file):
    try:
        image = Image.open(io.BytesIO(image_file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return image_array, img_str
    except Exception as e:
        print(f"❌ Error in image processing: {e}")
        raise ValueError("Image processing error")

# --- Get disease info (Example) ---
def get_disease_info(disease_name):
    disease_info = {
        'Anthracnose': {'description': 'Fungal disease affecting plant tissues.', 'treatment': ['Apply fungicides', 'Remove infected leaves']},
        'Bacterial Spot': {'description': 'Spots on leaves caused by bacteria.', 'treatment': ['Use bactericides', 'Ensure good air circulation']},
        'Healthy Leaf': {'description': 'No disease detected. Keep it up!', 'treatment': ['No treatment needed']},
        'Leaf Curl': {'description': 'Leaves curl due to viruses or pests.', 'treatment': ['Use antiviral pesticides', 'Prune affected branches']}
    }
    return disease_info.get(disease_name, {'description': 'Unknown disease.', 'treatment': ['Consult an expert.']})

# --- Upload page ---
@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Plant Disease Predictor</title>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                body {
                    font-family: 'Roboto', sans-serif;
                    background-color: #2C2C2C;
                    color: #fff;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    background-color: #333;
                    border-radius: 12px;
                    padding: 40px;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    max-width: 500px;
                    width: 100%;
                }
                h1 {
                    font-size: 28px;
                    margin-bottom: 20px;
                    color: #4CAF50;
                    text-transform: uppercase;
                    letter-spacing: 1.5px;
                }
                label {
                    background-color: #4CAF50;
                    color: white;
                    padding: 15px 30px;
                    font-size: 16px;
                    border-radius: 10px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }
                label:hover {
                    background-color: #45a049;
                }
                input[type="file"] {
                    display: none;
                }
                button {
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px 24px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    margin-top: 20px;
                    width: 100%;
                    font-size: 16px;
                }
                button:hover {
                    background-color: #45a049;
                }
                .result-card {
                    background-color: #444;
                    border-radius: 12px;
                    margin-top: 30px;
                    padding: 30px;
                }
                .result-card img {
                    max-width: 100%;
                    border-radius: 10px;
                    margin-top: 20px;
                }
                .result-card h3 {
                    color: #4CAF50;
                    margin-top: 20px;
                }
                .result-card ul {
                    text-align: left;
                    margin-top: 10px;
                    color: #ccc;
                    list-style-type: square;
                }
                a {
                    color: #4CAF50;
                    text-decoration: none;
                    font-weight: bold;
                    margin-top: 20px;
                    display: inline-block;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Plant Disease Predictor</h1>
                <form action="/predict" method="POST" enctype="multipart/form-data">
                    <label for="image">Upload Plant Leaf Image</label>
                    <input type="file" name="image" id="image" accept="image/*" required>
                    <button type="submit">Predict Disease</button>
                </form>
            </div>
        </body>
        </html>
    """)

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded!'}), 400
        
        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format! Only jpg, jpeg, png, gif, bmp are allowed.'}), 400

        # Process the image
        processed_image, img_str = preprocess_image(file)
        predictions = model.predict(processed_image)
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_disease = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index]) * 100

        # Get disease info and treatment recommendations
        disease_info = get_disease_info(predicted_disease)

        # Generate results
        result_html = f"""
        <div class="container">
            <h1>Prediction Result</h1>
            <div class="result-card">
                <h3>Disease: {predicted_disease} (Confidence: {confidence:.2f}%)</h3>
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image">
                <h3>Description:</h3>
                <p>{disease_info['description']}</p>
                <h3>Treatment Recommendations:</h3>
                <ul>
                    {"".join([f"<li>{treatment}</li>" for treatment in disease_info['treatment']])}
                </ul>
            </div>
            <a href="/" >Go Back</a>
        </div>
        """
        return result_html
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

# --- Flask App Running ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
