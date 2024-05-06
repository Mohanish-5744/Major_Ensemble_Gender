from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import cv2

app = Flask(__name__, template_folder='templates')

# Load gender prediction models
gender_model_path_resnet = 'gender_model_resnet.h5'
gender_model_path_inception = 'gender_model_inception.h5'
gender_model_path_cnn = 'gender_model_cnn.h5'

gender_model_resnet = load_model(gender_model_path_resnet)
gender_model_inception = load_model(gender_model_path_inception)
gender_model_cnn = load_model(gender_model_path_cnn)

def process_and_predict_resnet50(pil_img):
    pil_img_resized = pil_img.resize((200, 200), Image.LANCZOS)
    pil_img_rgb = pil_img_resized.convert('RGB')
    img_array = np.array(pil_img_rgb)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_normalized = img_array_expanded.astype('float32') / 255.0
    gender_prob = gender_model_resnet.predict(img_array_normalized)
    gender = "male" if gender_prob[0][0] < 0.5 else "female"
    return gender

def process_and_predict_inceptionv3(pil_img):
    pil_img_resized = pil_img.resize((200, 200), Image.LANCZOS)
    pil_img_rgb = pil_img_resized.convert('RGB')
    img_array = np.array(pil_img_rgb)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_normalized = img_array_expanded.astype('float32') / 255.0
    gender_prob = gender_model_inception.predict(img_array_normalized)
    gender = "female" if gender_prob[0][0] < 0.5 else "male"
    return gender

def process_and_predict_cnn(img_gray_face):
    img_gray_face_resized = cv2.resize(img_gray_face, (100, 100))
    img_gray_face_reshaped = img_gray_face_resized.reshape(-1, 100, 100, 1)
    gender_prob = gender_model_cnn.predict(img_gray_face_reshaped)
    gender = "male" if gender_prob[0][0] < 0.5 else "female"
    return gender

def predict_gender_ensemble(pil_img):
    gender_resnet50 = process_and_predict_resnet50(pil_img)
    gender_inceptionv3 = process_and_predict_inceptionv3(pil_img)
    gender_cnn = process_and_predict_cnn(np.array(pil_img))
    all_predictions = [gender_resnet50, gender_inceptionv3, gender_cnn]
    return gender_resnet50, gender_inceptionv3, gender_cnn, max(set(all_predictions), key=all_predictions.count)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        # Handle image upload
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read image and perform gender prediction
        img = Image.open(file.stream).convert('RGB')
        gender_resnet50, gender_inceptionv3, gender_cnn, gender_ensemble = predict_gender_ensemble(img)

        return jsonify({
            "gender_resnet50": gender_resnet50,
            "gender_inceptionv3": gender_inceptionv3,
            "gender_cnn": gender_cnn,
            "gender_ensemble": gender_ensemble
        })
    else:
        return jsonify({"error": "Method Not Allowed"}), 405

if __name__ == '__main__':
    app.run(debug=True)
