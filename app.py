from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import cv2
from collections import Counter 

app = Flask(__name__, template_folder='templates')

# Load gender prediction models
gender_model_path_resnet = 'gender_model_resnet.h5'
gender_model_path_inception = 'gender_model_inception.h5'
gender_model_path_cnn = 'gender_model_cnn.h5'

gender_model_resnet = load_model(gender_model_path_resnet)
gender_model_inception = load_model(gender_model_path_inception)
gender_model_cnn = load_model(gender_model_path_cnn)
age_model1=load_model("Age_Model_1.h5")
age_model2=load_model("Age_Model_2.h5")
age_model3=load_model("Age_Model_3.h5")


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

age_ranges = ['1-2', '3-6','6-9', '10-15', '15-20', '21-27', '28-35', '35-46', '46-55', '55-65', '65-100']

def predict_age_ensemble(img_gray):
    age_image = cv2.resize(img_gray, (200, 200), interpolation=cv2.INTER_AREA)
    age_input = age_image.reshape(-1, 200, 200, 1)

    predictions = []
    for model in [age_model1, age_model2, age_model3]:
        predictions.append(np.argmax(model.predict(age_input)))

    age_counts = Counter(predictions)
    most_common_age, occurrences = age_counts.most_common(1)[0]
    if occurrences >= 2:
        ensembled_age = most_common_age
    else:
        ensembled_age = predictions[0]

    return predictions, ensembled_age

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        img = Image.open(file.stream).convert('RGB')
        gender_resnet50, gender_inceptionv3, gender_cnn, gender_ensemble = predict_gender_ensemble(img)

        img_path = 'temp_image.jpg'
        img.save(img_path)

        test_image = cv2.imread(img_path)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        age_predictions = []
        for (x, y, w, h) in faces:
            img_gray = gray[y:y + h, x:x + w]
            predictions, ensembled_age = predict_age_ensemble(img_gray)
            age_predictions.append({
                "predictions": [age_ranges[p] for p in predictions],
                "ensembled_age": age_ranges[ensembled_age]
            })

        return jsonify({
            "gender_resnet50": gender_resnet50,
            "gender_inceptionv3": gender_inceptionv3,
            "gender_cnn": gender_cnn,
            "gender_ensemble": gender_ensemble,
            "age_predictions": age_predictions
        })
    else:
        return jsonify({"error": "Method Not Allowed"}), 405

if __name__ == '__main__':
    app.run(debug=True)
