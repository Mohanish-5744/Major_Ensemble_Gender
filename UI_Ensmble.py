import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from collections import Counter
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import IPython.display as display
import ipywidgets as widgets

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

path = "crop_part1"
pixels = []
gender = []

for img in os.listdir(path):
    genders = img.split("_")[1]
    img = cv2.imread(str(path)+"/"+str(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    pixels.append(np.array(img))
    gender.append(np.array(genders))

pixels = np.array(pixels)
gender = np.array(gender, np.uint64)

x_train, x_test, y_train, y_test = train_test_split(pixels, gender, random_state=100)

resnet50_predictions = []
inceptionv3_predictions = []
cnn_predictions = []
true_labels = []

for img_gray, label in zip(x_test, y_test):
    pil_img = Image.fromarray(img_gray)

all_predictions = [resnet50_predictions, inceptionv3_predictions, cnn_predictions]
ensemble_predictions = []

for sample_predictions in zip(*all_predictions):
    majority_vote = Counter(sample_predictions).most_common(1)[0][0]
    ensemble_predictions.append(majority_vote)

label_encoder = LabelEncoder()

y_val_encoded = label_encoder.fit_transform(true_labels)

ensemble_predictions_encoded = label_encoder.transform(ensemble_predictions)

def predict_gender_ensemble(pil_img):
    gender_resnet50 = process_and_predict_resnet50(pil_img)
    gender_inceptionv3 = process_and_predict_inceptionv3(pil_img)
    gender_cnn = process_and_predict_cnn(np.array(pil_img))

    all_predictions = [gender_resnet50, gender_inceptionv3, gender_cnn]

    vote_counts = {gender: all_predictions.count(gender) for gender in gender_labels}

    max_count = max(vote_counts.values())
    if list(vote_counts.values()).count(max_count) > 1:
        probabilities = {
            'male': sum(gender_model_resnet.predict_proba(pil_img)[0]),
            'female': sum(1 - gender_model_resnet.predict_proba(pil_img)[0])
        }
        majority_vote = max(probabilities, key=probabilities.get)
    else:
        majority_vote = max(vote_counts, key=vote_counts.get)

    return majority_vote

def predict_gender_ensemble(pil_img):
    gender_resnet50 = process_and_predict_resnet50(pil_img)
    gender_inceptionv3 = process_and_predict_inceptionv3(pil_img)
    gender_cnn = process_and_predict_cnn(img_gray)
    all_predictions = [gender_resnet50, gender_inceptionv3, gender_cnn]
    return gender_resnet50, gender_inceptionv3, gender_cnn, max(set(all_predictions), key=all_predictions.count)

def predict_gender_from_file(image_file):
    filename = image_file.name
    img = cv2.imdecode(np.frombuffer(image_file.content, np.uint8), -1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    gender_resnet50, gender_inceptionv3, gender_cnn, gender_ensemble = predict_gender_ensemble(pil_img)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Uploaded Image: ' + filename)
    plt.axis('off')
    plt.show()
    print("Gender Predictions by Individual Models:")
    print("ResNet50 Model Prediction:", gender_resnet50)
    print("InceptionV3 Model Prediction:", gender_inceptionv3)
    print("CNN Model Prediction:", gender_cnn)
    print("\nGender Prediction by Ensemble Model:")
    print("Ensemble Model Prediction:", gender_ensemble)

def on_upload_button_clicked(change):
    # Check the type of file_upload.value
    if isinstance(file_upload.value, dict):
        # If it's a dictionary, handle the case for multiple uploaded files
        for uploaded_filename, uploaded_file in file_upload.value.items():
            predict_gender_from_file(uploaded_file)
    else:
        # If it's not a dictionary, assume a single uploaded file
        predict_gender_from_file(next(iter(file_upload.value)))

file_upload = widgets.FileUpload(accept='image/*')
upload_button = widgets.Button(description='Predict Image')
upload_button.on_click(on_upload_button_clicked)
display.display(file_upload)
display.display(upload_button)

