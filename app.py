import os
import cv2
import json
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load Model
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# Load disease info from JSON with explicit UTF-8 encoding
with open('plant_disease.json', 'r', encoding='utf-8') as file:
    plant_disease = json.load(file)

# Helper: Check for blur using Laplacian Variance
def is_blurry(image_path, threshold=80.0):
    image = cv2.imread(image_path)
    if image is None: 
        return True, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

# Helper: Preprocess image for Model
def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0)
    return feature

@app.route('/')
def home():
    selected_lang = request.args.get('lang', 'en')
    return render_template('home.html', lang=selected_lang, result=False)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('uploadimages', filename)

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    selected_lang = request.args.get('lang') or request.form.get('language', 'en') 

    if request.method == "POST":
        img_file = request.files.get('img')
        existing_image = request.form.get('existing_image')
        
        # State Recovery Scenario
        if (not img_file or img_file.filename == '') and existing_image:
            filename = existing_image.split('/')[-1]
            file_path = os.path.join('uploadimages', filename)
            
            # 1. Blur Detection
            blurry, val = is_blurry(file_path)
            if blurry:
                if selected_lang == 'ur':
                    blur_error = {
                        "name": "تصویر دھندلی ہے",
                        "cause": f"فوکس سکور: {val:.2f} (حد 80 سے کم hai)",
                        "cure": "براہ کرم اپنے کیمرے کا لینس صاف کریں اور پتے کی صاف تصویر لیں۔"
                    }
                else:
                    blur_error = {
                        "name": "Image Too Blurry",
                        "cause": f"Focus Score: {val:.2f} (Below threshold of 80)",
                        "cure": "Please clean your lens and take a steady, sharp photo of the leaf."
                    }
                return render_template('home.html', result=True, imagepath=existing_image, prediction=blur_error, lang=selected_lang)

            # 2. Model Prediction
            img_batch = extract_features(file_path)
            prediction_array = model.predict(img_batch)

            raw_confidence = float(np.max(prediction_array)) 
            confidence = raw_confidence / 100.0 if raw_confidence > 1.0 else raw_confidence
            raw_prediction = plant_disease[np.argmax(prediction_array)]

            # STEP A: BACKGROUND CHECK
            if "Background" in raw_prediction.get('name', ''):
                if selected_lang == 'ur':
                    no_leaf_error = {
                        "name": raw_prediction.get("name_urdu", "پتہ نہیں ملا (No Leaf Detected)"),
                        "cause": raw_prediction.get("cause_urdu", "اپ لوڈ کی گئی تصویر میں کوئی پودا یا پتہ نظر نہیں آ رہا۔"),
                        "cure": raw_prediction.get("cure_urdu", "براہ کرم یقینی بنائیں کہ پتہ تصویر کے درمیان میں ہو اور واضح ہو۔")
                    }
                else:
                    no_leaf_error = {
                        "name": "No Leaf Detected",
                        "cause": "The image appears to be background noise or non-plant material.",
                        "cure": "Please ensure the leaf is centered and fills most of the frame."
                    }
                return render_template('home.html', result=True, imagepath=existing_image, prediction=no_leaf_error, lang=selected_lang)

            # STEP B: CONFIDENCE GUARDRAIL
            if confidence < 0.80:
                if selected_lang == 'ur':
                    low_confidence_error = {
                        "name": "کم اعتماد کی پیشن گوئی (Low Confidence)",
                        "cause": f"سسٹم اس پتے کی بیماری کے بارے میں مکمل پرامید نہیں ہے ({confidence*100:.1f}%)",
                        "cure": "براہ کرم بہتر روشنی میں پتے کی ایک اور واضح تصویر اپ لوڈ کریں۔"
                    }
                else:
                    low_confidence_error = {
                        "name": "Low Confidence Prediction",
                        "cause": f"Model confidence score is too low ({confidence*100:.1f}%)",
                        "cure": "The system is not entirely confident about this leaf condition. Please upload a sharper photo under better lighting."
                    }
                return render_template('home.html', result=True, imagepath=existing_image, prediction=low_confidence_error, lang=selected_lang)

            # STEP C: LANGUAGE MAPPING
            if selected_lang == 'ur':
                final_prediction = {
                    "name": raw_prediction.get("name_urdu", raw_prediction["name"]),
                    "cause": raw_prediction.get("cause_urdu", raw_prediction["cause"]),
                    "cure": raw_prediction.get("cure_urdu", raw_prediction["cure"])
                }
            else:
                final_prediction = raw_prediction

            return render_template('home.html', result=True, imagepath=existing_image, prediction=final_prediction, lang=selected_lang)

        # Standard Fresh Form Upload logic flow
        if not img_file or img_file.filename == '':
            return redirect(f'/?lang={selected_lang}')

        # FIXED: CRASH-PROOF FILENAME LOGIC
        # Purane lambe naam ko khatam karke extension nikalna (.jpg, .png etc)
        ext = os.path.splitext(secure_filename(img_file.filename))[1] or '.jpg'
        # Sirf ek 10-character ki short ID aur extension laga kar safe name banana
        filename = f"img_{uuid.uuid4().hex[:10]}{ext}"
        
        # Ensure directory exists safely
        os.makedirs('uploadimages', exist_ok=True)
        file_path = os.path.join('uploadimages', filename)
        img_file.save(file_path)

        # 1. Blur Detection
        blurry, val = is_blurry(file_path)
        if blurry:
            if selected_lang == 'ur':
                blur_error = {
                    "name": "تصویر دھندلی ہے",
                    "cause": f"فوکس سکور: {val:.2f} (حد 80 سے کم ہے)",
                    "cure": "براہ کرم اپنے کیمرے کا لینس صاف کریں اور پتے کی صاف تصویر لیں۔"
                }
            else:
                blur_error = {
                    "name": "Image Too Blurry",
                    "cause": f"Focus Score: {val:.2f} (Below threshold of 80)",
                    "cure": "Please clean your lens and take a steady, sharp photo of the leaf."
                }
            return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=blur_error, lang=selected_lang)

        # 2. Model Prediction
        img_batch = extract_features(file_path)
        prediction_array = model.predict(img_batch)

        raw_confidence = float(np.max(prediction_array)) 
        confidence = raw_confidence / 100.0 if raw_confidence > 1.0 else raw_confidence
        raw_prediction = plant_disease[np.argmax(prediction_array)]

        # STEP A: BACKGROUND CHECK
        if "Background" in raw_prediction.get('name', ''):
            if selected_lang == 'ur':
                no_leaf_error = {
                    "name": raw_prediction.get("name_urdu", "پتہ نہیں ملا (No Leaf Detected)"),
                    "cause": raw_prediction.get("cause_urdu", "اپ لوڈ کی گئی تصویر میں کوئی پودا یا پتہ نظر نہیں آ رہا۔"),
                    "cure": raw_prediction.get("cure_urdu", "براہ کرم یقینی بنائیں کہ پتہ تصویر کے درمیان میں ہو اور واضح ہو۔")
                }
            else:
                no_leaf_error = {
                    "name": "No Leaf Detected",
                    "cause": "The image appears to be background noise or non-plant material.",
                    "cure": "Please ensure the leaf is centered and fills most of the frame."
                }
            return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=no_leaf_error, lang=selected_lang)

        # STEP B: CONFIDENCE GUARDRAIL
        if confidence < 0.80:
            if selected_lang == 'ur':
                low_confidence_error = {
                    "name": "کم اعتماد کی پیشن گوئی (Low Confidence)",
                    "cause": f"سسٹم اس پتے ki بیماری کے بارے میں مکمل پرامید نہیں ہے ({confidence*100:.1f}%)",
                    "cure": "براہ کرم بہتر روشنی میں پتے کی ایک اور واضح تصویر اپ لوڈ کریں۔"
                }
            else:
                low_confidence_error = {
                    "name": "Low Confidence Prediction",
                    "cause": f"Model confidence score is too low ({confidence*100:.1f}%)",
                    "cure": "The system is not entirely confident about this leaf condition. Please upload a sharper photo under better lighting."
                }
            return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=low_confidence_error, lang=selected_lang)

        # STEP C: LANGUAGE MAPPING
        if selected_lang == 'ur':
            final_prediction = {
                "name": raw_prediction.get("name_urdu", raw_prediction["name"]),
                "cause": raw_prediction.get("cause_urdu", raw_prediction["cause"]),
                "cure": raw_prediction.get("cure_urdu", raw_prediction["cure"])
            }
        else:
            final_prediction = raw_prediction

        return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=final_prediction, lang=selected_lang)
    
    return redirect(f'/?lang={selected_lang}')

if __name__ == "__main__":
    app.run(debug=True)