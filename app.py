# import os
# from flask import Flask, render_template,request,redirect,send_from_directory,url_for
# import numpy as np
# import json
# import uuid
# import tensorflow as tf

# app = Flask(__name__)
# model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
# label = ['Apple___Apple_scab',
#  'Apple___Black_rot',
#  'Apple___Cedar_apple_rust',
#  'Apple___healthy',
#  'Background_without_leaves',
#  'Blueberry___healthy',
#  'Cherry___Powdery_mildew',
#  'Cherry___healthy',
#  'Corn___Cercospora_leaf_spot Gray_leaf_spot',
#  'Corn___Common_rust',
#  'Corn___Northern_Leaf_Blight',
#  'Corn___healthy',
#  'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#  'Grape___healthy',
#  'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot',
#  'Peach___healthy',
#  'Pepper,_bell___Bacterial_spot',
#  'Pepper,_bell___healthy',
#  'Potato___Early_blight',
#  'Potato___Late_blight',
#  'Potato___healthy',
#  'Raspberry___healthy',
#  'Soybean___healthy',
#  'Squash___Powdery_mildew',
#  'Strawberry___Leaf_scorch',
#  'Strawberry___healthy',
#  'Tomato___Bacterial_spot',
#  'Tomato___Early_blight',
#  'Tomato___Late_blight',
#  'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot',
#  'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#  'Tomato___Tomato_mosaic_virus',
#  'Tomato___healthy']

# with open("plant_disease.json",'r') as file:
#     plant_disease = json.load(file)

# # print(plant_disease[4])

# @app.route('/uploadimages/<path:filename>')
# def uploaded_images(filename):
#     return send_from_directory('./uploadimages', filename)

# @app.route('/',methods = ['GET'])
# def home():
#     return render_template('home.html')

# def extract_features(image):
#     image = tf.keras.utils.load_img(image,target_size=(160,160))
#     feature = tf.keras.utils.img_to_array(image)
#     feature = np.array([feature])
#     return feature

# def model_predict(image):
#     img = extract_features(image)
#     prediction = model.predict(img)
#     # print(prediction)
#     prediction_label = plant_disease[prediction.argmax()]
#     return prediction_label

# @app.route('/upload/',methods = ['POST','GET'])
# def uploadimage():
#     if request.method == "POST":
#         image = request.files['img']
#         temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
#         image.save(f'{temp_name}_{image.filename}')
#         print(f'{temp_name}_{image.filename}')
#         prediction = model_predict(f'./{temp_name}_{image.filename}')
#         return render_template('home.html',result=True,imagepath = f'/{temp_name}_{image.filename}', prediction = prediction )
    
#     else:
#         return redirect('/')
        
    
# if __name__ == "__main__":
#     app.run(debug=True)
import os
import cv2
import json
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_from_directory

app = Flask(__name__)

# Load Model
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# Load disease info from JSON
with open("plant_disease.json", 'r') as file:
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
    feature = np.expand_dims(feature, axis=0) # Same as np.array([feature])
    return feature

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('uploadimages', filename)

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        img_file = request.files['img']
        if not img_file:
            return redirect('/')

        # Save the file
        filename = f"temp_{uuid.uuid4().hex}_{img_file.filename}"
        os.makedirs('uploadimages', exist_ok=True)
        file_path = os.path.join('uploadimages', filename)
        img_file.save(file_path)

        # 1. Blur Detection
        blurry, val = is_blurry(file_path)
        if blurry:
            # We use 'name', 'cause', and 'cure' to match your HTML
            blur_error = {
                "name": "Image Too Blurry",
                "cause": f"Focus Score: {val:.2f} (Below threshold of 80)",
                "cure": "Please clean your lens and take a steady, sharp photo of the leaf."
            }
            return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=blur_error)

        # 2. Model Prediction
        img_batch = extract_features(file_path)
        prediction_array = model.predict(img_batch)
        prediction = plant_disease[np.argmax(prediction_array)]

        # 3. Check for Background (if the model misses)
        if "Background" in prediction.get('name', ''):
             no_leaf_error = {
                 "name": "No Leaf Detected",
                 "cause": "The image appears to be background noise or non-plant material.",
                 "cure": "Please ensure the leaf is centered and fills most of the frame."
             }
             return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=no_leaf_error)

        return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=prediction)
    
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)