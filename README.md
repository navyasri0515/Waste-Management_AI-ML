# Waste Classification using VGG16 and Flask Deployment

## 📌 Project Overview

This project is focused on building a **deep learning model for classifying waste** into three categories: **biodegradable**, **recyclable**, and **trash**. The core model uses **VGG16 with transfer learning**, and the final product is a web application built using **Flask** that allows users to upload an image and get a prediction.

---

## 🎯 Objectives

- Understand and apply deep learning concepts like CNNs and transfer learning.
- Build a robust image classification model using **VGG16**.
- Deploy the model with a simple web UI using **Flask**.
- Demonstrate practical skills in model training, evaluation, and integration.

---

## 📁 Project Structure

waste-classification/
│
├── dataset/ # Contains image data for 3 classes
│
├── templates/ # Contains HTML files for Flask app
│ └── index.html
│
├── static/ # For CSS, JS or uploaded images
│ └── style.css
│
├── model/ # Contains the saved model
│ └── vgg16_waste.h5
│
├── app.py # Flask backend logic
├── train_model.py # Model training and saving script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore

---

## 🧠 Prior Knowledge and Concepts Used

- **Neural Networks** ([CNN vs RNN vs MLP](https://www.analyticsvidhya.com/blog/2020/02/cnn-vs-rnn-vs-mlp-analyzing-3-types-of-neural-networks-in-deep-learning/))
- **Deep Learning Frameworks** ([PyTorch vs TensorFlow](https://www.knowledgehut.com/blog/data-science/pytorch-vs-tensorflow))
- **Transfer Learning** ([VGG16 with Transfer Learning](https://towardsdatascience.com/a-demonstration-of-transfer-learning-of-vgg-convolutional-neural-network-pre-trained-model-with-c9f5b8b1ab0a))
- **VGG16** ([GeeksforGeeks on VGG16](https://www.geeksforgeeks.org/vgg-16-cnn-model/))
- **CNNs** ([CNN Tutorial](https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/))
- **Overfitting & Regularization** ([Techniques](https://www.analyticsvidhya.com/blog/2021/07/prevent-overfitting-using-regularization-techniques/))
- **Optimizers** ([Overview](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/))
- **Flask Basics** ([Flask Tutorial](https://www.youtube.com/watch?v=lj4I_CvBnt0))

---

## 🗂️ Dataset

The dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets) and contains 3 classes:

- `biodegradable/`
- `recyclable/`
- `trash/`

Each folder contains labeled images. The data is split into training and testing sets.

---

## 🧹 Data Preprocessing

Steps:
- Image resizing to (224x224)
- Normalization using ImageDataGenerator
- Categorical class mode
- No need for augmentation (already varied dataset)

---

## 🔍 Data Visualization

Used Python's `os` and `random` to select random images from folders and display them using `IPython.display`. Confirmed that the images are correctly labeled and interpreted.

Example:

```python
import os, random
from IPython.display import Image, display

folder_path = "dataset/biodegradable"
image_file = random.choice(os.listdir(folder_path))
display(Image(filename=os.path.join(folder_path, image_file)))
```

🏗️ Model Building: VGG16 with Transfer Learning
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
```
# Load pre-trained VGG16
```python
vgg = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
vgg.trainable = False
```
# Define the model
```python
model = Sequential([
    vgg,
    Flatten(),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])
```
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
---

📊 Training and Evaluation
```python
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```
# Train
```python
history = model.fit(train_data, validation_data=val_data, epochs=10)
```
# Save model
```python
model.save("model/vgg16_waste.h5")
```
# 🌐 Flask Web Application
```python
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model/vgg16_waste.h5")
class_names = ['biodegradable', 'recyclable', 'trash']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            img_path = os.path.join("static", img_file.filename)
            img_file.save(img_path)

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            label = class_names[np.argmax(prediction)]

            return render_template("index.html", label=label, image=img_path)

    return render_template("index.html", label=None)

if __name__ == "__main__":
    app.run(debug=True)
```
# templates/index.html
```
<!DOCTYPE html>
<html>
<head>
    <title>Waste Classifier</title>
</head>
<body>
    <h1>Upload an Image to Classify</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <input type="submit" value="Predict">
    </form>

    {% if label %}
        <h2>Prediction: {{ label }}</h2>
        <img src="{{ image }}" width="300">
    {% endif %}
</body>
</html>
```
# ✅ Final Output
The app allows users to upload any waste image and provides a prediction like:

Prediction: recyclable

With the image shown on the screen.

🧪 Results & Evaluation
Accuracy Achieved: ~90%

Overfitting was mitigated with dropout

Model was trained over 10 epochs

Data was balanced across classes

# 💾 Requirements

tensorflow
flask
numpy
pillow
Install with:



pip install -r requirements.txt
🚀 How to Run the Project
Clone the repository:


git clone https://github.com/navyasri0515/waste-classification-flask.git
cd waste-classification-flask
Place the dataset in the dataset/ folder.

Train the model:

python train_model.py
Run the Flask app:

python app.py
Open browser and go to http://127.0.0.1:5000/.

📌 Conclusion
This project demonstrates my ability to build an end-to-end deep learning pipeline — from data preprocessing and model training to deploying a working web application using Flask. Transfer learning with VGG16 enabled efficient training, and the deployed model provides real-time predictions with high accuracy.

📧 Contact
Name: =NAVYA SRI

Email: navyasrichillapalli@gmail.com

GitHub: github.com/navyasri0515



---



