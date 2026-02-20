from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ==============================
# Upload folder
# ==============================
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# Load trained model
# ==============================
MODEL_PATH = "model/fruit_model.h5"
model = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
else:
    print("❌ Model file not found")

# ==============================
# Load class names from dataset
# ==============================
DATASET_PATH = "dataset/train"

if os.path.exists(DATASET_PATH):
    class_names = sorted([
        folder for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
    ])
    print("✅ Classes loaded:", class_names)
else:
    class_names = []
    print("❌ dataset/train folder not found")

# ==============================
# Home page
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# About page
# ==============================
@app.route("/about")
def about():
    return render_template("about.html")

# ==============================
# Prediction route
# ==============================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # If model not loaded
    if model is None:
        return render_template(
            "result.html",
            quality="Model not trained",
            fruit="-",
            confidence=0,
            image_path=filepath
        )

    try:
        # ==============================
        # Image preprocessing
        # ==============================
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # ==============================
        # Prediction
        # ==============================
        prediction = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(prediction))

        # ⭐ IMPORTANT → confidence number only (no % symbol)
        confidence = round(float(np.max(prediction)) * 100, 2)

        # ==============================
        # Default values
        # ==============================
        quality = "Unknown"
        fruit_name = "Unknown"

        # ==============================
        # Convert class index → name
        # Example: F_Lemon → Fresh + Lemon
        # ==============================
        if class_names and predicted_index < len(class_names):
            raw_class = class_names[predicted_index]

            if "_" in raw_class:
                status, fruit = raw_class.split("_", 1)

                if status == "F":
                    quality = "Fresh"
                elif status == "S":
                    quality = "Spoiled"

                fruit_name = fruit
            else:
                fruit_name = raw_class

        return render_template(
            "result.html",
            quality=quality,
            fruit=fruit_name,
            confidence=confidence,   # number only
            image_path=filepath
        )

    except Exception as e:
        print("Prediction Error:", e)
        return "Error processing image"

# ==============================
# Run app
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
