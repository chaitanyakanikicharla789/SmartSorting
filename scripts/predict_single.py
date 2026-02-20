import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# =========================
# Load trained model
# =========================
model = tf.keras.models.load_model("model/fruit_model.h5")

# =========================
# Test image path (templates folder lo undi)
# =========================
img_path = "templates/test.jpg"

# =========================
# Check image exists
# =========================
if not os.path.isfile(img_path):
    print("❌ Image file dorakaledu →", img_path)
    exit()

# =========================
# Image preprocessing
# =========================
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# =========================
# Prediction
# =========================
prediction = model.predict(img_array)

print("\n✅ Prediction probabilities:")
print(prediction)

print("\n✅ Predicted class index:")
print(np.argmax(prediction))
