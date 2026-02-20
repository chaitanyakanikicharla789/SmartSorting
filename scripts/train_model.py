import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import os
import json

# ==============================
# SETTINGS
# ==============================
train_path = "dataset/train"
val_path = "dataset/test"   # using test as validation

img_size = (224, 224)
batch_size = 32
EPOCHS = 10

# ==============================
# CREATE MODEL FOLDER
# ==============================
os.makedirs("model", exist_ok=True)

# ==============================
# DATA PREPROCESSING
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

print("Number of classes:", train_data.num_classes)
print("Class indices:", train_data.class_indices)

# ==============================
# SAVE CLASS LABELS
# ==============================
class_indices = train_data.class_indices
class_labels = {str(v): k for k, v in class_indices.items()}

with open("model/class_labels.json", "w") as f:
    json.dump(class_labels, f, indent=4)

print("✅ class_labels.json saved!")

# ==============================
# VGG16 TRANSFER LEARNING MODEL
# ==============================
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# TRAIN MODEL
# ==============================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==============================
# SAVE MODEL
# ==============================
model.save("model/fruit_model.h5")

print("✅ Model + labels saved successfully!")
