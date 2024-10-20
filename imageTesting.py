import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing import image
import keras

# Load the classifier model
classifier_model = keras.models.load_model('Class_weights_Model.keras')

# Labels for the 18 classes
labels = [
    "freshapples",
    "freshbanana",
    "freshbittergourd",
    "freshcapsicum",
    "freshcucumber",
    "freshokra",
    "freshoranges",
    "freshpotato",
    "freshtomato",
    "rottenapples",
    "rottenbanana",
    "rottenbittergourd",
    "rottencapsicum",
    "rottencucumber",
    "rottenokra",
    "rottenoranges",
    "rottenpotato",
    "rottentomato",
]

# Path to the single image you want to classify
image_path = './Resource_2/WhatsApp Image 2024-09-30 at 07.39.30_87d4ec4f.jpg'  # Replace with your image path

# Load the image
img = cv.imread(image_path)

if img is None:
    print(f"Failed to load image: {image_path}")
else:
    # Preprocess the image: resize to (180, 180) and scale pixel values
    resized_image = cv.resize(img, (180, 180))
    img_array = image.img_to_array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]

    # Predict the class
    predicted_item = classifier_model.predict(img_array)
    index = np.argmax(predicted_item[0], axis=0)
    predicted_label = labels[index]

    # Output the predicted label
    print(f"Predicted label for the image is: {predicted_label}")
