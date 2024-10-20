import numpy as np
import cv2 as cv
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.preprocessing import image
import keras
import time
import json
from datetime import datetime
from flask import Flask, jsonify

yolo_model = YOLO('best.pt')
classifier_model = keras.models.load_model('classifier_new_2.keras')
# classifier_model.load_weights('resnet_model_weights.weights.h5')

labels = [
    "fresh_apple",
    "fresh_bitter_gourd",
    "fresh_capsicum",
    "fresh_orange",
    "fresh_tomato",
    "stale_apple",
    "stale_bitter_gourd",
    "stale_capsicum",
    "stale_orange",
    "stale_tomato",
]

time_interval = 10  # Time interval for detecting objects
save_interval = 20  # Time interval for saving results (in seconds)
start_time = time.time()
save_start_time = time.time()  # Initialize save start time

object_counts = {label: 0 for label in labels}
result_array = []

capture = cv.VideoCapture(1)

if not capture.isOpened():
    print("Error opening video stream or file")

while True:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()

            if confidence > 0.75:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                detected_object = frame[y1:y2, x1:x2]
                resized_object = cv.resize(detected_object, (224, 224))

                img_array = image.img_to_array(resized_object)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                predicted_item = classifier_model.predict(img_array)
                index = np.argmax(predicted_item[0], axis=0)
                print(predicted_item)
                predicted_label = labels[index]

                object_counts[predicted_label] += 1

                cv.putText(frame, f"{predicted_label} ({confidence * 100:.2f}%)", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('Video Feed', frame)

    current_time = time.time()
    if current_time - start_time >= time_interval:
        most_detected_label = max(object_counts, key=object_counts.get)
        most_detected_count = object_counts[most_detected_label]

        result_json = {
            "timestamp": datetime.now().isoformat(),
            "detected_object": most_detected_label,
        }

        result_array.append(result_json)
        print(f"Detected objects in the last {time_interval} seconds: {json.dumps(result_json)}")

        object_counts = {label: 0 for label in labels}
        start_time = current_time

    # Save results every minute
    if current_time - save_start_time >= save_interval:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results_{timestamp}.json'  # Create a filename with timestamp

        with open(filename, 'w') as json_file:
            json.dump(result_array, json_file, indent=4)

        print(f"Results saved to {filename} at {datetime.now().isoformat()}")

        result_array = []  # Reset result_array to 0 after saving
        save_start_time = current_time  # Reset save start time

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print(result_array)
capture.release()
cv.destroyAllWindows()
