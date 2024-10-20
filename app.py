import numpy as np
import cv2 as cv
from ultralytics import YOLO
from tensorflow.keras.preprocessing import image
import keras
import time
import json
from datetime import datetime
from flask import Flask, jsonify
import threading

# Load YOLO and classifier models
yolo_model = YOLO('datasets/weights/best.pt')
classifier_model = keras.models.load_model('classifier_model3.keras')

# Labels for classification
labels = [
    "fresh_apple", "fresh_bitter_gourd", "fresh_capsicum", "fresh_orange", "fresh_tomato",
    "stale_apple", "stale_bitter_gourd", "stale_capsicum", "stale_orange", "stale_tomato"
]

# Flask app
app = Flask(__name__)

# Shared data
result_array = []
object_counts = {label: 0 for label in labels}
lock = threading.Lock()

# Time intervals for updating results
time_interval = 10
start_time = time.time()

# Function to process video frames in the background
def process_video_feed():
    global start_time, object_counts

    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("Error opening video stream or file")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO model on the frame
        results = yolo_model(frame)

        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()

                if confidence > 0.75:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Extract the detected object from the frame
                    detected_object = frame[y1:y2, x1:x2]

                    # Resize and prepare for classification
                    resized_object = cv.resize(detected_object, (256, 256))
                    img_array = image.img_to_array(resized_object)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0

                    # Classify the detected object
                    predicted_item = classifier_model.predict(img_array)
                    index = np.argmax(predicted_item[0], axis=0)
                    predicted_label = labels[index]

                    # Increment object count
                    with lock:
                        object_counts[predicted_label] += 1

                    # Display label on frame
                    cv.putText(frame, f"{predicted_label} ({confidence * 100:.2f}%)", (x1, y1 - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

        # Display the video feed
        cv.imshow('Video Feed', frame)

        # Check if it's time to update the results every `time_interval` seconds
        current_time = time.time()
        if current_time - start_time >= time_interval:
            with lock:
                if any(object_counts.values()):  # Ensure there are detected objects
                    most_detected_label = max(object_counts, key=object_counts.get)
                    result_json = {
                        "timestamp": datetime.now().isoformat(),
                        "detected_object": most_detected_label,
                        "count": object_counts[most_detected_label]
                    }
                    result_array.append(result_json)
                    print(result_array)
                    print(f"Detected objects in the last {time_interval} seconds: {json.dumps(result_json)}")

                # Reset object counts for the next interval
                object_counts = {label: 0 for label in labels}
                start_time = current_time

        # Stop the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    capture.release()
    cv.destroyAllWindows()

# Flask endpoint to get the results
@app.route('/get_results', methods=['GET'])
def get_results():
    with lock:
        response = {
            "results": result_array
        }
    return jsonify(response)

# Main function to start the video processing thread and the Flask app
if __name__ == '__main__':
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=process_video_feed, daemon=True)
    video_thread.start()

    # Start the Flask app
    app.run(debug=True, port=5000)
