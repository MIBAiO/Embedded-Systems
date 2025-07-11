#!/usr/bin/env python3
# Realtime hand gesture prediction with LED feedback via Sense HAT

import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
from sense_hat import SenseHat
from time import sleep

# --- Constants & Thresholds ---
MODEL_FILE = "cnn_gesture_model.tflite"
PREDICT_THRESHOLD = 0.65  # Adjust based on model confidence

FRAME_RES = (640, 480)
PREVIEW_RES = (320, 240)

# --- Load the Model ---
tflite_model = tflite.Interpreter(model_path=MODEL_FILE)
tflite_model.allocate_tensors()

input_info = tflite_model.get_input_details()[0]
output_info = tflite_model.get_output_details()[0]
INPUT_WIDTH, INPUT_HEIGHT = input_info["shape"][2], input_info["shape"][1]
INPUT_INDEX = input_info["index"]
OUTPUT_INDEX = output_info["index"]

# --- Init Hardware ---
hat = SenseHat()
hat.clear()

cam = Picamera2()
cam_config = cam.create_still_configuration(
    main={"size": FRAME_RES, "format": "RGB888"},
    lores={"size": PREVIEW_RES, "format": "RGB888"},
    display="lores"
)
cam.configure(cam_config)
cam.start()

# --- Create Display Window ---
WINDOW_NAME = "Gesture Viewer"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

print("Press 'q' to quit")

try:
    while True:
        # Fetch live camera feed
        frame_rgb = cam.capture_array("main")
        preview = cv2.resize(frame_rgb, PREVIEW_RES)
        cv2.imshow(WINDOW_NAME, preview)

        # Preprocess for prediction
        resized_input = cv2.resize(frame_rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        normalized_input = resized_input / 255.0
        input_tensor = np.expand_dims(normalized_input.astype(np.float32), axis=0)

        # Predict gesture
        tflite_model.set_tensor(INPUT_INDEX, input_tensor)
        tflite_model.invoke()
        predictions = tflite_model.get_tensor(OUTPUT_INDEX)[0]
        top_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        # LED feedback
        if confidence < PREDICT_THRESHOLD:
            led_color = (0, 255, 0)   # green
        elif top_index == 0:
            led_color = (255, 0, 0)   # red
        else:
            led_color = (0, 0, 255)   # blue

        hat.clear(led_color)

        # Break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.stop()
    hat.clear()
    cv2.destroyAllWindows()
