import time
import numpy as np
from PIL import Image
import tensorflow as tf
from sense_hat import SenseHat
from picamera2 import Picamera2

sense = SenseHat()
sense.clear()

# load TFLite model
interpreter = tf.lite.Interpreter(model_path="cnn_gesture_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()
input_size = tuple(input_details[0]["shape"][1:3])

# start camera
picam2 = Picamera2()

# 640 480 RGB frames , delivered as a numpy array
config = picam2.create_preview_configuration(
main ={"format": "RGB888", "size": (640,480)})

picam2.configure(config)
picam2.start()
time.sleep(2)
GREEN = (0,255,0)
RED = (255,0,0)

def show(color):
	sense.clear(color)
# main loop
try :
	while True :
		frame = picam2.capture_array()
		img = Image.fromarray(frame).resize(input_size)
		img = (np.asarray(img ,dtype=np.float32)/255.0)[None,...] # add batch dim
		interpreter.set_tensor(input_details [0]["index"],img)
		interpreter.invoke()
		prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]
	if prediction > 0.5:
		print ("Thumbs up? detected !")
		show(GREEN)
	else:
		print("No detection")
		show(RED)
		time.sleep(2)
except KeyboardInterrupt:
	print (" Exited with Ctrl -C")

finally:
	sense.clear()
	picam2.stop()
