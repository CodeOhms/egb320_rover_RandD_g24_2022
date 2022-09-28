import io
import time
import picamera
import cv2
import numpy as np

# Create the in-memory stream
stream = io.BytesIO()
with picamera.PiCamera() as camera:
	time.sleep(2)
	camera.capture(stream, format='jpeg')

# Construct a numpy array from the stream
data = np.frombuffer(stream.getvalue(), dtype=np.uint8)

# "Decode" the image from the array, preserving colour
frame = cv2.imdecode(data, 1)

frame_blue = frame[:,:,0];							# Extract blue channel
ret, thresholded_frame = cv2.threshold(frame_blue, 127, 255, cv2.THRESH_BINARY)	# Threshold blue channel

cv2.imshow("Binary Thresholded Frame", thresholded_frame)			# Display thresholded frame
cv2.waitKey(0)									# Exit on keypress

cv2.destroyAllWindows()