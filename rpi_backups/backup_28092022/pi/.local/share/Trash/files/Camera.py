import cv2
import picamera
import time

with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.start_preview()
    time.sleep(2)
    camera.capture('foo.jpg')

    cap = cv2.VideoCapture(0)  		# Connect to camera 0 (or the only camera)
    cap.set(3, 320)                     	# Set the width to 320
    cap.set(4, 240)                     	# Set the height to 240
    ret, frame = cap.read()	     		# Get a frame from the camera 
    
    if ret == True:				     # Check if data was obtained successfully
        	cv2.imshow("CameraImage", frame)     # Display the obtained frame in a window called "CameraImage"
    cv2.waitKey()
	     # Make the program wait until you press a key before continuing.
         
    cap.release()
    cv2.destroyAllWindows() 