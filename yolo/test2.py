import cv2

# Simple test pattern - no camera needed
pipeline = "videotestsrc ! video/x-raw,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print(f"Test source isOpened: {cap.isOpened()}")

if cap.isOpened():
    ret, frame = cap.read()
    print(f"Got frame: {ret}, shape: {frame.shape if ret else 'N/A'}")
    cap.release()
