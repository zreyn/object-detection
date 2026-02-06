import cv2

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, framerate=(fraction)30/1 ! "
    "nvvidconv ! "
    "video/x-raw, width=(int)960, height=(int)540, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! "
    "appsink drop=true max-buffers=1"
)

print(f"Pipeline:\n{pipeline}\n")

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print(f"isOpened: {cap.isOpened()}")

if cap.isOpened():
    ret, frame = cap.read()
    print(f"Got frame: {ret}, shape: {frame.shape if ret else 'N/A'}")
    cap.release()
