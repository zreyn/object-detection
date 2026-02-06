import time
import cv2
from ultralytics import YOLO

MODEL_NAME = 'yolo26n-seg.pt'


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1"
    )


def main():
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            start = time.perf_counter()
            results = model(frame)
            end = time.perf_counter()

            total_time = end - start
            fps = 1 / total_time

            annotated_frame = results[0].plot()

            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (10, 30)
            font_scale = 1
            color = (0, 255, 0)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", position, font, font_scale, color)
            cv2.imshow(MODEL_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    print("cap no longer/never opened")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
