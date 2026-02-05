import time
import cv2
from ultralytics import YOLO

MODEL_NAME = 'yolo26n-seg.pt'

def main():
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(0)

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
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
