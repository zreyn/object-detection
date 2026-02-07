import grpc
import cv2
import time
import numpy as np
import service_pb2
import service_pb2_grpc
import os

def get_camera_capture():
    """Get video capture - supports Jetson CSI cameras via GStreamer or USB webcams."""
    use_csi = os.environ.get('USE_CSI_CAMERA', 'false').lower() == 'true'

    if use_csi:
        # Jetson CSI camera using GStreamer pipeline
        width = int(os.environ.get('CAMERA_WIDTH', '1280'))
        height = int(os.environ.get('CAMERA_HEIGHT', '720'))
        framerate = int(os.environ.get('CAMERA_FRAMERATE', '30'))

        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={framerate}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        print(f"Using CSI camera with GStreamer pipeline")
        return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        # USB webcam or default V4L2
        device = int(os.environ.get('CAMERA_DEVICE', '0'))
        print(f"Using V4L2 camera device {device}")
        return cv2.VideoCapture(device)


def main():
    # Connect to the inference server
    # 'inference' is the hostname in docker-compose
    server_address = os.environ.get('INFERENCE_SERVER', 'inference:50051')

    # Wait for server to be healthy
    from grpc_health.v1 import health_pb2
    from grpc_health.v1 import health_pb2_grpc
    
    print(f"Connecting to inference server at {server_address}...")
    channel = grpc.insecure_channel(server_address)
    health_stub = health_pb2_grpc.HealthStub(channel)
    
    print("Waiting for server to be ready...")
    while True:
        try:
            response = health_stub.Check(health_pb2.HealthCheckRequest(service="yolo.YoloService"))
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                print("Server is ready!")
                break
        except grpc.RpcError as e:
            # Server might not be up yet
            pass
        time.sleep(1)

    stub = service_pb2_grpc.YoloServiceStub(channel)

    cap = get_camera_capture()
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        height, width, channels = frame.shape
        
        # Prepare request
        # service.proto expects bytes
        image_bytes = frame.tobytes()
        
        request = service_pb2.ImageFrame(
            data=image_bytes,
            width=width,
            height=height,
            channels=channels
        )
        
        start_time = time.perf_counter()
        
        try:
            response = stub.Detect(request)
            detections = response.detections
        except grpc.RpcError as e:
            print(f"RPC Error: {e}")
            detections = []

        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time) if end_time > start_time else 0
        
        # Plot detections
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            conf = det.confidence
            cls_name = det.class_name
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), font, 1, (0, 255, 0), 2)
        
        # Show frame
        # Wrap in try/except for headless environments
        try:
            cv2.imshow("YOLO Client", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            print(f"cv2 error (HEADLESS?): {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Give the server a moment to start if running together
    time.sleep(5) 
    main()
