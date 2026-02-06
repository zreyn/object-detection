import grpc
import cv2
import time
import numpy as np
import service_pb2
import service_pb2_grpc
import os

def main():
    # Connect to the inference server
    # 'inference' is the hostname in docker-compose
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

    cap = cv2.VideoCapture(0)
    
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
