import grpc
from concurrent import futures
import time
import numpy as np
import cv2
from ultralytics import YOLO
import service_pb2
import service_pb2_grpc

MODEL_NAME = 'yolo26n-seg.pt'

class YoloService(service_pb2_grpc.YoloServiceServicer):
    def __init__(self):
        print(f"Loading model {MODEL_NAME}...")
        self.model = YOLO(MODEL_NAME)
        print("Model loaded.")

    def Detect(self, request, context):
        # Decode image
        image_bytes = request.data
        width = request.width
        height = request.height
        channels = request.channels
        
        # Convert bytes to numpy array
        # Note: Depending on how bytes are sent, might need np.frombuffer or similar
        # Assuming raw bytes from tobytes()
        frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, channels))
        
        # Run inference
        results = self.model(frame)
        
        detections = []
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # box.xyxy is [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]
                
                detection = service_pb2.Detection(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls,
                    class_name=class_name
                )
                detections.append(detection)
                
        return service_pb2.DetectionResponse(detections=detections)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add Yolo Servicer
    service_pb2_grpc.add_YoloServiceServicer_to_server(YoloService(), server)
    
    # Add Health Servicer
    from grpc_health.v1 import health
    from grpc_health.v1 import health_pb2
    from grpc_health.v1 import health_pb2_grpc

    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=10),
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    # Mark all services as healthy
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("yolo.YoloService", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port('[::]:50051')
    print("Starting server on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
