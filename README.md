# YOLO Object Detection with gRPC Microservices

This project implements a YOLO-based object detection system split into two microservices communicating via gRPC:

1.  **Inference Service**: A Python server hosting the YOLOv8 model.
2.  **Camera Service**: A Python client that captures video from a webcam, sends frames to the inference service, and visualizes the results.

## Prerequisites

- Docker and Docker Compose
- A webcam (for the camera service)

## Quick Start (Linux/Windows with Webcam Passthrough)

If your Docker environment supports direct webcam access (e.g., Linux with `/dev/video0`), you can run the entire stack in containers:

```bash
docker-compose up --build
```
Or, if you're using a Jetson CSI camera:
```bash                                                                                      
  USE_CSI_CAMERA=true docker compose up                                                                           
```

This will start both services. The camera service will attempt to open a GUI window with the video feed.  You may need to run this if the camera container can't connect to the display:

```bash
xhost +local:docker
```

The environment supports some camera options:
```
  Optional camera settings:                                                                                       
  - CAMERA_WIDTH (default: 1280)                                                                                  
  - CAMERA_HEIGHT (default: 720)                                                                                  
  - CAMERA_FRAMERATE (default: 30)                                                                                
  - CAMERA_DEVICE (default: 0, for USB webcams)  
```

## Running on macOS (or if Webcam/GUI fails in Docker)

On macOS and some Windows setups, accessing the webcam and showing a GUI from inside a Docker container is difficult. The recommended approach is to run the **Inference Service** in Docker and the **Camera Client** locally.

### 1. Start the Inference Service
```bash
docker-compose up inference
```
Wait for the healthy signal (the server is ready).

### 2. Run the Camera Client Locally

Open a new terminal and install dependencies:
```bash
# Optional: Create a virtual environment
python3 -m venv venvgit 
source venv/bin/activate

# Install requirements
pip install -r camera/requirements.txt
```

Generate the gRPC code (required if running locally):
```bash
python -m grpc_tools.protoc -I. --python_out=./camera --grpc_python_out=./camera protos/service.proto
```

Run the client:
```bash
# Point the client to localhost since Docker maps port 50051
INFERENCE_SERVER=localhost:50051 python camera/client.py
```

## Project Structure

- `protos/`: gRPC service definitions (`service.proto`).
- `inference/`: Server code and Dockerfile.
- `camera/`: Client code and Dockerfile.
- `docker-compose.yml`: Service orchestration.
