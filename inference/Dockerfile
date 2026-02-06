FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model
COPY inference/yolo26n-seg.pt .

# Copy proto file
COPY protos/service.proto ./protos/service.proto

# Copy server code
COPY inference/server.py .

# Generate gRPC code
RUN cd protos && python -m grpc_tools.protoc -I. --python_out=../ --grpc_python_out=../ service.proto

EXPOSE 50051

CMD ["python", "-u", "server.py"]
