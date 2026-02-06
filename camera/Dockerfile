FROM python:3.10-slim

WORKDIR /app

# Install dependencies for OpenCV GUI
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev pkg-config && rm -rf /var/lib/apt/lists/*

COPY camera/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto file
COPY protos/service.proto ./protos/service.proto

# Copy client code
COPY camera/client.py .

# Generate gRPC code
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. protos/service.proto

CMD ["python", "-u", "client.py"]
