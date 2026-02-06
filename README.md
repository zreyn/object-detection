
sudo docker pull dustynv/l4t-ml:r36.2.0

sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    --device /dev/video0 \
    -v ~/object-detection/yolo:/workspace \
    -v /tmp/argus_socket:/tmp/argus_socket \
    dustynv/l4t-ml:r36.2.0 \
    bash

cd workspace
apt update && apt install -y libgl1-mesa-glx libglib2.0-0
pip install ultralytics "numpy<2"
python3 convert.py
python3 main.py
