FROM ros:humble-ros-base
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY ros2/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r /tmp/requirements.txt
WORKDIR /app
COPY src/onnx_yolo.py src/iou_tracker.py src/warning_logic.py /app/
COPY ros2/container_check.py /app/container_check.py
COPY ros2/nodes /app/nodes
COPY ros2/launch /app/launch
ENV PYTHONPATH=/app
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && exec ros2 launch /app/launch/pipeline.launch.py"]
