# ================================
# Stage 1: Base Image
# ================================
FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# ================================
# Install system dependencies
# ================================
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Create ROS 2 workspace
# ================================
WORKDIR /ros2_ws/src

# Copy your workspace (VERY IMPORTANT)
COPY ./ /ros2_ws/src/

# ================================
# Create Python virtual environment
# ================================
RUN python3 -m venv /opt/ros_venv

RUN /opt/ros_venv/bin/pip install --upgrade pip

RUN /opt/ros_venv/bin/pip install \
    gymnasium \
    stable-baselines3 \
    numpy \
    matplotlib \
    opencv-python

# ================================
# Build ROS workspace
# ================================
WORKDIR /ros2_ws

RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build"

# ================================
# Source everything automatically
# ================================
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /opt/ros_venv/bin/activate" >> /root/.bashrc

# ================================
# Default command
# ================================
CMD ["/bin/bash"]