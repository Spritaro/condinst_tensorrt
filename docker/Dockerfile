FROM nvcr.io/nvidia/tensorrt:20.12-py3
# TensorRT 7.2.2
# CUDA 11.1.1
# cuDNN 8.0.5
# Python 3.8.5

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-dev \
    libgtk2.0-dev \
    pkg-config \
    python3-dev \
    wget

# Create non-root user
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --groups video -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME

# Install latest pip
ENV PATH=/home/appuser/.local/bin:${PATH}
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# Install COCOAPI
RUN pip3 install --upgrade cython
RUN git clone https://github.com/philferriere/cocoapi.git \
    && cd cocoapi/PythonAPI \
    && python3 setup.py build_ext install --user

# Install pytorch
RUN pip3 install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install \
    albumentations==1.3.0 \
    matplotlib==3.6.3 \
    "numpy<1.20" \
    opencv-python==4.8.0.76 \
    "pillow<10.0.0" \
    pytorch-lightning==1.7.0 \
    torchmetrics==0.11.4
