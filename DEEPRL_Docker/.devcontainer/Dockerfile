FROM nvidia/cuda:10.2-base
CMD nvidia-smi

RUN apt-get update &&\
    apt-get install -y libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    ffmpeg \
    curl \
    patchelf \
    libglfw3 \
    libglfw3-dev \
    cmake \
    zlib1g \
    zlib1g-dev \
    swig \
    python3 \
    python3-pip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install 'gym'
RUN pip3 install 'gym[Box2D]'