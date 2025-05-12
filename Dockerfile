FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="10.0"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    wget \
    tar \
    build-essential \
    libgl1-mesa-dev \
    curl \
    unzip \
    git \
    python3-dev \
    python3-pip \
    libglib2.0-0 \
    && apt clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-12.8" >> /etc/bash.bashrc

RUN pip3 install --pre \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

COPY . /streamdiffusion
WORKDIR /streamdiffusion

RUN python setup.py develop easy_install streamdiffusion[tensorrt] \
    && sed -i 's/from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info/from huggingface_hub import HfFolder, hf_hub_download, model_info/' /usr/local/lib/python3.10/dist-packages/diffusers-0.24.0-py3.10.egg/diffusers/utils/dynamic_modules_utils.py

RUN pip3 install tensorrt-cu12 polygraphy onnx-graphsurgeon

WORKDIR /home/ubuntu/streamdiffusion
