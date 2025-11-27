FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04 AS base

ENV HF_HOME=/runpod-volume

# install python and other packages
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# install uv
RUN pip install uv

# install python dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --system

RUN rm -rf /usr/local/lib/python3.11/dist-packages/optimum*

# install torch
RUN python -m pip install --upgrade transformers accelerate colpali-engine

# Add src files
ADD src .

# Add test input
COPY test_input.json /test_input.json

# start the handler
CMD python -u /handler.py
