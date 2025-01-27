ARG WORKER_CUDA_VERSION=12.4.1
FROM runpod/pytorch:2.4.0-py3.11-cuda${WORKER_CUDA_VERSION}-devel-ubuntu22.04

#Reinitialize, as its lost after the FROM command
ARG WORKER_CUDA_VERSION=12.4.1

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

RUN pip uninstall torch -y && \
    CUDA_VERSION_SHORT=$(echo ${WORKER_CUDA_VERSION} | cut -d. -f1,2 | tr -d .) && \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/test/cu${CUDA_VERSION_SHORT} --no-cache-dir

ENV HF_HOME=/runpod-volume

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py
