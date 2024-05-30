ARG WORKER_CUDA_VERSION=12.1.0
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

#Reinitialize, as its lost after the FROM command
ARG WORKER_CUDA_VERSION=12.1.0

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

RUN pip uninstall torch -y && \
    CUDA_VERSION_SHORT=$(echo ${WORKER_CUDA_VERSION} | cut -d. -f1,2 | tr -d .) && \
    pip install --pre torch==2.4.0.dev20240518+cu${CUDA_VERSION_SHORT} --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION_SHORT} --no-cache-dir

ENV HF_HOME=/runpod-volume

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py
