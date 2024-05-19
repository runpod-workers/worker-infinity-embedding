ARG WORKER_CUDA_VERSION=11.8.0
FROM runpod/base:0.6.1-cuda${WORKER_CUDA_VERSION}


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
ENV HF_HOME=/runpod-volume

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py
