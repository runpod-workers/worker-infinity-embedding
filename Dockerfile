ARG WORKER_CUDA_VERSION=12.1.0
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

#Reinitialize, as its lost after the FROM command
ARG WORKER_CUDA_VERSION=12.1.0

# Define the argument for the Hugging Face token
ARG JOORI_EMBEDDING_HUGGINGFACE_TOKEN

# Python dependencies
COPY builder/requirements.txt /requirements.txt
COPY src/download.py /download.py

# Install Python dependencies
RUN python3.11 -m pip install --upgrade pip

RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

RUN python3.11 -m pip install --no-cache-dir flash-attn --no-build-isolation

# Download the model
RUN python3.11 download.py $JOORI_EMBEDDING_HUGGINGFACE_TOKEN

RUN pip uninstall torch -y && \
    CUDA_VERSION_SHORT=$(echo ${WORKER_CUDA_VERSION} | cut -d. -f1,2 | tr -d .) && \
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION_SHORT} --no-cache-dir

ENV HF_HOME=/runpod-volume

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py
