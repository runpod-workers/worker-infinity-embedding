# ATTENTION: is this still the right CUDA version?
ARG WORKER_CUDA_VERSION=12.1.0 
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

#Reinitialize, as its lost after the FROM command
# &efron: this doesn't quite follow to me. 
ARG WORKER_CUDA_VERSION=12.1.0

# Python dependencies.



RUN --mount-type=cache,target=/root/.cache/pip python3.11 -m pip install --upgrade pip

# we're always going to do this. important to do this FIRST - it can take >2m and we want to cache the result as early as possible.
# TODO: pin this to a specific version 
RUN --mount-type=cache,target=/root/.cache/pip python3.11 -m pip install torch torchvision torchaudio

# ourother requirements may change; updating the version of infinity embedding, for instance.
COPY builder/requirements.txt /requirements.txt
RUN --mount-type=cache,target=/root/.cache/pippython3.11 -m pip install -r /requirements.txt
COPY ./src /src
CMD python3.11 -u /src/handler.py