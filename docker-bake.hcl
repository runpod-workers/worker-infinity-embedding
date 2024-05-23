variable "PUSH" {
  default = "true"
}

variable "REPOSITORY" {
  default = "runpod"
}

variable "WORKER_VERSION" {
  default = "0.0.1"
}

group "all" {
  targets = ["worker-1180", "worker-1210"]
}

target "worker-1180" {
  tags = ["${REPOSITORY}/worker-infinity-text-embedding:${WORKER_VERSION}-cuda11.8.0"]
  context = "."
  dockerfile = "Dockerfile"
  args = {
    WORKER_VERSION = "${WORKER_VERSION}"
    WORKER_CUDA_VERSION = "11.8.0"
  }
  output = ["type=docker,push=${PUSH}"]
}

target "worker-1210" {
  tags = ["${REPOSITORY}/worker-infinity-text-embedding:${WORKER_VERSION}-cuda12.1.0"]
  context = "."
  dockerfile = "Dockerfile"
  args = {
    WORKER_VERSION = "${WORKER_VERSION}"
    WORKER_CUDA_VERSION = "12.1.0"
  }
  output = ["type=docker,push=${PUSH}"]
}