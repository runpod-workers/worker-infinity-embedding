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
  targets = ["worker-1241"]
}

target "worker-1241" {
  tags = ["${REPOSITORY}/worker-infinity-embedding:${WORKER_VERSION}-cuda12.9"]
  context = "."
  dockerfile = "Dockerfile"
  output = ["type=docker,push=${PUSH}"]
}