import os
from dotenv import load_dotenv
from functools import cached_property
from runpod import RunPodLogger

logger = RunPodLogger()

DEFAULT_BATCH_SIZE = 32
DEFAULT_BACKEND = "torch"

if not os.environ.get("INFINITY_QUEUE_SIZE"):
    # how many items can be in the queue
    os.environ["INFINITY_QUEUE_SIZE"] = "48000"

MODEL_CACHE_PATH_TEMPLATE = "/runpod/cache/{model}/{revision}"

CONFIG_MESSAGE_TEMPLATE = "{message} [see https://github.com/runpod-workers/worker-infinity-embedding for more information]"


class EmbeddingServiceConfig:
    def __init__(self):
        load_dotenv()

    def _get_no_required_multi(self, name, default=None):
        out = os.getenv(name, f"{default};" * len(self.model_names)).split(";")
        out = [o for o in out if o]
        if len(out) != len(self.model_names):
            raise ValueError(
                f"Env var: {name} must have the same number of elements as MODEL_NAMES"
            )
        return out

    @cached_property
    def backend(self):
        return os.environ.get("BACKEND", DEFAULT_BACKEND)

    @cached_property
    def model_names(self) -> list[str]:
        # check if the legacy env var is defined
        deprecated_model_names = os.environ.get("MODEL_NAMES", "")
        if not deprecated_model_names:
            logger.warn(
                CONFIG_MESSAGE_TEMPLATE.format(
                    message="MODEL_NAMES is deprecated, use RUNPOD_HUGGINGFACE_MODEL"
                )
            )
        cache_paths: list[str] = [
            MODEL_CACHE_PATH_TEMPLATE.format(
                # the model is always the first element
                model=repository_and_revision[0],
                # the revision is the second element if it exists
                revision=repository_and_revision[1]
                if len(repository_and_revision) > 1
                else "main",
            )
            # the repository is split into the model and revision by the last colon
            for repository_and_revision in (
                repository.rsplit(":", 1)
                for repository in (
                    *(os.environ.get("RUNPOD_HUGGINGFACE_MODEL", "").split(",")),
                    *deprecated_model_names.split(";"),
                )
            )
        ]
        if not cache_paths:
            raise ValueError(
                CONFIG_MESSAGE_TEMPLATE.format(
                    message="RUNPOD_HUGGINGFACE_MODEL environment variable is required"
                )
            )
        return sorted(cache_paths)

    @cached_property
    def batch_sizes(self) -> list[int]:
        batch_sizes = self._get_no_required_multi("BATCH_SIZES", DEFAULT_BATCH_SIZE)
        batch_sizes = [int(batch_size) for batch_size in batch_sizes]
        return batch_sizes

    @cached_property
    def dtypes(self) -> list[str]:
        dtypes = self._get_no_required_multi("DTYPES", "auto")
        return dtypes

    @cached_property
    def runpod_max_concurrency(self) -> int:
        return int(os.environ.get("RUNPOD_MAX_CONCURRENCY", 300))
