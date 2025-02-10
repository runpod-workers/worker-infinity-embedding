import os
from dotenv import load_dotenv
from functools import cached_property
from runpod import RunPodLogger
from urllib.parse import urljoin

logger = RunPodLogger()

DEFAULT_BATCH_SIZE = 32
DEFAULT_BACKEND = "torch"

if not os.environ.get("INFINITY_QUEUE_SIZE"):
    # how many items can be in the queue
    os.environ["INFINITY_QUEUE_SIZE"] = "48000"

MODEL_CACHE_PATH_TEMPLATE = "/runpod/cache/model/{path}"

CONFIG_MESSAGE_TEMPLATE = "{message} [see https://github.com/runpod-workers/worker-infinity-embedding for more information]"


def topath(raw: str) -> str:
    raw = raw.strip()
    if ":" in raw:
        model, branch = raw.rsplit(":", maxsplit=1)
    else:
        model, branch = raw, "main"
    if "/" not in model:
        raise ValueError(
            f"invalid model: expected one in the form user/model[:path], but got {model}"
        )
    user, model = model.rsplit("/", maxsplit=1)
    return MODEL_CACHE_PATH_TEMPLATE.format(
        path="/".join(c.strip("/") for c in (user, model, branch))
    )


def modelpaths(path: str = "") -> list[str]:
    raw = os.environ.get("RUNPOD_HUGGINGFACE_MODEL", path)
    if not raw:
        return []
    return [topath(m) for m in raw.split(",")]


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
        deprecated_model_names = os.environ.get(
            "MODEL_NAMES", "/BAAI/bge-small-en-v1.5"
        )
        if not deprecated_model_names:
            logger.warn(
                CONFIG_MESSAGE_TEMPLATE.format(
                    message="MODEL_NAMES is deprecated, use RUNPOD_HUGGINGFACE_MODEL"
                )
            )
        cache_paths: list[str] = modelpaths(deprecated_model_names)
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
