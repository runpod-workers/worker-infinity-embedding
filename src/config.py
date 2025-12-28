import os
from dotenv import load_dotenv
from functools import cached_property
from typing import Optional

DEFAULT_BATCH_SIZE = 32
DEFAULT_BACKEND = "torch"

DEFAULT_HTTP_CLIENT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_HTTP_CLIENT_TIMEOUT_SECONDS = 10.0
DEFAULT_HTTP_CLIENT_MAX_CONNECTIONS = 50
DEFAULT_HTTP_CLIENT_MAX_KEEPALIVE_CONNECTIONS = 20

if not os.environ.get("INFINITY_QUEUE_SIZE"):
    # how many items can be in the queue
    os.environ["INFINITY_QUEUE_SIZE"] = "48000"


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
        model_names = os.environ.get("MODEL_NAMES")
        if not model_names:
            raise ValueError(
                "Missing required environment variable 'MODEL_NAMES'.\n"
                "Please provide at least one HuggingFace model ID, or multiple IDs separated by a semicolon.\n"
                "Examples:\n"
                "  MODEL_NAMES=BAAI/bge-small-en-v1.5\n"
                "  MODEL_NAMES=BAAI/bge-small-en-v1.5;intfloat/e5-large-v2\n"
            )
        model_names = model_names.split(";")
        model_names = [model_name for model_name in model_names if model_name]
        return model_names

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


class HttpClientConfig:
    ENV_USER_AGENT = "HTTP_CLIENT_USER_AGENT"
    ENV_TIMEOUT = "HTTP_CLIENT_TIMEOUT"
    ENV_MAX_CONNECTIONS = "HTTP_CLIENT_MAX_CONNECTIONS"
    ENV_MAX_KEEPALIVE = "HTTP_CLIENT_MAX_KEEPALIVE_CONNECTIONS"

    def __init__(self):
        load_dotenv()

    def _get_env_float(self, key: str, default: float) -> float:
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Environment variable {key} must be a float, got {value!r}") from exc

    def _get_env_int(self, key: str, default: int) -> int:
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Environment variable {key} must be an integer, got {value!r}") from exc

    def _get_env_str(self, key: str, default: str) -> str:
        value: Optional[str] = os.environ.get(key)
        if value is None:
            return default
        value = value.strip()
        return value or default

    @cached_property
    def user_agent(self) -> str:
        return self._get_env_str(self.ENV_USER_AGENT, DEFAULT_HTTP_CLIENT_USER_AGENT)

    @cached_property
    def timeout_seconds(self) -> float:
        return self._get_env_float(self.ENV_TIMEOUT, DEFAULT_HTTP_CLIENT_TIMEOUT_SECONDS)

    @cached_property
    def max_connections(self) -> int:
        return self._get_env_int(
            self.ENV_MAX_CONNECTIONS,
            DEFAULT_HTTP_CLIENT_MAX_CONNECTIONS,
        )

    @cached_property
    def max_keepalive_connections(self) -> int:
        return self._get_env_int(
            self.ENV_MAX_KEEPALIVE,
            DEFAULT_HTTP_CLIENT_MAX_KEEPALIVE_CONNECTIONS,
        )
