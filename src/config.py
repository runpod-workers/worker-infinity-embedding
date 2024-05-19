import os
from dotenv import load_dotenv
from functools import cached_property 
DEFAULT_BATCH_SIZE = 32
DEFAULT_BACKEND = "torch"

class EmbeddingServiceConfig:
    def __init__(self):
        load_dotenv()
        
    @cached_property
    def backend(self):
        return os.environ.get("BACKEND", DEFAULT_BACKEND)
    
    @cached_property
    def model_names(self) -> list[str]:
        model_names = os.environ.get("MODEL_NAMES")
        if not model_names:
            raise ValueError("MODEL_NAMES environment variable is required")
        model_names = model_names.split(";")
        model_names = [model_name for model_name in model_names if model_name]
        return model_names
    
    @cached_property
    def batch_sizes(self) -> list[int]:
        batch_sizes = os.getenv("BATCH_SIZES", f"{DEFAULT_BATCH_SIZE};" * len(self.model_names)).split(";")
        batch_sizes = [batch_size for batch_size in batch_sizes if batch_size]
        if len(batch_sizes) != len(self.model_names):
            raise ValueError("BATCH_SIZES must have the same number of elements as MODEL_NAMES")
        batch_sizes = [int(batch_size) for batch_size in batch_sizes]
        return batch_sizes
    
    @cached_property
    def default_model_name(self):
        env_default_model_name = os.environ.get("DEFAULT_MODEL_NAME")
        if env_default_model_name in self.model_names:
            return env_default_model_name
        elif len(self.model_names) == 1:
            return self.model_names[0]
        else:
            return None