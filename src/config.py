import os
from dotenv import load_dotenv
from constants import DEFAULT_BATCH_SIZE, DEFAULT_BACKEND

class EmbeddingServiceConfig:
    def __init__(self):
        load_dotenv()
        self.backend = os.environ.get("BACKEND", DEFAULT_BACKEND)
        self.model_names = self._get_model_names()
        self.batch_sizes = self._get_batch_sizes()
        self.default_model_name = self._get_default_model_name()
        
    def _get_model_names(self):
        model_names = os.environ.get("MODEL_NAMES")
        if not model_names:
            raise ValueError("MODEL_NAMES environment variable is required")
        model_names = model_names.split(";")
        return model_names
    
    def _get_batch_sizes(self):
        batch_sizes = os.getenv("BATCH_SIZES", f"{DEFAULT_BATCH_SIZE};" * len(self.model_names)).split(";")
        batch_sizes = [batch_size for batch_size in batch_sizes if batch_size]
        if len(batch_sizes) != len(self.model_names):
            raise ValueError("BATCH_SIZES must have the same number of elements as MODEL_NAMES")
        batch_sizes = [int(batch_size) for batch_size in batch_sizes]
        return batch_sizes
    
    def _get_default_model_name(self):
        env_default_model_name = os.environ.get("DEFAULT_MODEL_NAME")
        if env_default_model_name in self.model_names:
            return env_default_model_name
        elif len(self.model_names) == 1:
            return self.model_names[0]
        else:
            return None