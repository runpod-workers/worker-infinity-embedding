""" Example handler file. """
import os

import runpod
from utils import create_error_response, OpenAIEmbeddingInput, OpenAIEmbeddingResult, OpenAIModelInfo, ModelInfo, list_embeddings_to_response
from typing import Optional, Union
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

model_names = os.environ.get("MODEL_NAMES")
if not model_names:
    raise ValueError("MODEL_NAMES environment variable is required")
model_names = model_names.split(";")
default_model_name = None
if len(model_names) == 1:
    default_model_name = model_names[0]

batch_size = int(os.environ.get("BATCH_SIZE", 64))

backend = os.environ.get("BACKEND", "torch")

engine_args_list = {model_name: EngineArgs(model_name_or_path=model_name, engine=backend, batch_size=batch_size) for model_name in model_names}
engines = {model_name: AsyncEmbeddingEngine(engine_args) for model_name, engine_args in engine_args_list.items()}

async def async_generator_handler(job):
    job_input = job['input']
    
    openai_route = job_input.get("openai_route")
    if openai_route:
        openai_input = job_input.get("openai_input")
        model_name = openai_input.get("model")
        engine = engines.get(model_name)
        engine_args = engine_args_list.get(model_name)
        if not engine:
            return create_error_response(f"Model '{model_name}' not found").model_dump()
        
        if openai_route == "/v1/embeddings":
            embedding_input = openai_input.get("input")
            if isinstance(embedding_input, str):
                embedding_input = [embedding_input]
            try:
                embeddings, usage = await engine.embed(embedding_input)
                result = list_embeddings_to_response(embeddings, model_name, usage)
                return OpenAIEmbeddingResult(**result).model_dump()
            except Exception as e:
                return create_error_response(str(e)).model_dump()
        elif openai_route == "/v1/models":
            return OpenAIModelInfo(data=ModelInfo(id=engine_args.model_name_or_path, stats=dict(batch_size=engine_args.batch_size), backend=engine_args.engine.name)).model_dump()
        else:
            return create_error_response(f"Invalid route: {openai_route}").model_dump()
        
    else:
        model_name = job_input.get("model_name")
        request_type = job_input.get("request_type", "embed")
        
        if not model_name:
            if not default_model_name:
                return {"error": "model_name input is required when there is more than one model"}
            else:
                model_name = default_model_name
        engine = engines.get(model_name)
        
        if request_type == "embed":
            sentences = job_input.get("sentences", [])
            if not sentences:
                return {"error": "'sentences' input is required for embedding"}
            if not all(isinstance(sentence, str) for sentence in sentences):
                return {"error": "sentences must be a list of strings"}
            async with engine:
                embeddings, usage = await engine.embed(sentences=sentences)
            return {"embeddings": embeddings, "usage": usage}
        elif request_type == "rerank":
            query = job_input.get("query")
            if not query:
                return {"error": "'query' input is required"}
            docs = job_input.get("docs", [])
            if not docs:
                return {"error": "'docs' input is required for reranking"}
            async with engine:
                ranking, usage = await engine.rerank(query=query, docs=docs)
            return {"ranking": ranking, "usage": usage}


runpod.serverless.start({"handler": async_generator_handler})
