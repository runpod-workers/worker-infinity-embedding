import runpod
from utils import create_error_response
from embedding_service import EmbeddingService

embedding_service = EmbeddingService()

async def async_generator_handler(job):
    job_input = job['input']
    openai_route, openai_input = job_input.get("openai_route"), job_input.get("openai_input")
    if openai_route:
        if openai_route == "/v1/models":
            return embedding_service.openai_get_models()
        elif openai_route == "/v1/embeddings":
            if not openai_input:
                return create_error_response("Missing input").model_dump()
            requested_model_name = openai_input.get("model")
            input_to_process = openai_input.get("input")
            processor_func = embedding_service.openai_get_embeddings
        else:
            return create_error_response(f"Invalid OpenAI Route: {openai_route}").model_dump()
    else:
        requested_model_name = job_input.get("model") or embedding_service.config.default_model_name
        input = job_input.get("input")
        query, docs, return_docs = job_input.get("query"), job_input.get("docs"), job_input.get("return_docs")
        if query:
            input_to_process = {"query": query, "docs": docs, "return_docs": return_docs}
            processor_func = embedding_service.infinity_rerank
        elif input: 
            ### Non-OpenAI Embed not available until fixed
            # input_to_process = job_input.get("sentences", [])
            # processor_func = embedding_service.infinity_embed
            input_to_process = input
            processor_func = embedding_service.openai_get_embeddings
        else:
            return create_error_response(f"Invalid input: {job_input}").model_dump()
        
    model = embedding_service.models.get(requested_model_name)
    if not model:
        return create_error_response(f"Model '{requested_model_name}' not found.\n Request: {job}\n Available models: {list(embedding_service.models.keys())}").model_dump()
    
    try:
        return await processor_func(input_to_process, model)
    except Exception as e:
        return create_error_response(str(e)).model_dump()

runpod.serverless.start({"handler": async_generator_handler, "concurrency_modifier": lambda x: 3000})