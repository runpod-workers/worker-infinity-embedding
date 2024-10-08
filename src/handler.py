import runpod
from utils import create_error_response
from typing import Any
from embedding_service import EmbeddingService

embedding_service = EmbeddingService()


async def async_generator_handler(job: dict[str, Any]):
    """Handle the requests and embedding/rerank them asynchronously."""
    job_input = job["input"]
    if job_input.get("openai_route"):
        openai_route, openai_input = job_input.get("openai_route"), job_input.get(
            "openai_input"
        )

        if openai_route and openai_route == "/v1/models":
            call_fn, kwargs = embedding_service.route_openai_models, {}
        elif openai_route and openai_route == "/v1/embeddings":
            model_name = openai_input.get("model")
            if not openai_input:
                return create_error_response("Missing input").model_dump()
            if not model_name:
                return create_error_response(
                    "Did not specify model in openai_input"
                ).model_dump()
            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": openai_input.get("input"),
                "model_name": model_name,
                "return_as_list": True,
            }
        else:
            return create_error_response(
                f"Invalid OpenAI Route: {openai_route}"
            ).model_dump()
    else:
        # handle the request for reranking
        if job_input.get("query"):
            call_fn, kwargs = embedding_service.infinity_rerank, {
                "query": job_input.get("query"),
                "docs": job_input.get("docs"),
                "return_docs": job_input.get("return_docs"),
                "model_name": job_input.get("model"),
            }
        elif job_input.get("input"):
            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": job_input.get("input"),
                "model_name": job_input.get("model"),
            }
        # handle image urls (for image embeddings)
        elif job_input.get("image_input"):
            call_fn, kwargs = embedding_service.route_get_image_embeddings, {
                "image_input": job_input.get("image_input"),
                "model_name": job_input.get("model"),
            }
        else:
            return create_error_response(f"Invalid input: {job}").model_dump()
    try:
        out = await call_fn(**kwargs)
        return out
    except Exception as e:
        return create_error_response(str(e)).model_dump()


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": async_generator_handler,
            "concurrency_modifier": lambda x: embedding_service.config.runpod_max_concurrency,
        }
    )
