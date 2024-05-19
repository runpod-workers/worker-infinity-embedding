from http import HTTPStatus
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4
import time
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, conlist

EmbeddingReturnType = npt.NDArray[Union[np.float32, np.float32]]
try:
    from pydantic import StringConstraints

    INPUT_STRING = StringConstraints(max_length=8192 * 15, strip_whitespace=True)
    ITEMS_LIMIT = {
        "min_length": 1,
        "max_length": 8192,
    }
except ImportError:
    from pydantic import constr

    INPUT_STRING = constr(max_length=8192 * 15, strip_whitespace=True)  # type: ignore
    ITEMS_LIMIT = {
        "min_items": 1,
        "max_items": 8192,
    }


async def process_embedding_request(job_input, engines):
    openai_input = job_input.get("openai_input")
    model_name = openai_input.get("model")
    engine = engines.get(model_name)
    if not engine:
        return create_error_response(f"Model '{model_name}' not found").model_dump()

    embedding_input = openai_input.get("input")
    if isinstance(embedding_input, str):
        embedding_input = [embedding_input]
    try:
        async with engine:
            embeddings, usage = await engine.embed(embedding_input)
        result = list_embeddings_to_response(embeddings, model_name, usage)
        return OpenAIEmbeddingResult(**result).model_dump()
    except Exception as e:
        return create_error_response(str(e)).model_dump()


def process_model_info_request(job_input, engines):
    openai_input = job_input.get("openai_input")
    model_name = openai_input.get("model")
    engine_args = engines.get(model_name)
    if not engine_args:
        return create_error_response(f"Model '{model_name}' not found").model_dump()
    return OpenAIModelInfo(
        data=ModelInfo(
            id=engine_args.model_name_or_path,
            stats=dict(batch_size=engine_args.batch_size),
            backend=engine_args.engine,
        )
    ).model_dump()


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    return ErrorResponse(message=message, type=err_type, code=status_code.value)


class OpenAIEmbeddingInput(BaseModel):
    input: Union[  # type: ignore
        conlist(  # type: ignore
            Annotated[str, INPUT_STRING],
            **ITEMS_LIMIT,
        ),
        Annotated[str, INPUT_STRING],
    ]
    model: Optional[str] = None
    user: Optional[str] = None


class _EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class _Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class ModelInfo(BaseModel):
    id: str
    stats: Dict[str, Any]
    object: Literal["model"] = "model"
    owned_by: Literal["infinity"] = "infinity"
    created: int = int(time.time())  # no default factory
    backend: str = ""


class OpenAIModelInfo(BaseModel):
    data: List[ModelInfo] = Field(default_factory=list)
    object: str = "list"


class OpenAIEmbeddingResult(BaseModel):
    object: str = "list"
    data: List[_EmbeddingObject]
    model: str
    usage: _Usage


def list_embeddings_to_response(
    embeddings: Union[EmbeddingReturnType, Iterable[EmbeddingReturnType]],
    model: str,
    usage: int,
) -> Dict[str, Any]:
    return dict(
        model=model,
        object="list",
        data=[
            dict(
                object="embedding",
                embedding=emb.tolist(),
                index=count,
            )
            for count, emb in enumerate(embeddings)
        ],
        usage=dict(prompt_tokens=usage, total_tokens=usage),
    )


def to_rerank_response(
    scores: List[float],
    model=str,
    usage=int,
    documents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if documents is None:
        return dict(
            model=model,
            results=[
                dict(relevance_score=score, index=count)
                for count, score in enumerate(scores)
            ],
            usage=dict(prompt_tokens=usage, total_tokens=usage),
        )
    else:
        return dict(
            model=model,
            results=[
                dict(relevance_score=score, index=count, document=doc)
                for count, (score, doc) in enumerate(zip(scores, documents))
            ],
            usage=dict(prompt_tokens=usage, total_tokens=usage),
        )
