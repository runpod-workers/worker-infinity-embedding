from http import HTTPStatus
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4
import time
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, conlist

EmbeddingReturnType = npt.NDArray[Union[np.float32, np.float32]]

# potential backwards compatibility to pydantic 1.X
# pydantic 2.x is preferred by not strictly needed
try:
    from pydantic import StringConstraints

    # Note: adding artificial limit, this might reveal splitting issues on the client side
    #      and is not a hard limit on the server side.
    INPUT_STRING = StringConstraints(max_length=8192 * 15, strip_whitespace=True)
    ITEMS_LIMIT = {
        "min_length": 1,
        "max_length": 2048,
    }
except ImportError:
    from pydantic import constr

    INPUT_STRING = constr(max_length=8192 * 15, strip_whitespace=True)  # type: ignore
    ITEMS_LIMIT = {
        "min_items": 1,
        "max_items": 2048,
    }


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int
    
def create_error_response(message: str, err_type: str = "BadRequestError", status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
    return ErrorResponse(message=message,
                            type=err_type,
                            code=status_code.value)
    
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
    data: ModelInfo
    object: str = "list"
    
class OpenAIEmbeddingResult(BaseModel):
    object: Literal["embedding"] = "embedding"
    data: List[_EmbeddingObject]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))

def list_embeddings_to_response(
    embeddings: Union[EmbeddingReturnType, Iterable[EmbeddingReturnType]],
    model: str,
    usage: int,
) -> Dict[str, Any]:
    return dict(
        model=model,
        data=[
            dict(
                object="embedding",
                embedding=emb,
                index=count,
            )
            for count, emb in enumerate(embeddings)
        ],
        usage=dict(prompt_tokens=usage, total_tokens=usage),
    )
