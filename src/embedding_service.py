import asyncio
import logging

from infinity_emb.engine import AsyncEngineArray, EngineArgs
from infinity_emb.primitives import ModelNotDeployedError
from PIL import Image

from config import EmbeddingServiceConfig
from http_client import create_http_client
from multimodal_utils import validate_item_for_modality
from utils import (
    ModelInfo,
    OpenAIModelInfo,
    list_embeddings_to_response,
    to_rerank_response,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.config = EmbeddingServiceConfig()
        engine_args = []
        for model_name, batch_size, dtype in zip(
            self.config.model_names, self.config.batch_sizes, self.config.dtypes
        ):
            engine_args.append(
                EngineArgs(
                    model_name_or_path=model_name,
                    batch_size=batch_size,
                    engine=self.config.backend,
                    dtype=dtype,
                    model_warmup=False,
                    lengths_via_tokenize=True,
                )
            )

        self.engine_array = AsyncEngineArray.from_args(engine_args)
        self.is_running = False
        self.sepamore = asyncio.Semaphore(1)
        self.http_client = None

    async def start(self):
        """starts the engine background loop"""
        async with self.sepamore:
            if not self.is_running:
                await self.engine_array.astart()
                if self.http_client is None:
                    self.http_client = create_http_client()
                    logger.info("Created persistent HTTP client for image downloads")

                self.is_running = True

    async def stop(self):
        """stops the engine background loop"""
        async with self.sepamore:
            if self.is_running:
                if self.http_client is not None:
                    await self.http_client.aclose()
                    self.http_client = None
                    logger.info("Closed HTTP client")

                await self.engine_array.astop()
                self.is_running = False

    async def route_openai_models(self) -> OpenAIModelInfo:
        return OpenAIModelInfo(
            data=[ModelInfo(id=model_id, stats={}) for model_id in self.list_models()]
        ).model_dump()

    def list_models(self) -> list[str]:
        return list(self.engine_array.engines_dict.keys())

    async def route_openai_get_embeddings(
        self,
        embedding_input: str | list[str] | list[str | bytes | Image.Image],
        model_name: str,
        modality: str = "text",
        return_as_list: bool = False,
    ):
        """
        Returns embeddings for the input based on specified modality.

        Args:
            embedding_input: Input text(s) or image(s) to embed
            model_name: Name of the model to use
            modality: Type of input - "text", "image", or "audio" (not yet implemented)
            return_as_list: Whether to return results as a list

        Raises:
            ValueError: If model not available, modality invalid, or validation fails
            NotImplementedError: If modality is "audio"
        """
        try:
            if not self.is_running:
                await self.start()

            available_models = self.list_models()
            if model_name not in available_models:
                available_models_msg = (
                    ", ".join(available_models) if available_models else "none"
                )
                error_msg = (
                    f"Model '{model_name}' is not deployed. "
                    f"Available deployments: {available_models_msg}. "
                    f"Deploy the requested model or choose one of the available models."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not isinstance(embedding_input, list):
                embedding_input = [embedding_input]

            # Validate all items for the specified modality in parallel
            try:
                validated_items = await asyncio.gather(
                    *[
                        validate_item_for_modality(
                            item, modality, idx, client=self.http_client
                        )
                        for idx, item in enumerate(embedding_input)
                    ]
                )
            except (ValueError, NotImplementedError) as e:
                logger.error(f"Validation failed for modality '{modality}': {e}")
                raise

            logger.info(
                f"Processing {len(validated_items)} {modality} items for model '{model_name}'"
            )

            # Route to appropriate embedding method based on modality
            try:
                if modality == "text":
                    logger.debug(
                        f"Calling .embed() with {len(validated_items)} text items"
                    )
                    embeddings, usage = await self.engine_array[model_name].embed(
                        validated_items
                    )
                    logger.debug(
                        f"Successfully got {len(embeddings)} text embeddings"
                    )

                elif modality == "image":
                    logger.debug(
                        f"Calling .image_embed() with {len(validated_items)} image items"
                    )
                    embeddings, usage = await self.engine_array[model_name].image_embed(
                        images=validated_items
                    )
                    logger.debug(
                        f"Successfully got {len(embeddings)} image embeddings"
                    )

                elif modality == "audio":
                    raise NotImplementedError(
                        "Audio modality is not yet implemented. "
                        "Currently supported modalities: 'text', 'image'"
                    )

                else:
                    raise ValueError(
                        f"Invalid modality: '{modality}'. "
                        f"Supported modalities: 'text', 'image', 'audio' (not yet implemented)"
                    )
            except ModelNotDeployedError as e:
                available_capabilities = sorted(
                    getattr(self.engine_array[model_name], "capabilities", set())
                )
                capabilities_hint = (
                    ", ".join(available_capabilities) if available_capabilities else "none"
                )
                error_msg = (
                    f"Model '{model_name}' does not expose the '{modality}' capability. "
                    f"Detected capabilities: {capabilities_hint}. Deploy a model that supports "
                    f"this modality or adjust the request."
                )
                logger.error(f"{error_msg} Original error: {e}")
                raise ValueError(error_msg) from e

            if return_as_list:
                return [
                    list_embeddings_to_response(
                        embeddings,
                        model=model_name,
                        usage=usage,  # type: ignore[arg-type]
                    )
                ]
            else:
                return list_embeddings_to_response(
                    embeddings,
                    model=model_name,
                    usage=usage,  # type: ignore[arg-type]
                )

        except (ValueError, NotImplementedError) as e:
            logger.warning(
                f"Expected error in route_openai_get_embeddings: {type(e).__name__}: {e}"
            )
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error in route_openai_get_embeddings: "
                f"model='{model_name}', modality='{modality}', "
                f"input_length={len(embedding_input) if isinstance(embedding_input, list) else 1}"
            )
            raise RuntimeError(
                f"Internal error while processing embeddings: {type(e).__name__}: {str(e)}"
            ) from e

    async def infinity_rerank(
        self, query: str, docs: str, return_docs: str, model_name: str
    ):
        """Rerank the documents based on the query"""
        if not self.is_running:
            await self.start()
        scores, usage = await self.engine_array[model_name].rerank(
            query=query, docs=docs, raw_scores=False
        )
        if not return_docs:
            docs = None
        return to_rerank_response(
            scores=scores, documents=docs, model=model_name, usage=usage
        )
