from config import EmbeddingServiceConfig
from infinity_emb import EngineArgs, AsyncEmbeddingEngine
from utils import create_error_response, OpenAIEmbeddingResult, OpenAIModelInfo, ModelInfo, list_embeddings_to_response, to_rerank_response
class EmbeddingService:
    def __init__(self):
        self.config = EmbeddingServiceConfig()
        self.models = self._initialize_models()
            
    def _initialize_models(self):
        models = {}
        for model_name, batch_size in zip(self.config.model_names, self.config.batch_sizes):
            models[model_name] = EmbeddingModel(model_name, batch_size, self.config.backend)
        return models
    
    async def _embed(self, input, engine):
        if not isinstance(input, list):
            input = [input]
        async with engine:
            embeddings, usage = await engine.embed(input)
        return embeddings, usage
    
    def openai_get_models(self):
        return OpenAIModelInfo(data=[
            ModelInfo(id=model.engine_args.model_name_or_path,
                      stats=dict(batch_size=model.engine_args.batch_size,
                        	backend=model.engine_args.engine.name))
                    for model in self.models.values()]).model_dump()
        
    async def openai_get_embeddings(self, embedding_input, model):
        embeddings, usage = await self._embed(embedding_input, model.engine)
        result = list_embeddings_to_response(embeddings, model.engine_args.model_name_or_path, usage)
        return OpenAIEmbeddingResult(**result).model_dump()
    
    async def infinity_embed(self, embedding_input, model):
        embeddings, usage = await self._embed(embedding_input, model.engine)
        return list_embeddings_to_response(embeddings, model.engine_args.model_name_or_path, usage)
    
    async def infinity_rerank(self, input, model):
        query, docs, return_docs = input["query"], input["docs"], input["return_docs"]
        async with model.engine:
            scores, usage = await model.engine.rerank(query=query, docs=docs, raw_scores=False)
        if not return_docs:
            docs = None 
        return to_rerank_response(scores=scores, documents=docs, model=model.engine_args.model_name_or_path, usage=usage)
        
        
        
class EmbeddingModel:
    def __init__(self, model_name, batch_size, backend):
        print(f"Initializing model {model_name} with batch size {batch_size} and backend {backend}")
        self.model_name = model_name
        self.batch_size = batch_size
        self.engine_args = EngineArgs(model_name_or_path=model_name, batch_size=batch_size, engine=backend)
        self.engine = AsyncEmbeddingEngine.from_args(self.engine_args)