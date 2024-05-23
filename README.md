> [!WARNING]  
> This is a work in progress and is not yet ready for use in production.


# Infinity Text Embedding and ReRanker Worker (OpenAI Compatible)
Based on [Infinity Text Embedding Engine](https://github.com/michaelfeil/infinity)

## Docker Image
You can directly use the following docker images and configure them via Environment Variables.
* CUDA 11.8: `not built`
* CUDA 12.1: `michaelf34/runpod-infinity-worker:0.0.5-cu121`

## RunPod Template Environment Variables
* `MODEL_NAMES`: HuggingFace repo of a single model or multiple models separated by semicolon.      
    * Example - Single Model: `BAAI/bge-small-en-v1.5;`
    * Example - Multiple Models: `BAAI/bge-small-en-v1.5;intfloat/e5-large-v2;`
* `BATCH_SIZES`: Batch size for each model separated by semicolon. If not provided, default batch size of 32 will be used. 
* `BACKEND`: Backend for all models. Recommended is `torch` which is the default. Other options are `optimum` or `ctranslate2`.
* `DTYPES`: Dtype, by default `auto` or `fp16`.

## Supported Models
<details>
  <summary>What models are supported?</summary>
  
  - All models supported by the sentence-transformers library.
  - All models reuploaded on the sentence transformers org https://huggingface.co/sentence-transformers / sbert.net. 

  With the command `--engine torch` the model must be compatible with sentence-transformers library
  
  For the latest trends, you might want to check out one of the following models.
    https://huggingface.co/spaces/mteb/leaderboard
    
</details>

## Usage - OpenAI Compatibility
### Set up
Initialize OpenAI client and set the API Key to your RunPod API Key, and base URL to `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1`
```python
from openai import OpenAI

client = OpenAI(
  api_key=RUNPOD_API_KEY, 
  base_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1"
)
```
### Embedding
1. Define the input

    You may embed a single text or a list of texts
    - Single Text
        ```python
        embedding_input = "Hello, world!"
        ```
    - List of Texts
        ```python
        embedding_input = ["Hello, world!", "This is a test."]
        ```
2. Get the embeddings
    ```python
    client.embeddings.create(
        model="YOUR_DEPLOYED_MODEL_NAME",
        input=embedding_input
    )
    ```
    Where `YOUR_DEPLOYED_MODEL_NAME` is the name of one of the models you deployed to the worker.

## Usage - Standard
### Set up
You may use /run or /runsync

### Embedding
Inputs:
* `model`: name of one of the deployed models.
* `input`: single text string or list of texts to embed

### Reranking
Inputs:
* `model`: name of one of the deployed models
* `query`: query text (single string)
* `docs`: list of documents to rerank by query
* `return_docs`: whether to return the reranked documents or not


### Additional testing

For the Reranker models 
```bash
python src/handler.py --test_input '{"input": {"query": "Where is paris?", "docs": ["Paris is in France", "Rome is in Italy"], "model": "BAAI/bge-reranker-v2-m3"}}'
```