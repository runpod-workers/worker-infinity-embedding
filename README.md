<div align="center">

# Infinity Embedding Serverless Worker

Deploy almost any Text Embedding and Reranker models with high throughput OpenAI-compatible Endpoints on RunPod Serverless, powered by the fastest embedding inference engine, built for serving - [Infinity](https://github.com/michaelfeil/infinity)


</div>

# Supported Models
When using `torch` backend, you can deploy any models supported by the sentence-transformers library.

This also means that you can deploy any model from the [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), which is currently the most popular and comprehensive leaderboard for embedding models.



# Setting up the Serverless Endpoint
## Option 1: Deploy any models directly from RunPod Console with Pre-Built Docker Image

> [!NOTE]  
> We are adding a UI for deployment similar to [Worker vLLM](https://github.com/runpod-workers/worker-vllm), but for now, you can manually create the endpoint with the regular serverless configurator.


We offer a pre-built Docker Image for the Infinity Embedding Serverless Worker that you can configure entirely with Environment Variables when creating the Endpoint:

### 1. Select Worker Image Version
You can directly use the following docker images and configure them via Environment Variables.
| CUDA Version | Stable (Latest Release)                 | Development (Latest Commit)             | Note                                                        |
|--------------|-----------------------------------|-----------------------------------|----------------------------------------------------------------------|
| 11.8.0       | `runpod/worker-infinity-embedding:stable-cuda11.8.0`        | `runpod/worker-infinity-embedding:dev-cuda11.8.0`   | Available on all RunPod Workers without additional selection needed. |
| 12.1.0       | `runpod/worker-infinity-embedding:stable-cuda12.1.0` | `runpod/worker-infinity-embedding:dev-cuda12.1.0` | When creating an Endpoint, select CUDA Version 12.4, 12.3, 12.2 and 12.1 in the filter. About 10% less total available machines than 11.8.0, but higher performance. |

**[NOTE]** Latest image version (pre) `runpod/worker-infinity-text-embedding:0.0.1-cuda12.1.0`
### 2. Select your models and configure your deployment with Environment Variables
* `MODEL_NAMES`
    
    HuggingFace repo of a single model or multiple models separated by semicolon.      
    
    - Examples:
        - **Single** Model: `BAAI/bge-small-en-v1.5`
        - **Multiple** Models: `BAAI/bge-small-en-v1.5;intfloat/e5-large-v2;`
* `BATCH_SIZES`

    Batch Size for each model separated by semicolon. 

    - Default: `32`
* `BACKEND`

    Backend for all models. 
    
    - Options: 
        - `torch`
        - `optimum`
        - `ctranslate2`
    - Default: `torch`
* `DTYPES`

    Precision for each model separated by semicolon.

    - Options:
        - `auto`
        - `fp16`
        - `fp8` (**New!** Only compatible with H100 and L40S)
    - Default: `auto`

* `INFINITY_QUEUE_SIZE`

    How many requests can be queued in the Infinity Engine. 

    - Default: `48000`

* `RUNPOD_MAX_CONCURRENT_REQUESTS`

    How many requests can be processed concurrently by the RunPod Worker. 

    - Default: `300`

## Option 2: Bake models into Docker Image
Coming soon!

# Usage
There are two ways to use the endpoint - [OpenAI Compatibility](#openai-compatibility) matching how you would use OpenAI API, and [Standard Usage](#standard-usage) with the RunPod API. Note that reranking is only available with [Standard Usage](#standard-usage).
## OpenAI Compatibility
### Set up
1. Install OpenAI Python SDK
```bash
pip install openai
```
2. Initialize OpenAI client and set the API Key to your RunPod API Key, and base URL to `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1`, where `YOUR_ENDPOINT_ID` is the ID of your endpoint, e.g. `elftzf0lld1vw1`
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

## Standard Usage
### Set up
You may use `/run` (asynchronous, start job and return job ID) or `/runsync` (synchronous, wait for job to finish and return result)

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


# Acknowledgements
We'd like to thank [Michael Feil](https://github.com/michaelfeil) for creating the [Infinity Embedding Engine](https://github.com/michaelfeil/infinity) and actively being involved in the development of this worker!

