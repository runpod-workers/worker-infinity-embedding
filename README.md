> [!WARNING]  
> This is a work in progress and is not yet ready for use in production.


# Infinity Text Embedding and ReRanker Worker (OpenAI Compatible)
Based on [Infinity Text Embedding Engine](https://github.com/michaelfeil/infinity)

## Docker Image
You can directly use the following docker images and configure them via Environment Variables.
* CUDA 11.8: `alpayariyakrunpod/infinity-text-embedding-worker:mvp-cuda11.8`
* CUDA 12.1: `alpayariyakrunpod/infinity-text-embedding-worker:mvp-cuda12.1`

## RunPod Template Environment Variables
* `MODEL_NAMES`: HuggingFace repo of a single model or multiple models separated by semicolon.      
    * Example - Single Model: `BAAI/bge-small-en-v1.5;`
    * Example - Multiple Models: `BAAI/bge-small-en-v1.5;intfloat/e5-large-v2;`
* `BATCH_SIZES`: Batch size for each model separated by semicolon. If not provided, default batch size of 32 will be used. 
* `BACKEND`: Backend for all models. If not provided, default backend of `torch` will be used.

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
python src/handler.py --test_input '{"input": {"query": "Where is paris?", "docs": ["Paris is in France", "Rome is in Italy"], "model": "mixedbread-ai/mxbai-rerank-xsmall-v1"}}'
```
    

<!-- <div align="center">

<h1>Template | Worker</h1>

[![CI | Test Handler](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_handler.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_handler.yml)
&nbsp;
[![CD | Build-Test-Release](https://github.com/runpod-workers/worker-template/actions/workflows/build-test-release.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/build-test-release.yml)

🚀 | A simple worker that can be used as a starting point to build your own custom RunPod Endpoint API worker.
</div>

## 📖 | Getting Started

1. Clone this repository.
2. (Optional) Add DockerHub credentials to GitHub Secrets.
3. Add your code to the `src` directory.
4. Update the `handler.py` file to load models and process requests.
5. Add any dependencies to the `requirements.txt` file.
6. Add any other build time scripts to the`builder` directory, for example, downloading models.
7. Update the `Dockerfile` to include any additional dependencies.

### ⚙️ | CI/CD (GitHub Actions)

As a reference this repository provides example CI/CD workflows to help you test your worker and build a docker image. The three main workflows are:

1. `CI-test_handler.yml` - Tests the handler using the input provided by the `--test_input` argument when calling the file containing your handler.

### Test Handler

This workflow will validate that your handler works as expected. You may need to add some dependency installations to the `CI-test_handler.yml` file to ensure your handler can be tested.

The action expects the following arguments to be available:

- `vars.RUNNER_24GB` | The endpoint ID on RunPod for a 24GB runner.
- `secrets.RUNPOD_API_KEY` | Your RunPod API key.
- `secrets.GH_PAT` | Your GitHub Personal Access Token.
- `vars.GH_ORG` | The GitHub organization that owns the repository, this is where the runner will be added to.

### Test End-to-End

This repository is setup to automatically build and push a docker image to the GitHub Container Registry. You will need to add the following to the GitHub Secrets for this repository to enable this functionality:

- `DOCKERHUB_USERNAME` | Your DockerHub username for logging in.
- `DOCKERHUB_TOKEN` | Your DockerHub token for logging in.

Additionally, the following need to be added as GitHub actions variables:

- `DOCKERHUB_REPO` | The name of the repository you want to push to.
- `DOCKERHUB_IMG` | The name of the image you want to push to.

The `CD-docker_dev.yml` file will build the image and push it to the `dev` tag, while the `CD-docker_release.yml` file will build the image on releases and tag it with the release version.

The `CI-test_worker.yml` file will test the worker using the input provided by the `--test_input` argument when calling the file containing your handler. Be sure to update this workflow to install any dependencies you need to run your tests.

## Example Input

```json
{
    "input": {
        "name": "John Doe"
    }
}
```

## 💡 | Best Practices

System dependency installation, model caching, and other shell tasks should be added to the `builder/setup.sh` this will allow you to easily setup your Dockerfile as well as run CI/CD tasks.

Models should be part of your docker image, this can be accomplished by either copying them into the image or downloading them during the build process.

If using the input validation utility from the runpod python package, create a `schemas` python file where you can define the schemas, then import that file into your `handler.py` file.

## 🔗 | Links

🐳 [Docker Container](https://hub.docker.com/r/runpod/serverless-hello-world) -->
