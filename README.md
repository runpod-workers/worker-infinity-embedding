![Infinity Embedding Worker Banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-infinity-embedding_banner-9n86vTARpwknMZYnXHAUr7xJisiWXs.jpeg)

---

High-throughput, OpenAI-compatible text embedding & reranker powered by [Infinity](https://github.com/michaelfeil/infinity)

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-infinity-embedding)](https://www.runpod.io/console/hub/runpod-workers/worker-infinity-embedding)

---

1. [Quickstart](#quickstart)
2. [Endpoint Configuration](#endpoint-configuration)
3. [API Specification](#api-specification)
   1. [List Models](#list-models)
   2. [Create Embeddings](#create-embeddings)
   3. [Rerank Documents](#rerank-documents)
4. [Usage](#usage)
5. [Further Documentation](#further-documentation)
6. [Acknowledgements](#acknowledgements)

---

## Quickstart

1. 🐳 **Pull an image** – use the tag shown on the latest [GitHub release page](https://github.com/runpod-workers/worker-infinity-embedding/releases) (e.g. `runpod/worker-infinity-embedding:<version>`)
2. 🔧 **Configure** – set at least `MODEL_NAMES` (see [Endpoint Configuration](#endpoint-configuration))
3. 🚀 **Deploy** – create a [RunPod Serverless endpoint](https://docs.runpod.io/serverless/endpoints/manage-endpoints)
4. 🧪 **Call the API** – follow the example in the [Usage](#usage) section

---

## Endpoint Configuration

All behaviour is controlled through environment variables:

| Variable                 | Required | Default | Description                                                                                                      |
| ------------------------ | -------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `MODEL_NAMES`            | **Yes**  | —       | One or more Hugging-Face model IDs. Separate multiple IDs with a semicolon.<br>Example: `BAAI/bge-small-en-v1.5` |
| `BATCH_SIZES`            | No       | `32`    | Per-model batch size; semicolon-separated list matching `MODEL_NAMES`.                                           |
| `BACKEND`                | No       | `torch` | Inference engine for _all_ models: `torch`, `optimum`, or `ctranslate2`.                                         |
| `DTYPES`                 | No       | `auto`  | Precision per model (`auto`, `fp16`, `fp8`). Semicolon-separated, must match `MODEL_NAMES`.                      |
| `INFINITY_QUEUE_SIZE`    | No       | `48000` | Max items queueable inside the Infinity engine.                                                                  |
| `RUNPOD_MAX_CONCURRENCY` | No       | `300`   | Max concurrent requests the RunPod wrapper will accept.                                                          |

---

## API Specification

Two flavours, one schema.

- **OpenAI-compatible** – drop-in replacement for `/v1/models`, `/v1/embeddings`, so you can use this endpoint instead of the API from OpenAI by replacing the base url with the URL of your endpoint: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1` and use your [API key from RunPod](https://docs.runpod.io/get-started/api-keys) instead of the one from OpenAI
- **Standard RunPod** – call `/run` or `/runsync` with a JSON body under the `input` key.  
  Base URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>`

Except for transport (path + wrapper object) the JSON you send/receive is identical. The tables below describe the shared payload.

### List Models

| Method | Path                | Body                                            |
| ------ | ------------------- | ----------------------------------------------- |
| `GET`  | `/openai/v1/models` | –                                               |
| `POST` | `/runsync`          | `{ "input": { "openai_route": "/v1/models" } }` |

#### Response

```jsonc
{
  "data": [
    { "id": "BAAI/bge-small-en-v1.5", "stats": {} },
    { "id": "intfloat/e5-large-v2", "stats": {} }
  ]
}
```

---

### Create Embeddings

#### Request Fields (shared)

| Field   | Type                | Required | Description                                       |
| ------- | ------------------- | -------- | ------------------------------------------------- |
| `model` | string              | **Yes**  | One of the IDs supplied via `MODEL_NAMES`.        |
| `input` | string &#124; array | **Yes**  | A single text string _or_ list of texts to embed. |

OpenAI route vs. Standard:

| Flavour  | Method | Path             | Body                                          |
| -------- | ------ | ---------------- | --------------------------------------------- |
| OpenAI   | `POST` | `/v1/embeddings` | `{ "model": "…", "input": "…" }`              |
| Standard | `POST` | `/runsync`       | `{ "input": { "model": "…", "input": "…" } }` |

#### Response (both flavours)

```jsonc
{
  "object": "list",
  "model": "BAAI/bge-small-en-v1.5",
  "data": [
    { "object": "embedding", "embedding": [0.01, -0.02 /* … */], "index": 0 }
  ],
  "usage": { "prompt_tokens": 2, "total_tokens": 2 }
}
```

---

### Rerank Documents (Standard only)

| Field         | Type   | Required | Description                                                       |
| ------------- | ------ | -------- | ----------------------------------------------------------------- |
| `model`       | string | **Yes**  | Any deployed reranker model                                       |
| `query`       | string | **Yes**  | The search/query text                                             |
| `docs`        | array  | **Yes**  | List of documents to rerank                                       |
| `return_docs` | bool   | No       | If `true`, return the documents in ranked order (default `false`) |

Call pattern

```http
POST /runsync
Content-Type: application/json

{
  "input": {
    "model": "BAAI/bge-reranker-large",
    "query": "Which product has warranty coverage?",
    "docs": [
      "Product A comes with a 2-year warranty",
      "Product B is available in red and blue colors",
      "All electronics include a standard 1-year warranty"
    ],
    "return_docs": true
  }
}
```

Response contains either `scores` or the full `docs` list, depending on `return_docs`.

---

## Usage

Below are minimal `curl` snippets so you can copy-paste from any machine.

> Replace `<ENDPOINT_ID>` with your endpoint ID and `<API_KEY>` with a [RunPod API key](https://docs.runpod.io/get-started/api-keys).

### OpenAI-Compatible Calls

```bash
# List models
curl -H "Authorization: Bearer <API_KEY>" \
     https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/models

# Create embeddings
curl -X POST \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-small-en-v1.5","input":"Hello world"}' \
  https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/embeddings
```

### Standard RunPod Calls

```bash
# Create embeddings (wait for result)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"input":{"model":"BAAI/bge-small-en-v1.5","input":"Hello world"}}' \
  https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync

# Rerank
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"input":{"model":"BAAI/bge-reranker-large","query":"Which product has warranty coverage?","docs":["Product A comes with a 2-year warranty","Product B is available in red and blue colors","All electronics include a standard 1-year warranty"],"return_docs":true}}' \
  https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
```

---

## Further Documentation

- **[Infinity Engine](https://github.com/michaelfeil/infinity)** – how the ultra-fast backend works.
- **[RunPod Docs](https://docs.runpod.io/)** – serverless concepts, limits, and API reference.

---

## Acknowledgements

Special thanks to [Michael Feil](https://github.com/michaelfeil) for creating the Infinity engine and for his ongoing support of this project.
