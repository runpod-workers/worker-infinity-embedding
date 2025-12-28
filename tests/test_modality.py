"""Integration tests for the explicit modality API using pytest."""

import base64
import io
from collections.abc import Iterable, Mapping
from typing import Any

import pytest
import requests
from PIL import Image

BASE_URL = "http://localhost:8000"
RUNSYNC_URL = f"{BASE_URL}/runsync"
DEFAULT_TIMEOUT_SECONDS = 30


def _post_runsync(
    payload: Mapping[str, Any], timeout: int | float = DEFAULT_TIMEOUT_SECONDS
) -> requests.Response:
    """Send a request to the worker and print useful diagnostics."""
    response = requests.post(RUNSYNC_URL, json=payload, timeout=timeout)
    print(f"POST {RUNSYNC_URL} -> {response.status_code}")
    return response


def _extract_output(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Normalise RunPod /runsync output into a single mapping."""
    output = result.get("output")
    if isinstance(output, list) and output:
        return output[0]
    if isinstance(output, Mapping):
        return output
    return None


def _extract_error_message(result: Mapping[str, Any]) -> str | None:
    output = _extract_output(result)
    if output and output.get("object") == "error":
        message = output.get("message")
        return str(message) if message is not None else None
    return None


def generate_red_square_base64() -> str:
    """Generate a 5x5 red square PNG encoded as a data URI."""
    img = Image.new("RGB", (5, 5), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


@pytest.mark.parametrize(
    "name,payload,expected_count",
    [
        (
            "text modality",
            {
                "input": {
                    "openai_route": "/v1/embeddings",
                    "openai_input": {
                        "model": "BAAI/bge-small-en-v1.5",
                        "input": ["Hello world", "How are you?"],
                        "modality": "text",
                    },
                }
            },
            2,
        ),
        (
            "image modality (url)",
            {
                "input": {
                    "openai_route": "/v1/embeddings",
                    "openai_input": {
                        "model": "patrickjohncyh/fashion-clip",
                        "input": [
                            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
                        ],
                        "modality": "image",
                    },
                }
            },
            1,
        ),
        (
            "image modality (base64)",
            {
                "input": {
                    "openai_route": "/v1/embeddings",
                    "openai_input": {
                        "model": "patrickjohncyh/fashion-clip",
                        "input": [generate_red_square_base64()],
                        "modality": "image",
                    },
                }
            },
            1,
        ),
        (
            "default text modality",
            {
                "input": {
                    "openai_route": "/v1/embeddings",
                    "openai_input": {
                        "model": "BAAI/bge-small-en-v1.5",
                        "input": ["Default modality test"],
                    },
                }
            },
            1,
        ),
    ],
)
def test_modality_success(
    name: str, payload: Mapping[str, Any], expected_count: int
) -> None:
    response = _post_runsync(payload)
    assert response.status_code == 200, f"HTTP error for {name}: {response.text}"

    result = response.json()
    assert result.get("status") == "COMPLETED", (
        f"Unexpected status for {name}: {result}"
    )

    output = _extract_output(result) or {}
    data = output.get("data", [])
    assert isinstance(data, Iterable), f"Missing data for {name}: {output}"

    if isinstance(data, list):
        assert len(data) == expected_count, (
            f"Expected {expected_count} embeddings for {name}, got {len(data)}"
        )
    else:
        pytest.fail(f"Unexpected data format for {name}: {type(data)}")


def test_wrong_modality_error() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "BAAI/bge-small-en-v1.5",
                "input": [
                    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
                ],
                "modality": "image",
            },
        }
    }

    response = _post_runsync(payload)
    if response.status_code != 200:
        assert response.status_code >= 400
        return

    result = response.json()
    error_msg = _extract_error_message(result)
    assert error_msg is not None, f"Expected error object, got: {result}"
    lowered = error_msg.lower()
    assert "does not expose the 'image' capability" in lowered
    assert "detected capabilities" in lowered


def test_audio_not_implemented() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "BAAI/bge-small-en-v1.5",
                "input": ["audio data"],
                "modality": "audio",
            },
        }
    }

    response = _post_runsync(payload)
    if response.status_code != 200:
        assert response.status_code >= 400
        return

    result = response.json()
    error_msg = _extract_error_message(result)
    assert error_msg is not None, f"Expected NotImplementedError output, got: {result}"
    assert "not yet implemented" in error_msg.lower()


def test_validation_flexibility() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "BAAI/bge-small-en-v1.5",
                "input": ["https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"],
                "modality": "text",
            },
        }
    }

    response = _post_runsync(payload)
    assert response.status_code == 200, f"HTTP error: {response.text}"

    result = response.json()
    assert result.get("status") == "COMPLETED", (
        f"Expected success treating URL as text: {result}"
    )


@pytest.mark.parametrize(
    "payload,expected_count",
    [
        (
            {
                "input": {
                    "openai_route": "/v1/embeddings",
                    "openai_input": {
                        "model": "BAAI/bge-small-en-v1.5",
                        "input": [],
                        "modality": "text",
                    },
                }
            },
            0,
        ),
        (
            {
                "input": {
                    "openai_route": "/v1/embeddings",
                    "openai_input": {
                        "model": "BAAI/bge-small-en-v1.5",
                        "input": "Single text string",
                        "modality": "text",
                    },
                }
            },
            1,
        ),
    ],
)
def test_text_edge_cases(payload: Mapping[str, Any], expected_count: int) -> None:
    response = _post_runsync(payload)
    if response.status_code != 200:
        assert response.status_code >= 400
        return

    result = response.json()
    if result.get("status") != "COMPLETED":
        assert _extract_error_message(result) is not None
        return

    output = _extract_output(result) or {}
    data = output.get("data", [])
    assert isinstance(data, list)
    assert len(data) == expected_count


def test_edge_very_long_text() -> None:
    long_text = "This is a test sentence. " * 200
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "BAAI/bge-small-en-v1.5",
                "input": long_text,
                "modality": "text",
            },
        }
    }

    response = _post_runsync(payload)
    assert response.status_code == 200, f"HTTP error: {response.text}"

    result = response.json()
    assert result.get("status") == "COMPLETED", f"Unexpected status: {result}"


def test_edge_empty_string() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "BAAI/bge-small-en-v1.5",
                "input": "",
                "modality": "text",
            },
        }
    }

    response = _post_runsync(payload)
    if response.status_code != 200:
        assert response.status_code >= 400
        return

    result = response.json()
    if result.get("status") == "COMPLETED":
        output = _extract_output(result) or {}
        data = output.get("data", [])
        assert isinstance(data, list)
    else:
        assert _extract_error_message(result) is not None


def test_edge_invalid_modality() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "BAAI/bge-small-en-v1.5",
                "input": "test",
                "modality": "video",
            },
        }
    }

    response = _post_runsync(payload)
    if response.status_code != 200:
        assert response.status_code >= 400
        return

    result = response.json()
    error_msg = _extract_error_message(result)
    assert error_msg is not None, f"Expected invalid modality error: {result}"
    assert "invalid modality" in error_msg.lower()


def test_edge_missing_model() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "input": "test",
                "modality": "text",
            },
        }
    }

    response = _post_runsync(payload)
    if response.status_code != 200:
        assert response.status_code >= 400
        return

    result = response.json()
    error_msg = _extract_error_message(result)
    assert error_msg is not None, f"Expected missing model error: {result}"


def test_edge_nonexistent_model() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "nonexistent/model-12345",
                "input": "test",
                "modality": "text",
            },
        }
    }

    response = _post_runsync(payload)
    if response.status_code != 200:
        assert response.status_code >= 400
        return

    result = response.json()
    error_msg = _extract_error_message(result)
    assert error_msg is not None, f"Expected model missing error: {result}"
    lowered = error_msg.lower()
    assert "is not deployed" in lowered
    assert "available deployments" in lowered


def test_edge_invalid_image_url() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "patrickjohncyh/fashion-clip",
                "input": "https://example.com/nonexistent-image-12345.jpg",
                "modality": "image",
            },
        }
    }

    response = _post_runsync(payload)
    if response.status_code != 200:
        assert response.status_code >= 400
        return

    result = response.json()
    assert result.get("status") in {"COMPLETED", "FAILED"}, (
        f"Unexpected status: {result}"
    )


def test_edge_special_characters() -> None:
    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "model": "BAAI/bge-small-en-v1.5",
                "input": "Hello 世界! 🌍 Special chars: @#$%^&*()_+-=[]{}|;:',.<>?/~`",
                "modality": "text",
            },
        }
    }

    response = _post_runsync(payload)
    assert response.status_code == 200, f"HTTP error: {response.text}"

    result = response.json()
    assert result.get("status") == "COMPLETED", f"Unexpected status: {result}"


def test_edge_corrupted_base64() -> None:
    invalid_payloads = [
        "data:image/png;base64,NotValidBase64Data!!!",
        "data:image/png;base64,iVBORw0KGgoAAAA",
        "data:image/jpeg;base64,/9j/4AAQSkZJRg",
        "data:image/png;base64,SGVsbG8gV29ybGQh",
    ]

    for idx, invalid_base64 in enumerate(invalid_payloads, start=1):
        print(
            f"\n  Test variant {idx}/{len(invalid_payloads)}: {invalid_base64[:50]}..."
        )
        payload = {
            "input": {
                "openai_route": "/v1/embeddings",
                "openai_input": {
                    "model": "patrickjohncyh/fashion-clip",
                    "input": invalid_base64,
                    "modality": "image",
                },
            }
        }

        response = _post_runsync(payload)
        if response.status_code != 200:
            assert response.status_code >= 400
            continue

        result = response.json()
        error_msg = _extract_error_message(result)
        assert error_msg is not None, f"Expected error for corrupted base64: {result}"

    print("\n✓ All corrupted base64 variants handled without crash")
