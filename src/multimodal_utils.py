import base64
import io
import logging
import re
from typing import Any
from PIL import Image

logger = logging.getLogger(__name__)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """
    Ensure image is in RGB format (required for CLIP models).
    
    Args:
        img: PIL Image in any mode
        
    Returns:
        PIL Image in RGB mode
    """
    if img.mode != 'RGB':
        logger.debug(f"Converting image from {img.mode} to RGB")
        return img.convert('RGB')
    return img


def _is_url(text: str) -> bool:
    """Check if string is a URL"""
    return text.startswith('http://') or text.startswith('https://')


async def _download_image_from_url(url: str, client) -> Image.Image:
    """
    Download image from URL using httpx with proper User-Agent and timeout.
    
    Args:
        url: HTTP(S) URL to download image from
        client: httpx.AsyncClient instance with configured timeout and limits
    
    Returns:
        PIL.Image in RGB format
        
    Raises:
        ValueError: If download fails or content is not a valid image
    """
    try:
        logger.debug(f"Downloading image from URL: {url}")
        response = await client.get(url)
        response.raise_for_status()
        
        img_bytes = response.content
        logger.debug(f"Downloaded {len(img_bytes)} bytes from {url} (status: {response.status_code})")
    except Exception as e:
        raise ValueError(f"Failed to download image from URL: {type(e).__name__}: {e}") from e
    
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.load()  # Force load to validate it's a real image
        logger.debug(f"Successfully loaded image from URL: {img.size} {img.mode}")
        
        return _ensure_rgb(img)
    except Exception as e:
        raise ValueError(f"Failed to decode image from URL: {type(e).__name__}: {e}") from e


def _is_base64_image(data: str) -> Image.Image | None:
    """
    Try to decode string as base64-encoded image.
    Supports both data URI format (data:image/...) and raw base64.
    
    Returns:
        PIL.Image in RGB format, or None if not a valid base64 image
    
    Note: Converts all images to RGB format for compatibility with multimodal models.
    """
    try:
        # Handle data URI format: data:image/png;base64,iVBORw0KG...
        if data.startswith('data:'):
            match = re.match(r'data:image/[^;]+;base64,(.+)', data)
            if match:
                base64_data = match.group(1)
                logger.debug(f"Matched data URI, extracted base64 data (length: {len(base64_data)})")
            else:
                logger.debug("data: URI does not match expected format")
                return None
        else:
            # Try raw base64
            base64_data = data
            logger.debug("Treating as raw base64")
        
        img_bytes = base64.b64decode(base64_data)
        logger.debug(f"Decoded base64 to {len(img_bytes)} bytes")
        
        img = Image.open(io.BytesIO(img_bytes))
        img.load()  # Force load to validate it's a real image
        logger.debug(f"Successfully loaded image: {img.size} {img.mode}")
        
        return _ensure_rgb(img)
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {type(e).__name__}: {e}")
        return None


def validate_text_item(item: Any) -> str:
    """
    Validate and convert item to text string.
    Raises ValueError if item cannot be converted to text.
    """
    if isinstance(item, str):
        return item
    
    # Try to convert to string
    try:
        return str(item)
    except Exception as e:
        raise ValueError(f"Cannot convert item to text: {type(item).__name__}") from e


async def validate_image_item(item: Any, client=None) -> Image.Image:
    """
    Validate and process image item.
    Accepts: PIL.Image, bytes, URL string, or base64 string.
    Returns PIL.Image in RGB format.
    Raises ValueError if item is not a valid image format.
    
    Args:
        item: The image item to validate (PIL.Image, bytes, URL string, or base64 string)
        client: Optional httpx.AsyncClient for downloading URL images with timeout and User-Agent
    
    Note: All images are converted to RGB format for CLIP compatibility.
    """
    # PIL Image - convert to RGB if needed
    if isinstance(item, Image.Image):
        return _ensure_rgb(item)
    
    # Bytes - decode to PIL Image
    if isinstance(item, bytes):
        try:
            img = Image.open(io.BytesIO(item))
            img.load()
            logger.debug(f"Loaded image from bytes: {img.size} {img.mode}")
            return _ensure_rgb(img)
        except Exception as e:
            raise ValueError(f"Failed to decode image from bytes: {type(e).__name__}: {e}") from e
    
    if isinstance(item, str):
        # URL images - download with proper User-Agent and timeout
        if _is_url(item):
            if client is None:
                raise ValueError("HTTP client required for downloading images from URLs")
            return await _download_image_from_url(item, client)
        
        # Base64 images - decode and validate (already converted to RGB)
        img = _is_base64_image(item)
        if img is not None:
            return img
        
        # Not a valid image format
        raise ValueError(
            "String is not a valid image format (must be URL starting with http:// or https://, "
            "or base64-encoded image with 'data:image/...' prefix)"
        )
    
    raise ValueError(
        f"Invalid image type: {type(item).__name__}. "
        f"Expected PIL.Image, bytes, URL string, or base64 string."
    )


async def validate_item_for_modality(item: Any, modality: str, index: int, client=None) -> Any:
    """
    Validate a single item for the specified modality.
    
    Args:
        item: The input item to validate
        modality: One of "text", "image", "audio"
        index: The index of the item in the batch (for error messages)
        client: Optional httpx.AsyncClient for downloading URL images with timeout
    
    Returns:
        Validated and processed item suitable for infinity_emb
    
    Raises:
        ValueError: If item is not valid for the specified modality
        NotImplementedError: If modality is "audio"
    """
    try:
        if modality == "text":
            return validate_text_item(item)
        elif modality == "image":
            return await validate_image_item(item, client=client)
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
    except (ValueError, NotImplementedError) as e:
        # Re-raise with index information for better error messages
        raise type(e)(f"Item at index {index}: {str(e)}") from e
