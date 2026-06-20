"""LLM/VLM interface module for the damage claim evaluation system.

All Azure OpenAI calls are centralised here. Callers should use
`evaluate_claim()` and choose the strategy via the `use_cot` flag.
"""

import base64
import io
import os
import time
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from openai import AzureOpenAI
from opentelemetry import trace
from PIL import Image
import pillow_heif

# Register HEIF opener so PIL can natively read HEIC images disguised as .jpg
pillow_heif.register_heif_opener()

from .schema import (
    get_direct_schema,
    get_cot_schema,
)
from .prompts import (
    DIRECT_SYSTEM_PROMPT,
    COT_SYSTEM_PROMPT,
    build_direct_user_message,
    build_cot_user_message,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Azure OpenAI client — reads all config from environment
# ---------------------------------------------------------------------------

_client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get(
        "AZURE_OPENAI_API_VERSION", "2025-04-01-preview"
    ),
)

_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
)

_tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_image_b64(image_path: str | Path) -> str:
    """Encode a local image file as a base64 data URI.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Base64-encoded image string (JPEG MIME type).

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path.resolve()}"
        )
    
    # Use PIL to read the image (automatically handling HEIC/WEBP/PNG)
    # and re-encode it strictly as JPEG into a memory buffer.
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_image_content_blocks(
    image_paths: list[str],
    repo_root: Path,
) -> list[dict]:
    """Build the list of image content blocks for a multi-modal message.

    Args:
        image_paths: Paths relative to the repo root (from CSV column).
        repo_root: Absolute path to the repository root.

    Returns:
        List of dicts in OpenAI vision content-block format.
    """
    blocks: list[dict] = []
    for rel_path in image_paths:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            abs_path = repo_root / "dataset" / rel_path
        b64 = encode_image_b64(abs_path)
        blocks.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
                "detail": "auto",
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_claim(
    *,
    claim_object: str,
    user_claim: str,
    evidence_requirement: str,
    image_paths: list[str],
    image_ids: list[str],
    repo_root: Path,
    use_cot: bool = True,
    session_id: str = "default",
) -> dict:
    """Run VLM evaluation for a single damage claim.

    Args:
        claim_object: 'car', 'laptop', or 'package'.
        user_claim: Customer chat transcript.
        evidence_requirement: Minimum evidence requirement text.
        image_paths: Image paths relative to repo root.
        image_ids: Ordered image IDs matching image_paths.
        repo_root: Absolute path to the project root directory.
        use_cot: If True use Strategy B (CoT), else Strategy A (Direct).
        session_id: Session identifier for Arize Phoenix tracing.

    Returns:
        Dict with all structured output fields plus operational metadata:
        ``input_tokens``, ``output_tokens``, ``latency_seconds``.
    """
    strategy = "B_CoT" if use_cot else "A_Direct"
    system_prompt = COT_SYSTEM_PROMPT if use_cot else DIRECT_SYSTEM_PROMPT
    schema = (
        get_cot_schema(claim_object)
        if use_cot
        else get_direct_schema(claim_object)
    )
    build_user_msg = (
        build_cot_user_message if use_cot else build_direct_user_message
    )

    user_text = build_user_msg(
        claim_object=claim_object,
        user_claim=user_claim,
        evidence_requirement=evidence_requirement,
        image_ids=image_ids,
    )

    image_blocks = _build_image_content_blocks(image_paths, repo_root)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}]
            + image_blocks,
        },
    ]

    span_name = "openai.responses.create"
    with _tracer.start_as_current_span(span_name) as span:
        span.set_attribute("session.id", session_id)
        span.set_attribute("llm.model", _DEPLOYMENT)
        span.set_attribute("strategy", strategy)
        span.set_attribute(
            "llm.input_messages",
            str(messages)[:2000],
        )

        t0 = time.perf_counter()
        try:
            response = _client.responses.parse(
                model=_DEPLOYMENT,
                input=messages,
                text_format=schema,
            )
        except Exception as exc:
            span.record_exception(exc)
            raise

        latency = time.perf_counter() - t0

        usage = response.usage
        input_tokens: int = getattr(usage, "input_tokens", 0)
        output_tokens: int = getattr(usage, "output_tokens", 0)

        result_obj = response.output_parsed

        span.set_attribute(
            "llm.output_messages", str(result_obj)[:2000]
        )
        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)

    result_dict = result_obj.model_dump()
    result_dict["input_tokens"] = input_tokens
    result_dict["output_tokens"] = output_tokens
    result_dict["latency_seconds"] = round(latency, 3)
    return result_dict
