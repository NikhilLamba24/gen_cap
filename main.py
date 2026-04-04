"""
FastAPI proxy: upload an image and start a Galaxy flux_2_pro (edit) run against
https://app.galaxy.ai/api (see plan.md).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Matches galaxy_ai/plan.md
DEFAULT_BASE_URL = "https://app.galaxy.ai/api"

BASE_URL = os.environ.get("GALAXY_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
API_KEY = os.environ.get("GALAXY_API_KEY", "")
# If set (e.g. https://abc.ngrok.io), uploaded files are served at {PUBLIC_BASE_URL}/uploads/...
# and image_urls use that URL so Galaxy's servers can fetch the image.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
MAX_IMAGE_DOWNLOAD_BYTES = int(os.environ.get("MAX_IMAGE_DOWNLOAD_BYTES", str(25 * 1024 * 1024)))
DEFAULT_MAX_POLL_SECONDS = float(os.environ.get("GALAXY_MAX_POLL_SECONDS", "900"))
# Gemini (prompt_enhancement.md): generativelanguage.googleapis.com
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_GENERATE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Galaxy image upload → flux_2_pro")

from fastapi.responses import HTMLResponse
@app.get("/", response_class=HTMLResponse)
async def root():
    # This reads your index.html file and sends it to the browser
    return Path("index.html").read_text()


def _sniff_image_mime(content: bytes) -> str | None:
    if len(content) < 12:
        return None
    if content[:3] == b"\xff\xd8\xff" or content[:2] == b"\xff\xd8":
        return "image/jpeg"
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(content) >= 6 and content[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def _normalize_upload_mime(content: bytes, content_type: str | None) -> str:
    """Swagger often sends application/octet-stream; sniff magic bytes for Galaxy."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in ("", "application/octet-stream", "binary/octet-stream", "application/x-www-form-urlencoded"):
        return _sniff_image_mime(content) or "image/jpeg"
    if ct.startswith("image/"):
        return (content_type or "image/jpeg").split(";")[0].strip()
    return _sniff_image_mime(content) or "image/jpeg"


def _public_base_url_usable_for_galaxy() -> bool:
    """
    Only use PUBLIC_BASE_URL + /uploads/... when Galaxy's servers can reach that host.
    Localhost URLs fail fetch and can trigger odd backend errors on Galaxy's side.
    """
    if not PUBLIC_BASE_URL:
        return False
    p = urlparse(PUBLIC_BASE_URL)
    host = (p.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "::1"):
        return False
    return True


def _bytes_to_data_url_image_urls(content: bytes, content_type: str | None) -> list[str]:
    """In-memory only: base64 data URL for Galaxy image_urls (no disk write)."""
    mime = (content_type or "image/jpeg").split(";")[0].strip() or "image/jpeg"
    if not mime.startswith("image/"):
        mime = "image/jpeg"
    b64 = base64.b64encode(content).decode("ascii")
    return [f"data:{mime};base64,{b64}"]


def _image_urls_for_upload(content: bytes, content_type: str | None) -> list[str]:
    mime = content_type or "image/jpeg"
    if _public_base_url_usable_for_galaxy():
        ext = ".jpg"
        if "png" in mime:
            ext = ".png"
        elif "webp" in mime:
            ext = ".webp"
        name = f"{uuid.uuid4().hex}{ext}"
        path = UPLOAD_DIR / name
        path.write_bytes(content)
        return [f"{PUBLIC_BASE_URL}/uploads/{name}"]
    return _bytes_to_data_url_image_urls(content, mime)


async def _download_open_image(
    client: httpx.AsyncClient,
    url: str,
) -> tuple[bytes, str | None]:
    """Fetch image bytes from a public URL; does not write to disk."""
    cleaned = url.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="image_url is empty.")
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail="image_url must use http or https.",
        )
    try:
        resp = await client.get(cleaned, follow_redirects=True)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to download image: {e}") from e
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Image URL returned HTTP {resp.status_code}.",
        )
    data = resp.content or b""
    if not data:
        raise HTTPException(status_code=400, detail="Downloaded image is empty.")
    if len(data) > MAX_IMAGE_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image larger than MAX_IMAGE_DOWNLOAD_BYTES ({MAX_IMAGE_DOWNLOAD_BYTES}).",
        )
    ct = resp.headers.get("content-type")
    if ct:
        ct = ct.split(";")[0].strip()
    return data, ct


async def refine_prompt_with_gemini(
    client: httpx.AsyncClient,
    user_prompt: str,
    image_bytes: bytes,
    image_mime: str,
) -> str:
    """
    Multimodal refinement: user text + image → single refined prompt for FLUX image editing.
    See galaxy_ai/prompt_enhancement.md (Gemini generateContent).
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Set GEMINI_API_KEY (or GOOGLE_API_KEY) for prompt refinement.",
        )
    mime = (image_mime or "image/jpeg").split(";")[0].strip() or "image/jpeg"
    if not mime.startswith("image/"):
        mime = "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    instruction = (
       "You are an expert prompt architect for a high-end image-to-image editing model.\n"
        "Your task is to transform the user's raw instruction into a professional, high-fidelity, and neutral photographic prompt.\n\n"
        
        "CRITICAL RULES:\n"
        "1. SAFETY & NEUTRALITY: This prompt must NEVER be flagged as sensitive. Refer to all humans simply as 'a person'. Use neutral, non-descriptive language regarding clothing (e.g., 'wearing a coat', 'wearing casual items'). NEVER use terms like 'attire', 'mini skirt', 'lingerie', or any suggestive fashion-related vocabulary.\n"
        "2. TECHNICAL PHOTOGRAPHY: Enhance the scene description using professional parameters: '8k resolution', 'photorealistic textures', 'natural light', 'depth of field', 'sharp focus', and 'high dynamic range'.\n"
        "3. LIGHTING & ENVIRONMENT: Describe the lighting in a way that blends the edit with the original image (e.g., 'ambient natural lighting', 'soft diffused light').\n"
        "4. INTENT: Preserve exactly what the user wants to add or change, but describe it with professional, observational accuracy rather than subjective buzzwords like 'stunning' or 'beautiful'.\n"
        "5. OUTPUT: Provide ONLY the refined prompt text. No preamble, no conversational filler.\n\n"
        f"User instruction:\n{user_prompt.strip()}"
    )
    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": instruction},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ],
            },
        ],
        # Add this safetySettings block here
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
    }
    try:
        r = await client.post(
            GEMINI_GENERATE_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": GEMINI_API_KEY,
            },
            timeout=httpx.Timeout(120.0),
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Gemini prompt refinement failed: {e}") from e
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise HTTPException(status_code=502, detail={"gemini_error": err, "status": r.status_code})

    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected Gemini response shape: {data!r}",
        ) from e
    refined = (text or "").strip()
    if not refined:
        raise HTTPException(status_code=502, detail="Gemini returned an empty refined prompt.")
    return refined


_BASE64_KEY_NAMES = frozenset(
    {"base64", "imageBase64", "image_base64", "b64", "encodedImage", "imageData"},
)
_IMAGE_URL_KEY_NAMES = frozenset(
    {
        "url",
        "imageUrl",
        "image_url",
        "outputUrl",
        "output_url",
        "href",
        "src",
        "previewUrl",
        "preview_url",
    },
)


def _looks_like_http_url(s: str) -> bool:
    t = s.strip()
    return t.startswith("http://") or t.startswith("https://")


def _looks_like_image_bytes(b: bytes) -> bool:
    if len(b) < 4:
        return False
    if b[:3] == b"\xff\xd8\xff" or b[:2] == b"\xff\xd8":
        return True
    if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if len(b) >= 6 and b[:6] in (b"GIF87a", b"GIF89a"):
        return True
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return True
    return False


def _try_decode_raw_base64(s: str) -> str | None:
    s = s.strip()
    if _looks_like_http_url(s):
        return None
    if len(s) < 80:
        return None
    try:
        decoded = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError):
        return None
    if len(decoded) < 32 or not _looks_like_image_bytes(decoded):
        return None
    return s


def _split_data_url(s: str) -> tuple[str | None, str | None]:
    s = s.strip()
    if not s.startswith("data:") or ";base64," not in s:
        return None, None
    meta, _, b64 = s.partition(";base64,")
    mime = meta[5:] if meta.startswith("data:") else None
    raw = _try_decode_raw_base64(b64)
    if raw is None:
        return None, None
    return mime or "image/jpeg", raw


def _walk_extract_base64(obj: Any, depth: int = 0) -> tuple[str | None, str | None]:
    """Return (mime, raw_base64) from nested output."""
    if depth > 32:
        return None, None
    if obj is None:
        return None, None
    if isinstance(obj, str):
        if _looks_like_http_url(obj):
            return None, None
        mime, raw = _split_data_url(obj)
        if raw:
            return mime or "image/jpeg", raw
        raw = _try_decode_raw_base64(obj)
        if raw:
            return "image/jpeg", raw
        return None, None
    if isinstance(obj, dict):
        for k in _BASE64_KEY_NAMES:
            if k not in obj:
                continue
            v = obj[k]
            if isinstance(v, str):
                if _looks_like_http_url(v):
                    continue
                dm, dr = _split_data_url(v)
                if dr:
                    return dm or "image/jpeg", dr
                br = _try_decode_raw_base64(v)
                if br:
                    return "image/jpeg", br
            mime, raw = _walk_extract_base64(v, depth + 1)
            if raw:
                return mime, raw
        for v in obj.values():
            mime, raw = _walk_extract_base64(v, depth + 1)
            if raw:
                return mime, raw
    if isinstance(obj, (list, tuple)):
        for item in obj:
            mime, raw = _walk_extract_base64(item, depth + 1)
            if raw:
                return mime, raw
    return None, None


def _walk_extract_image_https_url(obj: Any, depth: int = 0) -> str | None:
    if depth > 32:
        return None
    if obj is None:
        return None
    if isinstance(obj, str):
        u = obj.strip()
        if u.startswith("http://") or u.startswith("https://"):
            p = urlparse(u)
            if p.scheme in ("http", "https") and p.netloc:
                return u
        return None
    if isinstance(obj, dict):
        for k in _IMAGE_URL_KEY_NAMES:
            if k in obj and isinstance(obj[k], str):
                got = _walk_extract_image_https_url(obj[k], depth + 1)
                if got:
                    return got
        for v in obj.values():
            got = _walk_extract_image_https_url(v, depth + 1)
            if got:
                return got
    if isinstance(obj, (list, tuple)):
        for item in obj:
            got = _walk_extract_image_https_url(item, depth + 1)
            if got:
                return got
    return None


_CDN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


async def _download_image_url_to_base64(
    client: httpx.AsyncClient,
    url: str,
) -> tuple[str, str] | None:
    """Fetch image bytes from CDN URL; return (raw_base64, mime) or None if not a valid image."""
    try:
        resp = await client.get(
            url.strip(),
            follow_redirects=True,
            headers=_CDN_HEADERS,
        )
    except httpx.RequestError:
        return None
    if resp.status_code >= 400 or not resp.content:
        return None
    if len(resp.content) > MAX_IMAGE_DOWNLOAD_BYTES:
        return None
    if not _looks_like_image_bytes(resp.content):
        return None
    ct = resp.headers.get("content-type")
    mime_out = (ct.split(";")[0].strip() if ct else None) or "image/jpeg"
    b64 = base64.b64encode(resp.content).decode("ascii")
    return b64, mime_out


async def _resolve_output_to_image_base64(
    client: httpx.AsyncClient,
    output: Any,
) -> tuple[str | None, str | None, str | None]:
    """
    Return (raw_base64, mime_type, image_url_if_downloaded).

    Galaxy often returns a CDN https URL (sometimes under a key like image_base64); we download
    that first and encode to base64. True base64 in output is used only when it decodes to image bytes.
    """
    url = _walk_extract_image_https_url(output)
    if url:
        got = await _download_image_url_to_base64(client, url)
        if got:
            b64, mime = got
            return b64, mime, url

    mime, raw = _walk_extract_base64(output)
    if raw:
        return raw, mime or "image/jpeg", None

    return None, None, None


@app.post("/run")
async def run_flux_edit(
    image: UploadFile | None = File(
        None,
        description="Input image file. Provide this or image_url, not both.",
    ),
    image_url: str | None = Form(
        None,
        description="Public http(s) URL to an image; downloaded in memory and sent as base64 (not saved).",
    ),
    prompt: str = Form(default="your text here"),
    prompt_refinement: bool = Form(
        default=False,
        description=(
            "If true, refine the prompt with Gemini (image + text) before Galaxy; "
            "response 'prompt' is the refined text sent to FLUX."
        ),
    ),
    num_images: int = Form(default=1),
    image_size: str = Form(default="Auto"),
    seed: int = Form(default=0),
    output_format: str = Form(default="JPEG"),
    poll_until_complete: bool = Query(
        default=True,
        alias="poll",
        description="Poll until FAILED, or COMPLETED with a base64 image (or fetchable image URL in output). Use poll=false to return only runId.",
    ),
    poll_interval_sec: float = Query(default=5.0, ge=0.5, le=60.0),
    max_poll_seconds: float = Query(
        default=DEFAULT_MAX_POLL_SECONDS,
        ge=10.0,
        le=3600.0,
        description="Give up if no base64 image is available within this wall time (seconds).",
    ),
    include_galaxy: bool = Query(
        default=False,
        description="If true, include the full last Galaxy status JSON under key 'galaxy'.",
    ),
) -> JSONResponse:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Set GALAXY_API_KEY in the environment.",
        )
    if prompt_refinement and not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Set GEMINI_API_KEY or GOOGLE_API_KEY when prompt_refinement is enabled.",
        )

    user_prompt = prompt

    # Swagger UI often auto-fills optional string fields with the literal "string".
    # Treat that placeholder as empty so users can upload files without manual cleanup.
    url_s = (image_url or "").strip()
    if url_s.lower() == "string":
        url_s = ""
    if url_s and image is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either multipart 'image' or form 'image_url', not both.",
        )
    if not url_s and image is None:
        raise HTTPException(
            status_code=400,
            detail="Provide multipart file 'image' or form field 'image_url'.",
        )

    timeout = httpx.Timeout(120.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if url_s:
            body, remote_mime = await _download_open_image(client, url_s)
            image_urls = _bytes_to_data_url_image_urls(body, remote_mime)
            image_mime = remote_mime or _sniff_image_mime(body) or "image/jpeg"
        else:
            assert image is not None
            body = await image.read()
            if not body:
                raise HTTPException(status_code=400, detail="Empty file.")
            upload_mime = _normalize_upload_mime(body, image.content_type)
            image_urls = _image_urls_for_upload(body, upload_mime)
            image_mime = upload_mime

        effective_prompt = user_prompt
        if prompt_refinement:
            effective_prompt = await refine_prompt_with_gemini(
                client,
                user_prompt,
                body,
                image_mime,
            )

        payload = {
            "nodeType": "flux_2_pro",
            "input": {
                "prompt": effective_prompt,
                "num_images": num_images,
                "image_urls": image_urls,
                "image_size": image_size,
                "seed": seed,
                "output_format": output_format,
            },
            "subModelId": "flux-2-pro-edit",
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        try:
            response = await client.post(
                f"{BASE_URL}/v1/nodes/flux_2_pro/run",
                json=payload,
                headers=headers,
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Galaxy request failed: {e}") from e

        if response.status_code >= 400:
            detail: str | dict
            try:
                detail = response.json()
            except Exception:
                detail = response.text or response.reason_phrase
            raise HTTPException(status_code=response.status_code, detail=detail)

        start = response.json()
        run_id = start.get("runId")
        if not poll_until_complete or not run_id:
            extra: dict[str, Any] = {"prompt": effective_prompt}
            if prompt_refinement:
                extra["original_prompt"] = user_prompt
            merged = {**start, **extra}
            return JSONResponse(content=merged)

        poll_deadline = time.monotonic() + max_poll_seconds

        # Poll like plan.md until FAILED, or COMPLETED and we have base64 (or downloadable image URL → base64).
        while True:
            if time.monotonic() > poll_deadline:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Polling exceeded max_poll_seconds without obtaining image_base64 from "
                        "Galaxy output (or from an image URL inside output)."
                    ),
                )
            try:
                poll_resp = await client.get(
                    f"{BASE_URL}/v1/nodes/runs/{run_id}",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Galaxy poll failed: {e}") from e

            if poll_resp.status_code >= 400:
                try:
                    err = poll_resp.json()
                except Exception:
                    err = poll_resp.text
                raise HTTPException(status_code=poll_resp.status_code, detail=err)

            status_payload = poll_resp.json()
            st = status_payload.get("status")
            if st == "FAILED":
                raise HTTPException(
                    status_code=502,
                    detail=status_payload.get("error", status_payload),
                )
            if st == "COMPLETED":
                out = status_payload.get("output")
                raw_b64, mime, source_url = await _resolve_output_to_image_base64(client, out)
                if raw_b64:
                    run_id_final = status_payload.get("runId") or run_id
                    mime_final = mime or "image/jpeg"
                    body: dict[str, Any] = {
                        "status": "COMPLETED",
                        "mime_type": mime_final,
                        "image_base64": raw_b64,
                        "image_data_url": f"data:{mime_final};base64,{raw_b64}",
                        "runId": run_id_final,
                        "prompt": effective_prompt,
                    }
                    if prompt_refinement:
                        body["original_prompt"] = user_prompt
                    if source_url:
                        body["image_url"] = source_url
                    if include_galaxy:
                        body["galaxy"] = status_payload
                    return JSONResponse(content=body)
                await asyncio.sleep(poll_interval_sec)
                continue

            await asyncio.sleep(poll_interval_sec)


# Serve uploaded files when PUBLIC_BASE_URL points at this host (e.g. behind ngrok).
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
