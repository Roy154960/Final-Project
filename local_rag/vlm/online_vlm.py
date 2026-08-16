"""
VLM method 3: vision-language model via a free HOSTED API (not local).

Same describe_image()/answer_with_image() interface as vlm/ollama_vlm.py's
OllamaVLM and vlm/hf_vlm.py's HFVLM, so ingestion/image_captioning.py's
load_vlm() and local_rag/personal_rag.py's caller can swap this in without
either needing to know which backend it's actually talking to (see
config.py's own API_KEY / MODEL_NAME / PERSONAL_RAG_SINGLE_IMAGE_VLM_BACKEND
for exactly when this one gets picked, and .env at the project root for
where API_KEY/MODEL_NAME actually get set -- see config.py's own
load_dotenv() call).

Uses Google's Gemini API free tier (see
https://ai.google.dev/gemini-api/docs/rate-limits for current limits, and
https://aistudio.google.com/apikey to generate a key) via plain REST +
`requests` -- deliberately NOT the `google-genai` SDK, so this doesn't add
a new hard dependency to a project whose whole README leads with "no paid
APIs, everything else runs locally." One image, one short prompt, one
response: this is a thin, single-purpose HTTP client, not a reason to pull
in a general-purpose SDK.

Never silently no-ops: a missing API_KEY raises a clear RuntimeError with
the exact fix (same "tell the person running this the real fix" convention
config.py's own TESSERACT_CMD comment and ingest_pdf.py's
TesseractNotFoundError message already follow), rather than returning an
empty caption that would look, from the caller's side, like the image
itself had nothing describable in it.

Run directly to smoke-test:
    python -m vlm.online_vlm data/raw/sample.png "What is shown in this image?"
"""

import base64
import mimetypes
import sys
from pathlib import Path

import requests

from config import API_KEY, MODEL_NAME

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# config.py's own MODEL_NAME is a plain os.getenv() with no default (see
# that file's own .env-based setup) -- None/"" whenever it isn't set in
# the environment at all, which would otherwise silently build a URL like
# ".../models/None:generateContent" and fail with a confusing 404 instead
# of a clear "you forgot to set this" error. This is the fallback used
# whenever config.MODEL_NAME is falsy -- "-flash", not "-pro", since
# captioning one image is squarely within flash's free-tier quota and
# speed target; set MODEL_NAME in your .env if you'd rather trade latency
# for quality with a pro model.
_DEFAULT_MODEL = "gemini-2.0-flash"

# Generous but bounded -- a hosted call over the open internet needs a real
# ceiling (unlike OllamaVLM's own local, unbounded call), so one slow/stuck
# request can't hang a chat turn indefinitely. Long enough for a normal
# image + short prompt on Gemini Flash under ordinary conditions.
_REQUEST_TIMEOUT_SECONDS = 30


class OnlineVLM:
    def __init__(self, model: str = None, api_key: str = API_KEY):
        if not api_key:
            raise RuntimeError(
                "OnlineVLM needs a free Gemini API key. Fix:\n"
                "  1. Generate one at https://aistudio.google.com/apikey (no card required)\n"
                "  2. Put it in your .env file at the project root:\n"
                "       API_KEY=your-real-key-here\n"
                "       MODEL_NAME=gemini-2.0-flash\n"
                "  3. Restart agents/api.py so config.py's load_dotenv() picks it up.\n"
                "Or set config.PERSONAL_RAG_SINGLE_IMAGE_VLM_BACKEND = \"ollama\" "
                "to skip the hosted VLM entirely and stay fully local."
            )
        resolved_model = model or MODEL_NAME or _DEFAULT_MODEL
        self.name = f"online-vlm:{resolved_model}"
        self.model = resolved_model
        self.api_key = api_key

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"  # Gemini needs SOME image/* mime type; a safe default
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return mime_type, data

    def _generate(self, prompt: str, image_path: str) -> str:
        mime_type, image_b64 = self._encode_image(image_path)
        url = f"{_GEMINI_BASE_URL}/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                    ]
                }
            ]
        }
        try:
            # Sent as the `x-goog-api-key` HEADER, not a `?key=` query
            # param. Google is actively migrating Gemini API keys from
            # the old "Standard" format (AIzaSy...) to a new "Auth key"
            # format (AQ....) -- as of mid-2026, every key Google AI
            # Studio issues is an Auth key by default (see
            # https://ai.google.dev/gemini-api/docs/api-key, "API key
            # types: standard versus authorization"). Google's own
            # current REST example authenticates with this header, not
            # the old query-param style this code used before -- and
            # multiple confirmed reports on Google's own developer forum
            # (e.g. discuss.ai.google.dev's "New API keys generated with
            # 'AQ.' prefix don't work with REST endpoint" thread) show
            # Auth-format keys failing specifically over the old
            # query-param path. A key that looks like AQ.Ab8... (not
            # AIzaSy...) is expected now, not a sign your .env is
            # misconfigured.
            resp = requests.post(
                url,
                headers={"x-goog-api-key": self.api_key},
                json=payload,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Never leak the raw exception (which can include the request
            # URL) up to a chat answer -- same "developer sees the real
            # error, the person sees a plain sentence" rule this project
            # applies everywhere else (see agents/api.py's _invoke_turn,
            # agents/specialists.py's _looks_like_tool_error). The caller
            # (personal_rag.py) catches this and falls back to the local
            # Ollama VLM, so this is a log line for the developer more
            # than it is the end of the road for the caption itself.
            raise RuntimeError(f"Gemini VLM request failed: {e}") from e

        data = resp.json()
        try:
            candidates = data["candidates"]
            parts = candidates[0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError):
            # A blocked/empty response (Gemini's own safety filters, or an
            # image it couldn't process) still comes back as valid JSON,
            # just without the usual candidates[0].content.parts shape --
            # degrade to a plain, honest string rather than raising on a
            # response that technically succeeded at the HTTP layer.
            reason = data.get("promptFeedback", {}).get("blockReason")
            if reason:
                return f"(the online VLM declined to describe this image: {reason})"
            return "(the online VLM returned an empty response for this image)"

    def describe_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        return self._generate(prompt, image_path)

    def answer_with_image(self, question: str, image_path: str, text_context: str = "") -> str:
        """
        Same contract as OllamaVLM.answer_with_image / HFVLM's equivalent:
        combine a retrieved/uploaded image with any retrieved text context
        and the user's question in one VLM call.
        """
        prompt = f"{('Context: ' + text_context + chr(10) + chr(10)) if text_context else ''}Question: {question}"
        return self._generate(prompt, image_path)


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.png"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is shown in this image?"

    if not Path(image_path).exists():
        print(f"No image found at {image_path} — pass a real image path to test this.")
    else:
        vlm = OnlineVLM()
        print(vlm.answer_with_image(question, image_path))
