"""Resolve a HuggingFace access token, optionally.

Order of preference:
  1. HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variable
  2. Colab secrets sidebar (a secret named HF_TOKEN, with notebook access granted)

Returns None if no token is available -- callers then fall back to anonymous
(rate-limited) requests, so this is safe to use everywhere.
"""
import os


def get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None
