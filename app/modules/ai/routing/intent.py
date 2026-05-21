"""
Lightweight intent router.

Classifies an incoming user message into one of:

  * ``ml_explain``     — user wants a tabular ML prediction explained.
  * ``vision_explain`` — user wants a vision prediction explained.
  * ``general_domain`` — on-topic project question (definitions, usage).
  * ``off_topic``      — out-of-scope; the chat service emits a refusal.

This is keyword-based on purpose. It runs in <1 ms and is fully
deterministic — we never trust the LLM to gate itself.
"""
from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    ML_EXPLAIN = "ml_explain"
    VISION_EXPLAIN = "vision_explain"
    GENERAL_DOMAIN = "general_domain"
    OFF_TOPIC = "off_topic"


# Turkish and English keywords that mark a message as on-topic.
_DOMAIN_KEYWORDS = {
    # Disease / domain
    "hanta", "hantavirüs", "hantavirus", "hps", "fare", "kemirgen",
    "rodent", "rodents",
    # ML / prediction
    "model", "tahmin", "prediction", "olasılık", "probability",
    "güven", "confidence", "risk", "skor", "score",
    "sınıflandırma", "classification", "f1", "doğruluk", "accuracy",
    # Vision
    "görüntü", "resim", "image", "fotoğraf", "fotograf", "photo",
    "grad-cam", "gradcam", "heatmap", "ısı haritası", "isı haritası",
    "kapı", "gate", "ilgisiz", "unrelated", "ilgili", "related",
    # Project / API
    "hantaproject", "api", "endpoint", "yükleme", "upload",
    "eşik", "threshold",
}

_VISION_KEYWORDS = {
    "görüntü", "resim", "image", "fotoğraf", "fotograf", "photo",
    "yükle", "upload", "grad-cam", "gradcam", "heatmap",
    "ısı haritası", "isı haritası", "ilgisiz", "gate", "kapı",
}

_ML_KEYWORDS = {
    "hasta", "patient", "olasılık", "probability", "tahmin",
    "prediction", "risk", "skor", "score", "ateş", "fever",
    "tablo", "feature", "özellik",
}

_OFF_TOPIC_HINTS = {
    "kod yaz", "write code", "şarkı", "song", "şiir", "poem",
    "siyaset", "politics", "futbol", "football", "borsa", "stock",
    "tarif", "recipe",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_intent(
    message: str,
    *,
    has_ml_context: bool = False,
    has_vision_context: bool = False,
) -> Intent:
    """
    Decide what kind of message we're dealing with.

    Parameters
    ----------
    message : str
        Raw user text.
    has_ml_context : bool
        True when the request includes a tabular ML prediction payload —
        we always treat it as ML-explain even if keywords are missing.
    has_vision_context : bool
        Same idea for vision payloads.
    """
    if has_vision_context:
        return Intent.VISION_EXPLAIN
    if has_ml_context:
        return Intent.ML_EXPLAIN

    normalized = _normalize(message)
    tokens = set(re.findall(r"[\wçğıöşüâîû\-]+", normalized))

    if any(hint in normalized for hint in _OFF_TOPIC_HINTS):
        # Off-topic hint AND no domain keyword → reject.
        if not (tokens & _DOMAIN_KEYWORDS):
            return Intent.OFF_TOPIC

    if tokens & _VISION_KEYWORDS:
        return Intent.VISION_EXPLAIN
    if tokens & _ML_KEYWORDS:
        return Intent.ML_EXPLAIN
    if tokens & _DOMAIN_KEYWORDS:
        return Intent.GENERAL_DOMAIN

    return Intent.OFF_TOPIC