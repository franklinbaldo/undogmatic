from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Lightweight, configurable embedding backend.
# - Default: local TF-IDF (no downloads).
# - API mode: call an embeddings endpoint (e.g., Gemma embeddings via a provider)
#   by setting EMBED_BACKEND=api and configuring EMBED_MODEL/EMBED_API_KEY/EMBED_BASE_URL.

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local").lower()  # 'local' | 'api'
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemma-2-embedding")  # suggest Gemma by default
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1/embeddings")


def _embed_api(texts: Sequence[str]) -> np.ndarray:
    payload = {"model": EMBED_MODEL, "input": list(texts)}
    headers = {"Content-Type": "application/json"}
    if EMBED_API_KEY:
        headers["Authorization"] = f"Bearer {EMBED_API_KEY}"
    request = urllib.request.Request(
        EMBED_BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60.0) as resp:  # pragma: no cover - I/O
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:  # pragma: no cover - I/O
        raise RuntimeError("Embedding API request failed") from exc

    vectors = [item["embedding"] for item in data.get("data", [])]
    arr = np.array(vectors, dtype=float)
    return normalize(arr)


_VECTORIZER: TfidfVectorizer | None = None


def _embed_local(texts: Sequence[str]) -> np.ndarray:
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
        )
        # Fit once on small Portuguese prototype corpus to stabilize space
        _VECTORIZER.fit(_HUBRIS_PT + _HUMILITY_PT)
    mat = _VECTORIZER.transform(texts)
    return normalize(mat.toarray())


def _embed(text_or_texts: str | Iterable[str]) -> np.ndarray:
    texts = [text_or_texts] if isinstance(text_or_texts, str) else list(text_or_texts)
    if EMBED_BACKEND == "api":
        return _embed_api(texts)
    return _embed_local(texts)


# ---- Prototypes (Portuguese, tiny seed; edit freely) ----
_HUBRIS_PT: List[str] = [
    "Este precedente encerra definitivamente a questão; inexiste autoridade em sentido contrário.",
    "O tema do tribunal resolve por completo o caso e dispensa análise adicional.",
]
_HUMILITY_PT: List[str] = [
    "Sem prejuízo de entendimento diverso e das peculiaridades fáticas, o precedente sugere um caminho.",
    "O resultado depende do contexto; a autoridade indica mas não substitui a fundamentação.",
]

# The prototype embeddings can be computed in a single batch
_PROTO_HUBRIS = np.mean(_embed(_HUBRIS_PT), axis=0)
_PROTO_HUMILI = np.mean(_embed(_HUMILITY_PT), axis=0)


def score_text(text: str) -> dict:
    """
    Zero-shot prototype scoring:
    ShameScore ~ scaled (cos(text, hubris) - cos(text, humility)) in [0..100].
    Works with local TF‑IDF or an external embeddings API (e.g., Gemma embeddings).
    """
    v = _embed(text)[0]
    s_hub = float(np.dot(v, _PROTO_HUBRIS))
    s_hum = float(np.dot(v, _PROTO_HUMILI))
    # cosines are in [-1,1]; map difference to [0,1]
    raw = 0.5 * (s_hub - s_hum + 1.0)
    score = int(round(100 * max(0.0, min(1.0, raw))))
    backend = "api" if EMBED_BACKEND == "api" else "local"
    rationale = (
        f"{backend} embedding score via prototypes (hubris={s_hub:.3f}, humility={s_hum:.3f})"
    )
    return {"shame_score": score, "confidence": 80, "rationale": rationale}
