from __future__ import annotations

import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Model config
MODEL_ID = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model load (once) ----
_model = SentenceTransformer(MODEL_ID, device=DEVICE)


def _embed(text: str | list[str]) -> np.ndarray:
    """Return a normalized sentence embedding."""
    # The model.encode method handles tokenization, pooling, and normalization
    embedding = _model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding


# ---- Prototypes (Portuguese, tiny seed; edit freely) ----
_HUBRIS_PT = [
    "Este precedente encerra definitivamente a questão; inexiste autoridade em sentido contrário.",
    "O tema do tribunal resolve por completo o caso e dispensa análise adicional.",
]
_HUMILITY_PT = [
    "Sem prejuízo de entendimento diverso e das peculiaridades fáticas, o precedente suge-re um caminho.",
    "O resultado depende do contexto; a autoridade indica mas não substitui a fundamentação.",
]

# The prototype embeddings can be computed in a single batch
_PROTO_HUBRIS = np.mean(_embed(_HUBRIS_PT), axis=0)
_PROTO_HUMILI = np.mean(_embed(_HUMILITY_PT), axis=0)


def score_text(text: str) -> dict:
    """
    Zero-shot prototype scoring:
    ShameScore ~ scaled (cos(text, hubris) - cos(text, humility)) in [0..100].
    """
    v = _embed(text)
    s_hub = float(np.dot(v, _PROTO_HUBRIS))
    s_hum = float(np.dot(v, _PROTO_HUMILI))
    raw = 0.5 * (s_hub - s_hum + 1.0)  # map [-1,1] -> [0,1]
    score = int(round(100 * max(0.0, min(1.0, raw))))
    rationale = f"Local embedding score via prototypes (hubris={s_hub:.3f}, humility={s_hum:.3f})"
    return {"shame_score": score, "confidence": 80, "rationale": rationale}
