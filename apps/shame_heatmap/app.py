"""Gradio app to visualize token similarity to a shame direction."""
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
BANKS_DIR = APP_DIR / "banks"
DEFAULT_MODEL_NAME = "xlm-roberta-base"


@dataclass
class TokenEmbeddings:
    tokens: List[str]
    offsets: List[Tuple[int, int]]
    vectors: np.ndarray


class Encoder:
    """Sentence and token encoder wrapper around Hugging Face models."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def sentence_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        """Return mean pooled embeddings for the provided sentences."""

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=512,
        ).to(self.device)
        outputs = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (outputs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts
        return embeddings.detach().cpu().numpy()

    @torch.no_grad()
    def token_embeddings(self, text: str) -> TokenEmbeddings:
        """Return token-level embeddings for a single string."""

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**inputs).last_hidden_state[0].detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return TokenEmbeddings(tokens=tokens, offsets=offsets, vectors=outputs)


def load_banks(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=8)
def cached_banks(language: str) -> Dict[str, List[str]]:
    file_name = f"{language}.json"
    bank_path = BANKS_DIR / file_name
    if not bank_path.exists():
        raise FileNotFoundError(
            f"Could not find banks file '{file_name}'. Place it under {BANKS_DIR.relative_to(APP_DIR)}."
        )
    return load_banks(bank_path)


def normalize_scores(scores: np.ndarray, percentile_low: float, percentile_high: float) -> np.ndarray:
    low = np.percentile(scores, percentile_low)
    high = np.percentile(scores, percentile_high)
    if high - low < 1e-8:
        return np.zeros_like(scores)
    clipped = np.clip((scores - low) / (high - low), 0.0, 1.0)
    return clipped


def color_for_score(score: float) -> str:
    red = 255
    green = int(round(255 * (1 - score)))
    blue = int(round(255 * (1 - score)))
    return f"#{red:02x}{green:02x}{blue:02x}"


def render_html(text: str, offsets: Sequence[Sequence[int]], scores: Sequence[float]) -> str:
    parts: List[str] = []
    cursor = 0
    for (start, end), score in zip(offsets, scores):
        start = max(0, min(len(text), start))
        end = max(0, min(len(text), end))
        if end <= start:
            continue
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        fragment = html.escape(text[start:end]) or " "
        parts.append(
            (
                f'<span title="{score:.3f}" '
                f'style="background:{color_for_score(score)};border-radius:3px;padding:1px 1px;">'
                f"{fragment}</span>"
            )
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def shame_vector(encoder: Encoder, banks: Dict[str, Iterable[str]], alpha: float, delta: float) -> np.ndarray:
    embarrassing = encoder.sentence_embeddings(banks["embarrassing_style"]).mean(axis=0)
    professional = encoder.sentence_embeddings(banks["professional_style"]).mean(axis=0)
    anti_values = encoder.sentence_embeddings(banks["anti_values"]).mean(axis=0)
    values = encoder.sentence_embeddings(banks["values"]).mean(axis=0)

    style = embarrassing - professional
    community = anti_values - values

    style /= np.linalg.norm(style) + 1e-8
    community /= np.linalg.norm(community) + 1e-8

    direction = alpha * style + delta * community
    return direction / (np.linalg.norm(direction) + 1e-8)


@lru_cache(maxsize=1)
def cached_encoder(model_name: str = DEFAULT_MODEL_NAME) -> Encoder:
    return Encoder(model_name=model_name)


def app_function(
    text: str,
    language: str,
    alpha: float,
    delta: float,
    percentile_low: float,
    percentile_high: float,
) -> str:
    if not text.strip():
        return "<i>Paste text to visualise token similarity.</i>"

    encoder = cached_encoder()
    banks = cached_banks(language)
    direction = shame_vector(encoder, banks, alpha=alpha, delta=delta)

    token_embeddings = encoder.token_embeddings(text)
    vectors = token_embeddings.vectors
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    scores = vectors @ direction
    normalised = normalize_scores(scores, percentile_low, percentile_high)

    rendered = render_html(text, token_embeddings.offsets, normalised)
    container_style = (
        "font-family:ui-monospace,monospace;white-space:pre-wrap;border:1px solid #eee;"
        "border-radius:8px;padding:12px;background:#fafafa;"
    )
    return f"<div style=\"{container_style}\">{rendered}</div>"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ShameFeeling Token Heatmap") as demo:
        gr.Markdown(
            "## ShameFeeling — Token Heatmap (Style + Values)\n"
            "Color intensity encodes similarity to the shame direction (white → red)."
        )
        text_input = gr.Textbox(lines=6, label="Text to visualise")
        with gr.Row():
            language = gr.Radio(choices=["pt", "en"], value="pt", label="Language")
            alpha = gr.Slider(0, 1, value=0.6, step=0.05, label="α — Style weight")
            delta = gr.Slider(0, 1, value=0.4, step=0.05, label="δ — Values weight")
        with gr.Row():
            percentile_low = gr.Slider(0, 20, value=5, step=1, label="Low percentile clip")
            percentile_high = gr.Slider(80, 100, value=95, step=1, label="High percentile clip")
        output = gr.HTML()
        render_button = gr.Button("Render")
        render_button.click(
            app_function,
            [text_input, language, alpha, delta, percentile_low, percentile_high],
            output,
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
