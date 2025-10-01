# Visualizing ShameScore Directions

This note explains how to paint streaming tokens with a "shame" intensity using
either a fully local model or a hosted API. The goal is to approximate the
cosine similarity between each generated token and a vector that represents the
"shame" concept.

## 1. Local models (full internals)

When you run the model locally you can directly inspect intermediate tensors.
This is the most accurate setup because you can re-use the model's own hidden
states to compute the shame direction and to monitor generation.

### What you can access

- Hidden states for each layer and decoding step.
- Logits (pre-softmax scores) at each step.
- Attentions (optional, often not needed for the heatmap).

### Pre-compute the shame direction

1. Create two small prototype banks of sentences (e.g., `Embarrassing` vs.
   `Professional` and `AntiValues` vs. `Values`).
2. Encode them with the exact same model you will use for generation.
3. Mean-pool each bank to obtain sentence representations.
4. Combine them into a single vector
   \(v_{\text{shame}} = \alpha (Embarrassing - Professional)
   + \delta (AntiValues - Values)\).
5. Normalize the result (L2 norm = 1).

### Monitor generation

Use a manual decoding loop or generation hooks in Hugging Face Transformers.
The manual loop gives more control:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16
).eval().to("cuda")

def token_rep(x_ids):
    out = model(input_ids=x_ids, output_hidden_states=True)
    h_last = out.hidden_states[-1]  # [B, T, H]
    last_tok = h_last[:, -1, :]     # [B, H]
    logits = out.logits[:, -1, :]   # [B, V]
    return last_tok, logits

# 1) precompute v_shame using the same model (encode your banks and mean-pool)
v_shame = ...  # shape [H], L2-normalized

ids = tok("Prompt...", return_tensors="pt").input_ids.to(model.device)
scores = []
for _ in range(max_new_tokens):
    h_t, logits = token_rep(ids)
    s = torch.nn.functional.cosine_similarity(
        h_t, torch.tensor(v_shame, device=h_t.device).unsqueeze(0), dim=-1
    )
    scores.append(float(s))
    next_id = logits.argmax(dim=-1)
    ids = torch.cat([ids, next_id], dim=-1)
```

Attach `scores` to the generated tokens and render them in your UI (e.g., via an
HTML heatmap). If you prefer to use `model.generate`, register a forward hook on
the final block to capture `hidden_states[-1][:, -1, :]` per step and compute
the cosine similarity afterwards.

### Practical notes

- Keep every step on the same model to avoid embedding space mismatches.
- Capture only the last layer's last token to reduce memory pressure.
- Normalize the heatmap per text span (e.g., clip between the 5th and 95th
  percentiles) for smoother colors.

## 2. Hosted APIs (logits only)

Most hosted APIs expose token log-probabilities but hide hidden states. You can
still build useful proxies with two complementary approaches.

### Option A — Logit-based proxy (streaming friendly)

1. Maintain two lexicons (e.g., boosters vs. hedges) plus a short list of
   "authority tone" tokens.
2. For each step, sum the probability mass assigned to booster tokens and hedge
   tokens in the returned top-\(k\) logprobs.
3. Define a per-token style score, such as
   `style_t = sum(p_boosters) - sum(p_hedges)`.
4. Smooth the live score with an exponential moving average before coloring the
   token to make the UI stable.

This produces a fast, live indicator of overconfident language even without
hidden states.

### Option B — Post-hoc embeddings (accurate, offline)

1. After the API returns the text (per chunk or full response), run a local
   encoder on the text.
2. Re-tokenize with that encoder and map token offsets back to character spans.
3. Compute cosine similarity with `v_shame` exactly as in the local workflow and
   recolor the spans.

This gives you the "true" shame heatmap at the cost of latency.

### Option C — Hybrid

Stream the logit-based proxy during generation, then overwrite it with the
post-hoc embedding heatmap once the chunk completes.

## 3. Quick comparison

| Setting              | Data available                     | Recommended approach                                |
| -------------------- | ---------------------------------- | --------------------------------------------------- |
| Local, own weights   | Hidden states + logits             | Compute cosine with `v_shame` per token in real time |
| Hosted API (logprobs)| Token text + logprobs (top-*k*)    | Live proxy via booster−hedge mass                    |
| Hosted API (text)    | Tokenized text only                | Post-hoc embeddings with local encoder               |

## 4. Implementation tips

- Use `return_offsets_mapping=True` when you tokenize so you can map subword
  tokens back to character ranges for coloring.
- For interactive dashboards, compute cosine similarity on the GPU and stream
  the score alongside the token.
- Limit retries and caching if you rely on paid APIs.
- For long sequences, periodically flush cached hidden states to disk or only
  keep the last positions required for visualization.

