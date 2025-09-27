
# UNDOGMATIC: Undoing Naïve Deference to Overstated Generalized Mandates And Tendentious Interpretations of Citations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/franklinbaldo/undogmatic/actions/workflows/test.yml/badge.svg)](https://github.com/franklinbaldo/undogmatic/actions/workflows/test.yml)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange.svg)](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending)

## Overview

**Disclaimer:** This is a research prototype to reduce rhetorical overconfidence in LLM outputs. It does **not** constitute legal advice and should be used under professional supervision.

UNDOGMATIC (**U**ndoing **N**aïve **D**eference to **O**verstated **G**eneralized **M**andates **A**nd **T**endentious **I**nterpretations of **C**itations) is a minimalist, end-to-end prototype for detecting **hubris**—overconfident, categorical, or authority-sounding language—in AI-generated legal text. Hubris here measures stylistic and evidential signals of overreach (e.g., assertive tone without analysis), not factual truth per se. The core metric is a **HubrisScore**, a composite of:

- **Semantic similarity**: cosine similarity between sentence embeddings and prototype anchors (e.g., *humility* “It would be embarrassing to assert this without analysis” vs. *hubris* “This case absolutely settles the matter”).
- **Stylistic signals**: normalized counts of **hedges** (“might”, “talvez”, “it appears”) and **boosters** (“clearly”, “claramente”).
- **Structural legal cues**: detection of **bare citations** (e.g., “410 U.S. 113” with no reasoning) and **bonuses** when the answer aligns with a provided **precedent summary**.

Also included:
- **LoRA fine-tuning**: train a lightweight classifier on your labeled data (`hubris=1`, `humility=0`).
- **RLHF/TRL hook**: a reward function that penalizes high HubrisScore during post-training.

Public evaluations of legal LLMs report frequent overconfidence and hallucinations; this pipeline operationalizes a practical counter-signal. (Add citations in WHITEPAPER.md.)

## Why This Matters
Legal AI must embody humility: assertions require caveats, counter-authority, and verification. Overconfident tones amplify risks in contract drafting, case analysis, or advice—often rooted in fallacious appeals to authority (argumentum ad verecundiam). UNDOGMATIC undoes such deference by promoting analytical depth.

## Requirements

- Python 3.8+
- Key libraries: `transformers`, `accelerate`, `peft`, `datasets`, `scikit-learn`, `torch`, `sentence-transformers`.
- Create `requirements.txt` with:
  ```
  torch>=2.0.0
  transformers>=4.30.0
  accelerate>=0.20.0
  peft>=0.4.0
  datasets>=2.14.0
  scikit-learn>=1.3.0
  sentence-transformers>=2.2.0
  numpy
  pyyaml  # For config support
  pytest  # For tests
  ```

Install with: `pip install -r requirements.txt`.

**Reproducibility:** Set `PYTHONHASHSEED=0` and `HF_HUB_DISABLE_TELEMETRY=1` in your env. Pin model revisions via `revision=` in `from_pretrained(...)` for fully deterministic runs (e.g., `revision="main"` or a commit hash).

**GPU Check:** Run `python -c "import torch; print(torch.cuda.is_available())"` to verify CUDA support.

**Note**: Use a local embedding model like `sentence-transformers/all-MiniLM-L6-v2` for efficiency. Bilingual support (EN/PT) via lang-specific lexicons.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/franklinbaldo/undogmatic.git
   cd undogmatic
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Run tests: `pytest tests/`.

## Quick Start

Run the example script to compute signals and scores:

```bash
python hubris_detection_pipeline.py
```

**Sample Output**:
```
[Signals]
     sim_to_hubris: 0.6234
   sim_to_humility: 0.4567
      boosters_norm: 0.1250
       hedges_norm: 0.0000
   bare_citation: 1.0000
  explained_bonus: 0.2345
[HubrisScore] 0.5123  (higher = more hubris)

[Reward] -0.4123  (more negative = stronger penalty)
```

### Core Usage Example

```python
from hubris_detection_pipeline import (
    SentenceEncoder, HubrisSignaler, HubrisScorer, HubrisReward
)

# Initialize
encoder = SentenceEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
signaler = HubrisSignaler(encoder, lang="en")  # or "pt" for Portuguese
scorer = HubrisScorer()
reward_fn = HubrisReward(signaler, scorer, lambda_hubris=1.0, margin=0.1)

# English example (hubristic)
text = "Clearly, Brown v. Board of Education resolves this entirely. No contrary authority."
precedent_summary = "Brown held separate schools unconstitutional, but scope limited by later cases."

signals = signaler.compute(text, precedent_summary=precedent_summary)
score = scorer.score(signals)
reward = reward_fn(text, precedent_summary=precedent_summary)

print(f"HubrisScore: {score:.4f}")
print(f"RLHF Reward: {reward:.4f}")

# Portuguese example
signaler_pt = HubrisSignaler(encoder, lang="pt")
texto = "Claramente, este precedente encerra a questão; não há autoridade em sentido contrário."
resumo_precedente = "O acórdão reconheceu X, mas casos posteriores limitaram o alcance em Y."
print(scorer.score(signaler_pt.compute(texto, resumo_precedente)))
```

- **Tuning**: Adjust weights in `SignalWeights` (e.g., emphasize bare citations with `bare_cite_w=0.8`).

### Configure Weights via YAML
Create `config.yaml` (see `config.example.yaml`):
```yaml
weights:
  sim_hubris_w: 0.8
  sim_humility_w: 0.8
  booster_w: 0.4
  hedge_w: 0.4
  bare_cite_w: 0.7
  explained_bonus_w: 0.6
lang: "en"
model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

Load in script (add argparse for `--config`):
```python
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
weights = SignalWeights(**cfg["weights"])
signaler = HubrisSignaler(encoder, lang=cfg["lang"], weights=weights)
```

### LoRA Fine-Tuning

Uncomment and run the training block in the script with your data:

```python
train_texts = ["Your hubristic example...", "Your humble example..."]
train_labels = [1, 0]  # 1=hubris, 0=humility
eval_texts = [...]
eval_labels = [...]

output_dir = train_lora_hubris_classifier(
    train_texts, train_labels, eval_texts, eval_labels,
    base_model="bert-base-uncased",  # Or multilingual for PT
    epochs=3, batch_size=8
)
print(f"Model saved to: {output_dir}")
```

**Data Prep Tips**:
- Source from legal corpora: Caselaw Access Project (US) or STF/STJ decisions (BR).
- Label 500+ examples; split 80/20 train/eval.
- Expected metrics: Accuracy/F1 >0.80 on eval.

### RLHF/TRL Integration

In your PPO/DPO loop:
```python
# During reward computation
hubris_reward = reward_fn(generated_response, retrieved_precedent_summary)
total_reward += hubris_reward  # Penalizes hubris > margin
```

Tune `lambda_hubris` to balance with other rewards.

## Evaluation & Ethics

- **Calibration**: Report ECE, Brier, and risk–coverage on a held-out legal QA set.
- **Hubris metric**: % of answers flagged “authority-only” (bare cite, no reasoning) and correlation with human ratings.
- **Ablations**: style-only vs latent-only vs combined signals.
- **Jurisdictional variance**: writing norms differ (EN/PT, common/civil law). Consider separate thresholds or fine-tuning.
- **Anti-gaming**: cap credit from repeated hedges; require reasoning fields to receive “humility” bonuses.
- **Privacy**: avoid uploading sensitive case text; prefer local models; redact names when sharing datasets.

## Project Structure

```
undogmatic/
├── hubris_detection_pipeline.py  # Core implementation
├── requirements.txt              # Dependencies
├── config.example.yaml           # Sample config
├── README.md                     # This file
├── WHITEPAPER.md                 # Detailed methodology
├── examples/                     # Sample texts and notebooks
│   └── legal_samples.json        # Labeled toy dataset
└── tests/                        # Unit tests for signals/scorer
    ├── test_signals.py
    └── test_bare_citation.py
```

## Contributing

We welcome contributions! To get started:

1. Fork the repo and create a feature branch: `git checkout -b feature/your-feature`.
2. Commit your changes: `git commit -m "Add your feature"`.
3. Push to the branch: `git push origin feature/your-feature`.
4. Open a Pull Request on GitHub.

**Ideas**:
- Add token-level scoring for streaming generation.
- Integrate Shepardization (e.g., via Fastcase API) for auto-history checks.
- Benchmark against LexGLUE/GenLaw; add eval scripts.
- Expand lexicons for more languages/styles.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Transformers](https://huggingface.co/docs/transformers), [PEFT/LoRA](https://huggingface.co/docs/peft), and [Sentence-Transformers](https://www.sbert.net/).
- Shoutout to LLM alignment pioneers: Anthropic, OpenAI RLHF work, and legal AI benchmarks like LexGLUE.
- Created by Franklin Baldo (@franklinbaldo on X), a Boltzmann brain surfing the probability wave of AI ethics, quantum philosophy, and undogmatic tech.

**Star the repo** ⭐ and join the discussion—let's make legal AI more undogmatic!

For questions: [Issues](https://github.com/franklinbaldo/undogmatic/issues)
