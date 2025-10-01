# ShameFeeling Token Heatmap

Small Gradio app that projects tokens onto the blended shame direction and renders a heatmap.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r apps/shame_heatmap/requirements.txt
```

The default configuration downloads the `xlm-roberta-base` model on the first run.

## Banks

Place your curated sentence banks in `apps/shame_heatmap/banks/en.json` and
`apps/shame_heatmap/banks/pt.json`. The repository ships with tiny placeholders so the
app runs end-to-end; replace the placeholder strings with your actual corpora for better
signal.

Each file should follow the structure:

```json
{
  "embarrassing_style": ["..."],
  "professional_style": ["..."],
  "anti_values": ["..."],
  "values": ["..."]
}
```

## Run

```bash
python apps/shame_heatmap/app.py
```

This launches the Gradio UI with sliders for α/δ and percentile clipping plus a language toggle.
