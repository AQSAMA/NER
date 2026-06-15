# Agent Instructions

## Project goal
Build a medical NER pipeline that trains from a blank/lightweight state and remains practical on a laptop, Google Colab, or Hugging Face Spaces/Hub workflows.

## Hard constraints
- Do not fine-tune heavy pretrained biomedical encoders for the core model.
- Prefer standard biomedical/clinical datasets over hand-written toy JSON.
- Preserve character offsets exactly through every preprocessing step.
- Keep the stack fast and modern: `uv` for environments/dependencies and `ruff` for linting/formatting.

## Preferred architecture
- Primary model: char/subword-aware BiLSTM-CRF or TCN/ID-CNN-CRF implemented in PyTorch.
- Baseline model: feature-based linear CRF for sanity checks and error analysis.
- Optional aids: licensed gazetteer features from UMLS/SNOMED CT/RxNorm/MeSH; never make them mandatory.
- Treat GLiNER/GLiNER-BioMed, spaCy pretrained pipelines, HunFlair, BioBERT, PubMedBERT, and LLM extraction as comparison baselines only unless the user explicitly changes the blank-state constraint.

## Code style
- Python 3.11+.
- Use type hints for public functions.
- Keep modules small and testable.
- Avoid hidden network calls in library code.
- Never wrap imports in try/except blocks.
- Use `pathlib.Path` for filesystem paths.
- Prefer pure conversion functions plus explicit CLI wrappers.

## Data rules
- Store raw data outside git under `data/raw/`.
- Store generated Arrow/Parquet artifacts under `data/processed/` and keep them out of git.
- Maintain one converter per corpus.
- Keep source-native labels and normalized labels separate.
- Add tests for offset round-trips, BIOUL conversion, and strict span metrics before training changes.

## Commands
- Install/sync: `uv sync --all-extras --dev`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Tests: `uv run pytest`

## Pull requests
- Summarize architectural or data-contract changes clearly.
- Include exact commands run under a Testing section.
