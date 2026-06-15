# Medical NER

A local-first Medical Named Entity Recognition project for training compact models from scratch: no heavy pretrained encoder fine-tuning, no toy hand-written datasets, and no infrastructure beyond what fits on a laptop, Google Colab, or Hugging Face workflow.

## Decision

The core model should be a **char/subword-aware BiLSTM-CRF or TCN/ID-CNN-CRF in PyTorch**, with a feature-based CRF as the first baseline. This is the most rational fit for the constraint: it is small, trainable on standard corpora, offset-friendly, and much more data-efficient than a randomly initialized transformer.

Modern options such as GLiNER/GLiNER-BioMed are worth tracking because they are lightweight and strong for open biomedical NER, but they rely on pretrained/distilled model families and should be treated as comparison baselines unless the project constraint changes.

## Datasets to target first

1. **BC5CDR / BioCreative V CDR** — first public benchmark for chemical + disease mentions in PubMed abstracts.
2. **NCBI Disease Corpus** — focused disease mention benchmark for boundary and normalization quality.
3. **MedMentions ST21pv** — broader UMLS-linked biomedical concept coverage without starting from the full sparse label set.
4. **n2c2 / i2b2 clinical corpora** — add later when data-use approval and PHI handling are in place.
5. **CRAFT / BioNLP corpora** — useful for robustness and specialized biomedical domains after the core loop works.

References: [BC5CDR](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414), [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/), [MedMentions](https://github.com/chanzuckerberg/MedMentions), [n2c2 2018 ADE](https://n2c2.dbmi.hms.harvard.edu/challenge/2018-track-2-ade-medication-extraction), [i2b2/VA 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC3168320/), [CRAFT](https://bionlp-corpora.sourceforge.net/CRAFT/), [BioNLP-ST](https://2013.bionlp-st.org/), [GLiNER-BioMed](https://arxiv.org/abs/2504.00676), [spaCy training](https://spacy.io/usage/training), [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/).

## Stack

- **Environment:** `uv`
- **Quality:** `ruff`, `pyright`, `pytest`
- **Data:** `polars`, `pyarrow`, `datasets`, `regex`
- **Modeling:** `torch`, CRF decoder, custom strict span metrics
- **CLI:** `typer`, `rich`

## Roadmap

1. Create corpus converters that emit a shared Arrow/Parquet schema: documents, entities, sentences, tokens, and BIOUL tags.
2. Add offset round-trip tests before training any model.
3. Train a linear CRF baseline on BC5CDR.
4. Train the char-CNN + BiLSTM + CRF model on BC5CDR, then NCBI Disease, then MedMentions ST21pv.
5. Report strict span micro-F1, per-label F1, boundary errors, OOV recall, latency per 1k tokens, and model size.
6. Add optional gazetteer features only after the neural baseline is reproducible.

## Quick start

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run pytest
```

Planned training commands:

```bash
uv run medical-ner prepare bc5cdr --raw-dir data/raw/bc5cdr --out-dir data/processed/bc5cdr
uv run medical-ner train configs/bilstm_crf_bc5cdr.toml
uv run medical-ner evaluate runs/bc5cdr/latest --split test
```
