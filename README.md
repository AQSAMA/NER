# Biomedical NER BiLSTM-CRF Workspace

A small, readable biomedical Named Entity Recognition (NER) project for pharmacy/medical research coursework. It trains a **custom PyTorch BiLSTM-CRF** model on the Hugging Face `tner/bc5cdr` dataset, which contains Chemical/Drug and Disease entities.

This is intentionally **not SOTA**. It is designed to be understandable, runnable from a terminal/Termux/Colab, and easy to discuss in a graduation research project.

## Quick start

```bash
python setup_ner_workspace.py --write-only
python -m pip install -r requirements.txt
python train.py --epochs 3 --batch-size 16 --max-train-samples 4000
python predict.py --checkpoint checkpoints/best_model.pt
```

Or let the master script write files, install packages, and start training:

```bash
python setup_ner_workspace.py --epochs 3 --batch-size 16 --max-train-samples 4000
```

## Termux notes

Install Python and required build tools first if needed:

```bash
pkg update
pkg install python clang rust git
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On low-memory devices, use smaller settings:

```bash
python train.py --epochs 2 --batch-size 4 --embedding-dim 64 --hidden-dim 128 --max-train-samples 1000
```

## Colab notes

Upload this folder or clone your repository, then run:

```python
!pip install -r requirements.txt
!python train.py --epochs 5 --batch-size 32
!python predict.py --checkpoint checkpoints/best_model.pt
```

## Files

- `setup_ner_workspace.py` — one-command master script that writes all project files.
- `data.py` — downloads `tner/bc5cdr`, builds vocab/tag maps, pads batches.
- `model.py` — custom BiLSTM-CRF architecture.
- `metrics.py` — entity-level precision, recall, and F1.
- `train.py` — training/evaluation/checkpoint pipeline.
- `predict.py` — interactive CLI inference.
