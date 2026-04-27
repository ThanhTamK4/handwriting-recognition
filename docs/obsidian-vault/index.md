---
type: index
tags: [type/doc, moc]
aliases: [Home, MOC, Map of Content]
---

# Handwriting Recognition — Knowledge Graph

Open the **Graph View** (Ctrl/Cmd + G) to see the full network. Nodes are color-coded by `type/*` tag via the settings in `.obsidian/graph.json`.

## Starting points

- **End-user flow** → [[Streamlit UI]] → [[Inference Pipeline]]
- **Model backends** → [[TrOCR Recognizer]] vs [[mltu CRNN Recognizer]]
- **Training** → [[Training Pipeline]] → [[train_mltu.py]] / [[train_mltu_colab.ipynb]]
- **Data** → [[IAM Words Dataset]]

## Modules (source code)

- [[streamlit_app.py]] — UI entry point
- [[recognizer.py]] — TrOCR backend
- [[mltu_recognizer.py]] — ONNX CRNN backend
- [[segment.py]] — line/word segmentation
- [[preprocess.py]] — image cleanup
- [[eval_iam.py]] — evaluation harness
- [[train_mltu.py]] — local TF training
- [[train_mltu_colab.ipynb]] — Colab GPU training

## Core ML concepts

- [[TrOCR]] · [[CRNN]] · [[CTC Loss]]
- [[ONNX Runtime]] · [[Connected Components Segmentation]]
- [[Data Augmentation]] · [[CNN Dropout]]
- [[NLP Correction]]
- [[PredictionResult]]

## Notable fixes / gotchas

- [[Double Softmax Bug]] — confidence stuck at 0.03
- [[Drive I O Bottleneck]] — 17 s/step fix
- [[Keras3 Compatibility]] — Permute + Rescaling swap
- [[CTC Not Enough Time]] — timestep vs label-length

## External docs

- [[README (project root)|Project README]]
- [[ARCHITECTURE (training)|Training Architecture]]
- [[MODELS (docs)|Model Comparison]]
