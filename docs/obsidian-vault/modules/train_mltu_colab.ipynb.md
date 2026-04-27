---
type: module
tags: [type/module, layer/training, platform/colab]
aliases: [Colab notebook, training/train_mltu_colab.ipynb]
path: training/train_mltu_colab.ipynb
---

# train_mltu_colab.ipynb

Google Colab GPU version of [[train_mltu.py]]. Structured as 12 clean phases, each a markdown heading + one code cell.

## Phases

1. **Setup** — pip install, GPU assertion
2. **Parameters** — all knobs in one cell
3. **Mount Drive & stage data locally** — copies dataset to `/content/IAM_Words_local` to solve the [[Drive I O Bottleneck]]
4. **Parse dataset** — `load_samples()` from `words.txt`
5. **Sample visualization** — 2×4 matplotlib grid of word crops
6. **Write configs.yaml**
7. **DataProvider + augmentation** — uses [[Data Augmentation|mltu augmentors]], wraps in `KerasSequenceProvider`
8. **Model definition** — [[CRNN]] with [[CNN Dropout]] and Keras-3-compatible `Permute` + `Rescaling` ([[Keras3 Compatibility]])
9. **Compile & train** — `CSVLogger` for later plotting; `Model2onnx` auto-export
10. **Training curves** — matplotlib plot of loss/CER/WER
11. **Export & verify** — file-size report
12. **Copy to Drive** — move artifacts from local SSD → Drive for download

## Expected timing

With data on local SSD + T4 GPU: ~30 min for 50 epochs. Without it: ~11 h.

## Related

- [[train_mltu.py]], [[Training Pipeline]]
- [[Drive I O Bottleneck]], [[Keras3 Compatibility]]
- [[Data Augmentation]], [[CNN Dropout]]
