---
type: bug
tags: [type/bug, fix/performance, platform/colab]
aliases: [Drive I/O Bottleneck]
status: fixed
---

# Drive I/O Bottleneck

## Symptom

Training on Colab clocked **17 s/step** (1357 steps/epoch × 50 epochs ≈ 11 h). Expected: < 1 s/step.

## Cause

`DataProvider` read every PNG directly from the Google Drive mount, which uses a slow network filesystem. Each batch of 64 images triggered 64 Drive requests.

## Fix ([[train_mltu_colab.ipynb]] phase 3)

`shutil.copytree` or `tarfile.extractall` copies the dataset to `/content/IAM_Words_local` (Colab VM's local SSD) **once** at the start. Subsequent reads hit local disk → ~10× speedup.

```python
if count_pngs(LOCAL_DATA_DIR) == 0:
    with tarfile.open(str(drive_tgz), 'r:gz') as tar:
        tar.extractall(path=str(target))
```

## Related

- [[train_mltu_colab.ipynb]], [[Training Pipeline]]
