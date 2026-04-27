---
type: bug
tags: [type/bug, fix/ctc]
aliases: [CTC Not Enough Time]
status: fixed
---

# CTC Not Enough Time

## Symptom

Training raised: `ctc_loss: Not enough time for target transition sequence (required: 21, available: 8)`.

## Cause

The CNN output was reshaped using **height** as the time axis, giving ~8 timesteps. CTC requires `T >= 2*L - 1`, so labels longer than 4 chars fail.

## Fix

1. Use `Permute((2, 1, 3))` so **width** becomes the time axis (see [[Keras3 Compatibility]])
2. Bump `WIDTH` from 128 → **256** so after two stride-2 downsamples we get 64 timesteps

## Related

- [[CTC Loss]], [[CRNN]]
