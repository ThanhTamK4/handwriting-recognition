---
type: bug
tags: [type/bug, fix/compatibility, platform/colab]
aliases: [Keras3 Compatibility]
status: fixed
---

# Keras 3 Compatibility

Colab ships with TensorFlow 2.16+ where Keras 3 is stricter about what can appear in a Functional model graph.

## Issues encountered

### 1. KerasTensor cannot be input to a tf function

`tf.transpose(x, perm=[0,2,1,3])` on a symbolic KerasTensor raises `ValueError`.

**Fix:** Use `Permute((2, 1, 3))` — a proper Keras layer with equivalent semantics.

### 2. Lambda division

`Lambda(lambda img: img / 255.0)` triggers the same issue.

**Fix:** Use `Rescaling(1.0 / 255.0)`.

### 3. `workers=` removed from fit

`TensorFlowTrainer.fit()` no longer accepts `workers` keyword.

**Fix:** Drop the argument.

### 4. DataProvider not recognized

mltu's DataProvider doesn't subclass `keras.utils.Sequence`. Keras 3 requires a `Sequence` for `fit()` input.

**Fix:** `KerasSequenceProvider(Sequence)` wraps the provider.

## Related

- [[train_mltu_colab.ipynb]], [[CRNN]]
