# DLPC quick test (draft)

This folder is a quick prototype for:
- Naive multimodal few-shot
- DLPC style flow:
  1) Layout-aware token extraction (visual patch tokens + answer text tokens)
  2) Cross-modal task compressor into short latent prefix
  3) Deep KV prefix injection at middle/deep layers

Default model:
- `Qwen/Qwen3-VL-4B-Instruct`

## Run

From repo root:

```bash
python draft/test_dlpc/src/run_dlpc_compare.py \
  --shots 1 \
  --target-layers 30\|31 \
  --top-n-visual 48 \
  --top-n-text 24 \
  --prefix-len 64 \
  --compressor-mode separate \
  --beta 0.02 \
  --value-blend-lambda 0.01 \
  --value-beta-scale 0.1 \
  --key-phase-tokens 48 \
  --prefix-cache draft/test_dlpc/results/prefix_cache.pt \
  --output draft/test_dlpc/results/compare_dlpc.json
```

Ultra-safe key-only mode (recommended for OCR fidelity and Vietnamese diacritics):

```bash
python draft/test_dlpc/src/run_dlpc_compare.py \
  --compressor-mode separate \
  --target-layers 30\|31 \
  --beta 0.02 \
  --value-blend-lambda 0.01 \
  --value-beta-scale 0.1 \
  --key-phase-tokens 48 \
  --rebuild-prefix \
  --prefix-cache draft/test_dlpc/results/prefix_cache_key_only.pt \
  --output draft/test_dlpc/results/compare_dlpc_key_only.json
```

To enable Value injection as well (stronger, riskier):

```bash
python draft/test_dlpc/src/run_dlpc_compare.py \
  --inject-value-prefix
```

## Ablation: cross-modal vs separate compression

Cross-modal compressor:

```bash
python draft/test_dlpc/src/run_dlpc_compare.py \
  --compressor-mode cross \
  --rebuild-prefix \
  --prefix-cache draft/test_dlpc/results/prefix_cross.pt \
  --output draft/test_dlpc/results/compare_dlpc_cross.json
```

Separate compressor:

```bash
python draft/test_dlpc/src/run_dlpc_compare.py \
  --compressor-mode separate \
  --rebuild-prefix \
  --prefix-cache draft/test_dlpc/results/prefix_sep.pt \
  --output draft/test_dlpc/results/compare_dlpc_separate.json
```

## Notes

- `cross` tries to compress visual tokens using text-conditioned routing scores.
- `separate` is a baseline that concatenates visual + text memory without cross-modal routing.
- `value_blend_lambda` controls soft prefix influence on injected value memory (small = safer).
- `value_beta_scale` lowers prefix strength after key-structure phase during decoding.
- `key_phase_tokens` defines how many initial generated tokens use full `beta`.
- Prefix key centering ("positional reset") is enabled by default; disable with `--no-positional-reset-prefix`.
- By default, script uses key-only injection (`--inject-value-prefix` is off).
- This is a draft inference-time fast-adaptation test, not a full training pipeline.

