# Quick DKI test (draft)

This folder is a minimal prototype to compare:
- Naive multimodal few-shot
- DKI-KV (demo pass -> select visual tokens -> extract KV -> inject KV at mid layers)

Model default:
- `Qwen/Qwen3-VL-4B-Instruct`

## Install

```bash
pip install torch torchvision accelerate pillow tqdm
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
pip install pypdfium2
```

## Run

From repo root:

```bash
python draft/test_dki/src/run_compare.py --shots 1
```

Optional:

```bash
python draft/test_dki/src/run_compare.py \
  --shots 1 \
  --target-layers "mid" \
  --top-n 8 \
  --beta 0.2 \
  --max-new-tokens 768 \
  --output draft/test_dki/results/compare_log.json
```

With query ground-truth JSON:

```bash
python draft/test_dki/src/run_compare.py \
  --shots 1 \
  --query-gt-json "/path/to/query_gt.json"
```

## Sweep search (recommended)

```bash
python draft/test_dki/src/sweep_configs.py \
  --shots 1 \
  --top-n-list "4,8,12" \
  --beta-list "0.05,0.1,0.2,0.3" \
  --layer-spec-list "mid,triad,16|20|24" \
  --output draft/test_dki/results/sweep_results.json
```

## Notes

- This is intentionally short and simple for quick idea testing.
- Output is parsed then aligned to schema for fair comparison.
- `sweep_configs.py` runs Naive once, then sweeps DKI configs and ranks results.
