# DLPC Implementation Report (Draft)

## 1) Objective

Build and test a **Dynamic Latent Prefix Compression (DLPC)** prototype for OCR few-shot at inference time:

- Compress support examples into a short latent prefix.
- Inject prefix directly into deep KV attention states.
- Compare against naive multimodal few-shot baseline.

Target problem in this draft:
- Structured OCR for Vietnamese banking forms.


## 2) Implemented Architecture

### Module A: Layout-aware extraction

Implemented in `src/dlpc.py`:

- Extract visual token positions (`<|image_pad|>`) from support pass.
- Extract answer/text token positions from teacher-forcing.
- Keep top visual tokens from answer-to-image attention scores.

Purpose:
- Preserve spatially useful visual evidence while reducing token count.


### Module B: Cross-modal task compressor

Implemented two compressor modes:

- `cross`:
  - Build cross-modal routing from visual K to text K.
  - Select prefix tokens via gated text-conditioned visual scores.
- `separate`:
  - Keep visual/text memory branches separate, then concatenate.
  - Used as stability baseline.

Main knobs:
- `top_n_visual`
- `top_n_text`
- `prefix_len`
- `compressor_mode`


### Module C: Deep KV prefix injection

Implemented in patched attention forward:

- Inject prefix K/V into selected deep layers at decode-time.
- Add prefix bias through attention mask (`beta`).
- Support cacheable prefix memory (`save_prefix_cache` / `load_prefix_cache`).

Advanced controls added during debugging:

- `inject_value_prefix` (default off): key-only injection mode.
- `value_blend_lambda`: soft scale for prefix V branch.
- `value_beta_scale`: lower beta after key-structure generation phase.
- `key_phase_tokens`: token boundary between "structure phase" and "value phase".
- `positional_reset_prefix`: center prefix K across prefix length to reduce absolute-position carryover.


## 3) Files Added / Updated

Added:
- `draft/test_dlpc/src/dlpc.py`
- `draft/test_dlpc/src/run_dlpc_compare.py`
- `draft/test_dlpc/README.md`
- `draft/test_dlpc/IMPLEMENTATION_REPORT.md` (this file)

Outputs generated:
- `draft/test_dlpc/results/compare_dlpc.json`
- `draft/test_dlpc/results/compare_dlpc_safe.json`
- `draft/test_dlpc/results/compare_dlpc_key_only.json`
- `draft/test_dlpc/results/compare_dlpc_ultralight.json`
- multiple `prefix_cache*.pt` artifacts.


## 4) Experiment Timeline

### Stage 1: Initial DLPC

Typical config:
- `compressor_mode=cross`
- `target_layers=triad`
- `prefix_len=8`
- `beta=1.0`

Observed:
- Strong semantic drift.
- Wrong field mappings (e.g. numeric fields confused).
- Missing Vietnamese diacritics.
- Significant drop in schema quality vs naive.


### Stage 2: Safe injection tuning

Changes:
- Move to deeper layers.
- Lower `beta`.
- Increase `prefix_len`.
- Switch to `separate` compressor.
- Add `value_blend_lambda`.

Observed:
- Slight quality recovery.
- Latency improved vs naive.
- Still below naive on extraction fidelity.


### Stage 3: Key-only and fidelity-first

Changes:
- Key-only by default (`inject_value_prefix=false`).
- Ultra-low `beta`/`lambda`.
- Deep layers only (`30|31`).
- Larger `prefix_len=64`.
- Add generation-aware beta schedule and positional-reset option.

Observed:
- Stable speedup remains.
- Quality still plateaus below naive.
- Same failure pattern persists (semantic shift / field confusion).


## 5) Latest Result Snapshot

From `results/compare_dlpc_ultralight.json`:

- Naive:
  - `leaf_presence_rate = 1.0`
  - `leaf_non_empty_rate = 0.8421`
  - `elapsed_sec ~ 40.05`
- DLPC (ultralight):
  - `leaf_presence_rate = 0.9211`
  - `leaf_non_empty_rate = 0.7105`
  - `elapsed_sec ~ 33.24`

Interpretation:
- DLPC is consistently **faster** (~15-17% speedup).
- DLPC is still **less accurate** than naive for this OCR fidelity target.


## 6) Failure Analysis

Main recurring errors:
- Semantic shift between nearby numeric fields (e.g., ID-like values).
- Empty outputs on harder text fields.
- Diacritic degradation on Vietnamese names/phrases.
- Value-format confusion in table fields.

Likely causes:
- Prefix memory still acts as global bias in deep attention.
- Compressed support representation lacks robust field-level localization constraints.
- Injection strategy influences generation beyond structure guidance.


## 7) Current Conclusion

For this draft setup:
- The implementation validates the DLPC pipeline and provides a reproducible ablation harness.
- The method currently shows a **speed/accuracy trade-off**:
  - speed gain is clear,
  - fidelity is not yet competitive with naive baseline.

This is still useful for paper direction as:
- a context-compression + fast-adaptation prototype,
- with well-documented failure modes and measurable latency benefits.


## 8) Recommended Next Steps

1. Run with `query_gt_json` to populate `value_match_metrics` (currently null in tested logs).
2. Add ablation with `beta=0` to isolate injection harm from other pipeline steps.
3. Sweep small grid for strict fidelity-first setup:
   - `beta in {0.0, 0.005, 0.01, 0.02}`
   - `value_beta_scale in {0.05, 0.1}`
   - `key_phase_tokens in {24, 48, 72}`
   - `compressor_mode=separate`
4. Add field-level exact match and CER-style diagnostics for high-value OCR fields.


## 9) Feedback-Driven Fixes (2026-04-26)

Implemented fixes in `draft/test_dlpc/src/dlpc.py` and `draft/test_dlpc/src/run_dlpc_compare.py`:

1. Fixed closure bug in `patch_qwen3_attention_layers()`:
   - Bound per-layer loop variable via default arg `__layer_idx=layer_idx`.
   - Replaced `layer_idx` usage inside patched forward with `__layer_idx`.
   - Prevents cross-layer memory mixup and decode-step drift.

2. Added schema-only support text for prefix construction:
   - New flag `--schema-only-prefix`.
   - Prefix is built from schema skeleton instead of full support answer values.
   - Avoids direct leakage of support OCR values into compressed memory.

3. Added cache-key safety for prefix text mode:
   - Cache now records and validates `prefix_text_mode` (`answer` vs `schema`).
   - Prevents accidental reuse of stale prefix cache across incompatible modes.

Minimum rerun set requested by feedback:

- A0: `--beta 0.0` (key-only)
- A1: `--beta 0.005` (key-only)
- A2: `--beta 0.02` (key-only)
- A3: `--beta 0.005 --inject-value-prefix --value-blend-lambda 0.005`

Output logs:
- `draft/test_dlpc/results/compare_dlpc_a0_noop.json`
- `draft/test_dlpc/results/compare_dlpc_a1_key_weak.json`
- `draft/test_dlpc/results/compare_dlpc_a2_key_moderate.json`
- `draft/test_dlpc/results/compare_dlpc_a3_value_very_weak.json`

Observed summary from these four runs:
- DLPC latency remains lower than naive baseline.
- `beta=0.0` still differs from naive output quality, indicating side effects beyond beta-only steering.
- Across A0-A3, schema-level rates are stable in this sample:
  - DLPC `leaf_presence_rate`: `0.9211`
  - DLPC `leaf_non_empty_rate`: `0.7105`
  - Naive baseline: `1.0` and `0.8421`

