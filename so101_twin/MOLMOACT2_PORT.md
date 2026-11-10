# Porting the pi0.5 recorder / patcher to MolmoAct2-BimanualYAM

Read from `modeling_molmoact2.py` and `config.json` of allenai/MolmoAct2-BimanualYAM on 2026-11-06.
Nothing here has been run against weights yet (the local GPU is shared with the live agents; the
checkpoint is ~22 GB and needs ~16-20 GB in bf16 for inference, which fits the 64 GB M3 Max when the
pi0.5 server is not loaded).

## Shapes

| part     | class                                  | layers | hidden         | notes                                                         |
| -------- | -------------------------------------- | ------ | -------------- | ------------------------------------------------------------- |
| vision   | `MolmoAct2VisionTransformer` (SigLIP2) | 27     | 1152, MLP 4304 | 729 patches per crop, 14 px; pooled 2x2 by `image_pooling_2d` |
| language | `MolmoAct2TextModel` (Qwen3-4B)        | 36     | 2560, MLP 9728 | 32 q heads, 8 kv heads, head_dim 128                          |
| expert   | `ActionExpert` (DiT)                   | 36     | 768, 8 heads   | flow matching, 10 steps, 30-step chunk, 32-D padded action    |

## Hook points (what the pi0.5 `Pi05Runtime` hooks become)

| pi0.5                                   | MolmoAct2                                           | module / attribute                                                                         |
| --------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| siglip `mlp.fc2` input                  | vision MLP hidden                                   | `ViTMLP` second linear input                                                               |
| backbone `mlp.down_proj` input          | language MLP hidden                                 | `LanguageModelMLP.ff_out` input (after silu-gate)                                          |
| expert `mlp.down_proj` input            | expert MLP hidden                                   | `ActionExpertMLP.down_proj` input                                                          |
| backbone K/V (runtime `kv_patch`)       | language per-layer K, V                             | `MolmoAct2Attention.att_proj` output split by `fused_dims` into (q, k, v)                  |
| expert reads backbone K/V via attention | expert reads language K/V via **named projections** | `ActionExpert.context_k_proj`, `context_v_proj` applied per layer in `_prepare_kv_context` |

The last row is the adapter's home. `_prepare_kv_context(encoder_kv_states)` loops over the 36 language
layers, projects each layer's (k, v) with `context_k_proj` / `context_v_proj`, applies the expert's
`k_norm`, and hands them to `ActionExpertCrossAttention` as `kv_k`, `kv_v`. A LoRA on `context_k_proj`
and `context_v_proj`, masked to the wrist-camera token positions, touches nothing else in the model.

## Token layout

Images arrive as `(batch, num_crops, num_patch, pixels)`; the language sequence carries
`image_start_token_id` / `image_patch_id` / `image_col_id` / `image_end_token_id` markers per image, so
the per-camera token slices can be recovered from `input_ids` the same way the pi0.5 runtime builds
`backbone_groups`. YAM order is fixed: top, left, right (`norm_stats.json`, tag `yam_dual_molmoact2`).
State is tokenised (256 bins per value), so there is a `state` text group as in pi0.5's discrete-state
configs. The action-token span is masked from the expert's conditioning.

## What stays the same

Recording per unit mean/max over a token group, paired contrast against a null, held-out halves, the
attribution-patching estimate (score = delta activation x gradient of the action change), the noise
floor from re-rolling the flow noise, the random-unit control, and the skeptic. Only module names, the
number of layers (36 everywhere), and the token-group builder change.

## Open items

- Confirm `ActionExpertCrossAttention._as_heads` reshape so per-head K/V patching lines up (8 heads x 96).
- `MolmoAct2-BimanualYAM` is bf16-loadable only with the servers' monkey-patch (README); check the
  snapshot date before relying on upstream.
- LeRobot port (`lerobot/policies/molmoact2`) wraps the same HF module; hook there if training uses it.

## Build status (2026-11-07 00:10, session inference-testing-72)

- `neuroscope/molmoact2/runtime.py` implements the table above: MLP-hidden hooks on vision / language / expert,
  per-camera token groups from `input_ids`, pi0.5-format record and edit requests, seed-controlled flow noise, and
  K/V patching by rows and layers through `generate_actions_from_inputs`'s `encoder_kv_states` path.
- Own venv (`neuroscope/molmoact2/.venv`, Python 3.12, transformers 5.3.0) because the remote code needs 5.x.
- Weight-free dry run passed (705 prefix tokens; 196 per camera; 14 state tokens). `selftest.py` runs once the
  21.8 GB download lands. Details in `neuroscope/molmoact2/README.md`.
- 00:25 the user pointed at `allenai/MolmoAct2-SO100_101` (an official single-arm SO-100/101 fine-tune I had missed): the
  runtime now defaults to it (`runtime.MODELS`), two cameras top/side, 6-D state and action, tag `so100_so101_molmoact2`.
  Same remote code as BimanualYAM byte for byte. Consequence for the plan: the "fine-tune on sim demos" step for a
  *single* SO-101 is unnecessary for localisation work; the bimanual rig still has no checkpoint.
