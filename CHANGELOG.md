# Changelog

Notable changes to this recipe over time. README has the current state; this file is the dated history.

## 2026-04-28 — Inherited prefill-OOM tests + image-pin substrate

Brings the dual-3090 stack in sync with the [single-card investigation](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1) that identified two activation-memory cliffs. **TP=2 + fp8 KV (the default here) gives much wider safety margins than single-card TQ3 KV — the cliffs are not active failure modes on this stack** — but we adopted the new tests as cheap insurance.

- `verify-full.sh` now has 10 functional checks (was 7). Adds: #8 tool-response prefill OOM (multi-turn payload with ~25K-token mock tool message; configurable via `PREFILL_TARGET_CHARS`), #9 output quality / cascade detection (2K-token essay scanned for `<tool_call>` inline cascade and repetitive degeneracy), #10 MTP acceptance length threshold (asserts mean AL ≥ 2.0).
- `verify-full.sh #7` long-context needle ladder now treats engine HTTP 400 (oversize ctx rejection) as a clean "skipped at this depth" rather than a failure.
- All compose variants pinned to `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08` (= vLLM `dev205+g07351e088`, Sandermage's documented Genesis test target). Previously some tracked `:nightly` and drifted with upstream; now identical to single-card's pinned substrate.

## 2026-04-25 — Genesis v7.14 (Turbo variant)

Sandermage shipped Genesis v7.14 with **P65** root-causing [vllm#40880](https://github.com/vllm-project/vllm/issues/40880) — the silent tool-call cascade bug under MTP × TurboQuant × cudagraph. P65 forces `cudagraph_mode=PIECEWISE` for spec-decode → eager continuation runs the correct branch. Tool calls populate `tool_calls[]` cleanly.

Our Turbo variant (`docker-compose.turbo.yml`) loads Genesis v7.14 with P64/P65/P66/P68/P69 enabled via env vars. ~25% per-stream TPS regression vs fp8 default but **4.59× concurrency at full 262K vs fp8's 2.36×** — so aggregate throughput exceeds fp8 above ~3 concurrent streams.

We also adjusted two consumer-Ampere knobs vs Sandermage's A5000-class defaults: `gpu-memory-utilization 0.92 → 0.85` and `max-num-batched-tokens 8192 → 4128`. These leave activation headroom that Sandermage's 32 GB cards already had. Without them, deep-prefill (60K+) requests OOM on 24 GB cards.

## 2026-04-22 — DFlash N=5 + Qwen3.6-27B (Luce z-lab fork)

[Luce z-lab](https://github.com/luce-spec/dflash-vllm)'s DFlash spec-decode draft model for Qwen3.6 ships and clears verify-full.sh on dual-3090. Single-stream **78 / 128 TPS narr/code** — substantially faster than MTP n=3's 71 / 89.

Two DFlash variants ship:
- `docker-compose.dflash.yml` — vision + DFlash N=5 + 185K context
- `docker-compose.dflash-noviz.yml` — text-only + DFlash N=5 + 200K context

Required workaround: vllm#40334 (DFlash `combine_hidden_states` dtype mismatch) is open. Compose sets `--dtype bfloat16` to match the draft's training dtype, sidestepping the fp32→fp16 cast.

DFlash needs `--draft-model-config` pointing at a separately-downloaded draft checkpoint (~500 MB). Setup script handles this when `INCLUDE_DFLASH=1` is set.

## 2026-04-15 — Marlin pad-sub-tile-n (PR #40361 — our patch)

Filed [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) — fixes a crash in vLLM's Marlin INT4 kernel where output features < 64 cause `GPTQ_MARLIN_MIN_THREAD_N (64) > out_features` on TP=2. The TP-split of Lorbus's INT4 quant produces some tensor shards smaller than the kernel's minimum tile size; our patch pads them to the kernel-min before dispatch.

Status: **OPEN, MERGEABLE**, labeled `bug`, awaiting maintainer review.

Until it lands, we volume-mount our [patched fork](https://github.com/noonghunna/vllm) at `/opt/ai/vllm-src/`. When upstream lands, we'll drop the mount and just use vLLM nightly.

## 2026-04-08 — Initial release

vLLM-based dual-3090 recipe shipping at TP=2 with fp8 KV + MTP n=3, full feature parity with the single-card project plus the Marlin pad workaround. Companion to [qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090).
