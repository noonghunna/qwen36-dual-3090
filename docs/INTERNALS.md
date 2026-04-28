# Internals — how the dual-3090 stack actually works

Deep-dive companion to the [README](../README.md). Read this when you want to understand:

- Why the Marlin pad-sub-tile-n patch is necessary (and what we want to land upstream)
- How TP=2 actually splits the model and why allreduce is the bottleneck on PCIe
- What DFlash N=5 is doing differently than MTP and why it wins on code
- Current upstream status of the patches we depend on

If you just want to use the stack, the README is enough.

---

## Why TP=2 doesn't double single-stream TPS

A common expectation: "I'm doubling the GPUs, I should get double the speed." Reality on PCIe-only consumer Ampere: per-stream TPS gain from TP=2 is small (~5%).

**Why:** decode on a single batch is memory-bandwidth-bound, not compute-bound. Each forward step reads weights from VRAM, computes, writes outputs. With TP=2:

- Each card holds half the weights (good — halves the per-card memory bandwidth requirement)
- But after each layer's matmul, partial results must combine via NCCL **all-reduce** across both cards
- All-reduce on **PCIe Gen 4** (~32 GB/s practical) is ~3-5× slower than NVLink (~600 GB/s on H100, ~200 GB/s on 3090 with bridge)
- All-reduce overhead approximately cancels the memory-bandwidth halving

Net: single-stream TPS per card ≈ same as single-card TPS. The TP=2 win is **concurrent** throughput — when 2-4 streams run simultaneously, all-reduce overhead amortizes across the larger batch and aggregate scales near-linearly to ~4 streams.

This is why we measure both single-stream TPS and "concurrent throughput at full ctx" — the latter is what you'd get serving 4 simultaneous users.

**When this would matter less:**
- If you have NVLink (e.g., A6000 + bridge, or H100 SXM): single-stream gain from TP=2 is closer to 1.6-1.8×
- If you're running attention-bound prefill, not decode, the TP win is bigger (prefill is more compute-bound)

---

## The Marlin pad-sub-tile-n patch (vllm#40361)

**Symptom:** vLLM crashes during model-load on TP=2 with:
```
RuntimeError: GPTQ_MARLIN_MIN_THREAD_N (64) > out_features
```

**Root cause:** vLLM's Marlin INT4 GEMM kernel requires `out_features ≥ GPTQ_MARLIN_MIN_THREAD_N` (64). Lorbus's Qwen3.6-27B-int4-AutoRound has tensors with `out_features = 128` for some MTP-related layers. With TP=2, those split into 64 per card — exactly at the boundary. With TP=4 (would-be) or any model with smaller layers, the split would go below 64 and crash.

The 27B + TP=2 case lands at the boundary. Some users with slightly different vLLM nightlies, slightly different layer sizes, or slightly different combinations of patches end up hitting `out_features = 32` or similar after the split → instant crash.

**Our fix (PR #40361):** Pad the tensor's `out_features` dimension up to the kernel minimum (64) before dispatch, then slice the result back. Trade-off: one extra memcpy per Marlin call. Measured cost: <0.5% TPS overhead. Gain: works on any TP-shape that produces sub-tile layers.

```python
# Pseudo-code of the patch
if out_features < GPTQ_MARLIN_MIN_THREAD_N:
    pad_to = GPTQ_MARLIN_MIN_THREAD_N
    weight_padded = pad_tensor(weight, pad_to)
    output = marlin_gemm(weight_padded)
    output = output[..., :out_features]  # slice back
else:
    output = marlin_gemm(weight)         # original fast path
```

Status: PR is **OPEN, MERGEABLE**, labeled `bug`, sitting in maintainer queue. When it lands, drop the `/opt/ai/vllm-src/` mount from the compose. Until then, the volume-mount path lets users get the fix without forking themselves.

---

## DFlash N=5 — what it does differently than MTP

**MTP (default):** Qwen3.6 ships an integrated 1-token MTP head trained jointly with the main model. We run it with `num_speculative_tokens=3` — predicts 3 tokens forward, verifies, accepts what's correct. AL ~3.4 means 3.4 tokens accepted per spec-decode step on average.

**DFlash (z-lab fork):** A separately-trained, larger draft model (5-token forward window) optimized specifically for **code workloads**. Runs in parallel with the main model on the same GPUs. AL ~4.7 on code prompts (vs 3.4 for MTP).

Why DFlash beats MTP on code:
- **Bigger draft model** = more accurate predictions per position (vs MTP's tiny ~1B head)
- **Trained on code-heavy data** = higher AL on structured tokens (function names, syntax, brackets)
- **N=5 vs N=3** = more shots per verify step when accept rate is high (which it is on code)

Cost of DFlash:
- Draft model adds ~500 MB VRAM (negligible on dual-3090)
- ~20% extra compute per forward (the parallel draft) — but accept rate gain pays it back
- Vision support is preserved in `dflash.yml`, dropped in `dflash-noviz.yml` (text-only path slightly faster + 200K vs 185K ctx)

When MTP wins:
- Narrative / chat workloads (code-token bias of DFlash doesn't help, AL drops to ~2.8)
- Tool-call generation (structured but different patterns than code)
- When you need full 262K ctx (DFlash caps at 185K with vision / 200K without due to draft-model VRAM)

So we ship both: default = fp8 + MTP for breadth, DFlash for code-heavy peak performance.

---

## Why we configure consumer-Ampere knobs differently than Sandermage's reference

Sandermage tests on 2× A5000 (32 GB each). Their default `gpu-memory-utilization=0.92` and `max-num-batched-tokens=8192` work great there because A5000 has 33% more VRAM. On 24 GB 3090s, those defaults don't leave enough activation headroom for chunked-prefill peaks at 60K+ context.

Our adjustments in the Turbo variant:
- `gpu-memory-utilization 0.92 → 0.85` — frees ~1.7 GB activation headroom per card
- `max-num-batched-tokens 8192 → 4128` — smaller chunked-prefill chunks, smaller activation peak per forward

Without these, deep-prefill (60K+) requests OOM. With them, the Turbo variant runs cleanly at full 262K with 4-stream concurrency.

For Sandermage's documented numbers on his A5000 setup, see his [MODELS.md](https://github.com/Sandermage/genesis-vllm-patches). For our adjusted numbers on 3090s, see this repo's bench section.

---

## Upstream status

| Issue | Status | Notes |
|---|---|---|
| [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) | **OPEN, MERGEABLE** | Our Marlin pad-sub-tile-n patch. Drops out as a dependency when this lands. |
| [vllm#40334](https://github.com/vllm-project/vllm/pull/40334) | **OPEN, NOT MERGED** | DFlash `combine_hidden_states` dtype mismatch fix. Workaround in our DFlash composes: `--dtype bfloat16`. |
| [vllm#40831](https://github.com/vllm-project/vllm/issues/40831) | Closed (ngram + MTP) | TurboQuant × spec-decode corruption. Closed for ngram via Sander's v7.13 + #40875; closed for MTP via Genesis v7.14 P65 (used in Turbo variant). |
| [vllm#40880](https://github.com/vllm-project/vllm/issues/40880) | Worked around (Genesis v7.14) | MTP × TurboQuant × cudagraph. P65 PIECEWISE downgrade. Proper P67 multi-query Triton kernel TBD. |
| [vllm#40914](https://github.com/vllm-project/vllm/pull/40914) | **OPEN** | Sandermage's upstream synthetic-args trick for spec-decode K+1 verify. Would close P67 design gap. We bench against the patched build occasionally; cross-rig data shared on the PR. |

When PR #40361 lands, we drop the `/opt/ai/vllm-src/` mount. When PR #40914 lands, the Turbo variant's TPS regression vs fp8 narrows substantially (P65 PIECEWISE downgrade can be replaced with a kernel-level fix).

---

## See also

- [docs/USE_CASES.md](USE_CASES.md) — per-workload guides (multi-tenant, frontier ctx, RAG, code, vision, advanced)
- [CHANGELOG.md](../CHANGELOG.md) — dated history
- [single-card INTERNALS.md](https://github.com/noonghunna/qwen36-27b-single-3090/blob/master/docs/INTERNALS.md) — same model on one GPU; different bug surface, same root patches
- [single-card docs/engines/](https://github.com/noonghunna/qwen36-27b-single-3090/tree/master/docs/engines) — comparison of vLLM / SGLang / llama.cpp for this model class
