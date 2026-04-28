# Use cases — what to run, what to expect, what trips people up

Practical guide matched to common dual-card workloads. For each: which compose, why it fits, gotchas, limitations, tuning levers.

> If you're new to the stack, read the [README](../README.md) first. This doc assumes you've got it running and want to dial it in for your specific workflow.

---

## Quick map

| Your workload | Boot this | Streams | Per-stream TPS | Aggregate |
|---|---|---|---|---|
| General chat / single-user (262K ctx, max-feature) | `docker-compose.yml` (default) | 1 | 71 narr / 89 code | 71 / 89 |
| Multi-tenant chat (2-3 users) | `docker-compose.yml` (default) | 2-3 | ~60 each | ~180 aggregate |
| Many-user / agent fleet (4 streams at 262K) | `docker-compose.turbo.yml` | 4 | ~50 each | ~200 aggregate |
| Code-heavy single-stream (peak speed) | `docker-compose.dflash.yml` | 1 | **78 / 128** | 78 / 128 |
| Long-doc / RAG text-only | `docker-compose.dflash-noviz.yml` | 1 | 77 / 124 | 77 / 124 |
| Frontier 262K with vision | `docker-compose.yml` (default) | 1 | 71 / 89 | 71 / 89 |
| Genesis P67 / vllm#40914 experiments | `docker-compose.p67-bench.yml` | 1 | varies | n/a (research) |

---

## General chat / Q&A (single user)

**Best for:** Single-user ChatGPT-replacement, browser UIs (Open WebUI, LibreChat), intermittent use.

**Recommended:** [`docker-compose.yml`](../compose/docker-compose.yml) (default — fp8 + MTP n=3, 262K, vision).

**Why default over single-card:** You can serve up to 2 concurrent chat requests at full ctx without splitting context across them — useful if you have a partner/teammate on the same backend. Single-card maxes at one user.

**Why NOT default if you only do single-user chat:** Single-stream TPS is virtually identical to single-card (71 narr vs 51 narr). The dual-card win is concurrency, not per-request speed. **If you'll only ever have one chat at a time, the [single-card setup](https://github.com/noonghunna/qwen36-27b-single-3090) is just as fast and half the cost.**

**Gotchas:**
- Cold prefill at 262K is genuinely slow (3-5 minutes). Pre-warm prefix cache before user-facing use.
- Concurrency at full 262K is 2 streams. Going beyond would split context (each stream gets ~131K). Use Turbo variant for 4 streams at full 262K.

**Limitations:**
- No way to push past 262K — that's Qwen3.6's natural max. Hard ceiling.

---

## Multi-tenant chat / agent fleet

**Best for:** Team setups where multiple people / agents hit the same backend simultaneously. Production-like serving on dev hardware.

**Recommended:** [`docker-compose.turbo.yml`](../compose/docker-compose.turbo.yml) (TurboQuant 3-bit + Genesis v7.14 + 262K + vision).

**Why turbo over default:** TurboQuant 3-bit KV is 3-bit instead of fp8's 8-bit, so the same VRAM holds ~9× more KV. **At full 262K, default fits 2 streams; turbo fits 4 streams.** Aggregate throughput at 4 concurrent streams is ~200 TPS, beating default's ~180 TPS at 2 streams.

**Trade-off:** Per-stream TPS is ~25% lower (58 / 69 vs 71 / 89). Genesis v7.14 P65 forces `cudagraph_mode=PIECEWISE` for spec-decode → eager continuation runs the correct branch but pays a per-step cost. The crossover where turbo wins on aggregate is ~3 concurrent streams.

**Gotchas:**
- Turbo loads Genesis v7.14 patches at boot. Verify via container logs: `Genesis Results: 27 applied, 36 skipped, 0 failed`.
- Higher concurrency = higher VRAM pressure. We adjusted `gpu-memory-utilization=0.85` and `max-num-batched-tokens=4128` from Sandermage's A5000 reference defaults; don't push these without testing.
- Aggregate TPS scales near-linearly to ~4 streams; beyond that, the KV pool starts splitting context and per-stream TPS drops.

**Limitations:**
- Tool-call cascade can re-emerge if Genesis doesn't apply (rare, but check the boot log markers).
- Multi-tenant SLAs need monitoring — set up Prometheus or similar to watch p99 latency at production volumes.

**Tuning levers:**
- For agents that don't need vision: drop `--language-model-only` for ~1 GB extra activation budget per card.
- For shorter conversations (lots of agents, each short): lower `max-model-len` to 96K-128K. Frees more KV pool for additional streams.
- Set `prompt_lookup_min=8` if you're using ngram spec-decode instead of MTP (relevant only if testing alternative spec-dec methods).

---

## Code-heavy workloads (peak single-stream TPS)

**Best for:** Coding agents (Cline, Roo, Cursor) where you want maximum throughput on code generation, single user.

**Recommended:** [`docker-compose.dflash.yml`](../compose/docker-compose.dflash.yml) (DFlash N=5 + 185K + vision) or [`docker-compose.dflash-noviz.yml`](../compose/docker-compose.dflash-noviz.yml) (text-only, 200K).

**Why DFlash over default:** DFlash N=5 is a custom 5-token draft model from z-lab specifically optimized for code. AL on code prompts is ~4.7 (vs MTP's 3.4 → 3.5). Translation: ~44% faster code TPS (128 vs 89), ~10% faster narrative TPS (78 vs 71).

**Why NOT DFlash:**
- 185K context vs default's 262K — DFlash's draft model adds ~500 MB VRAM that has to come from somewhere.
- Single-stream only. The DFlash draft + main model running in parallel uses up the concurrency budget.
- Vision is supported in `dflash.yml` but not in `dflash-noviz.yml` (which trades vision for 200K instead of 185K).

**Gotchas:**
- DFlash's draft model is downloaded separately. `setup.sh` handles it when `INCLUDE_DFLASH=1`.
- Compose sets `--dtype bfloat16` to work around vllm#40334 (DFlash combine_hidden_states fp32→fp16 cast bug). Don't change the dtype.
- Code AL (acceptance length) is workload-sensitive — verify on YOUR agent's typical prompts. We measured on quicksort + small transformer code; your milage will vary on niche languages.

**Limitations:**
- Multi-tenant doesn't work well — DFlash's parallel draft model makes concurrent serving inefficient. Pick the default for >1 stream.
- Narrative AL is lower than MTP's narrative AL — DFlash's training bias toward code hurts open-prose generation slightly.

**Tuning levers:**
- For more deterministic code: `temperature=0.0`. Spec-decode AL drops slightly (less variability for the draft to nail) but reproducibility is worth it.
- For batch code review (running a pipeline of small code prompts): wrap the whole pipeline in a single chat to maximize prefix-cache hits.

---

## Long-document analysis / RAG at scale

**Best for:** Loading large documents (whole books, codebases, log dumps) and querying them. Single-shot summarization. RAG pipelines with big retrievals.

**Recommended:** Default for vision-bearing docs (PDFs with images, mixed text+chart). Tools-text style — actually for THIS repo, [`docker-compose.dflash-noviz.yml`](../compose/docker-compose.dflash-noviz.yml) (200K + DFlash + text-only) for pure-text docs.

**Why DFlash text-only:** 200K is the longest context any of the dual variants offer. DFlash's higher AL means responses come back faster.

**Why default for mixed-modal docs:** 262K + vision + MTP n=3 = the most context, full feature support, slightly slower TPS but irrelevant when the bottleneck is cold prefill.

**Gotchas:**
- **Cold prefill at 200K-262K is GENUINELY slow.** Plan 3-5 minutes for the first request. Subsequent queries against the same prefix hit the cache.
- **Recall quality at 100K+ tokens degrades** — like most current LLMs, attention thins toward the document middle. **Test recall on YOUR actual corpus before trusting.** Our `verify-full.sh #7` needle ladder tests up to 90K but recall at 180K-240K is plausible-but-untested.
- **Prefix cache saves you.** Send a dummy request loading the doc; subsequent real queries skip the cold prefill entirely. Build this into your RAG pipeline.

**Limitations:**
- 262K is the model's natural max. Documents bigger than that need chunking + multi-shot reasoning.
- Single-stream throughput. For batch summarization of many docs, consider serially running one doc per request and using prefix cache between sibling questions.

**Tuning levers:**
- Pre-warm the prefix cache during off-peak: cron a request that loads the day's docs → real users hit warm cache.
- For multi-doc analysis: load all docs once into one long system prompt; ask multiple questions; prefix cache wins on every Q after the first.
- Lower `--max-num-batched-tokens` to 4128 (already default in non-Turbo variants) for smaller activation peaks during prefill — reduces OOM risk on giant docs.

---

## Vision-heavy workloads

**Best for:** Multimodal pipelines where users frequently send images alongside text. Code-screenshot review, document OCR-style tasks, visual Q&A.

**Recommended:** [`docker-compose.yml`](../compose/docker-compose.yml) (default — fp8 + 262K + vision tower active).

**Why default:** Vision tower is small (~1 GB VRAM), but adds activation overhead. Default leaves comfortable headroom. Turbo also has vision but with the 25% per-stream TPS cost; only worth it for multi-tenant.

**Gotchas:**
- Each image consumes 640-1280 tokens at default resolution. 5-image conversations chew through several thousand tokens before any text gets processed.
- High-resolution images (2048×2048+) get downsampled internally; details below the vision-tower's resolution won't be processed.
- Vision quality is good for charts, screenshots, natural images. Less reliable for OCR on dense text — Qwen team didn't optimize for OCR specifically.

**Limitations:**
- No image generation. For "draw a picture" use a separate diffusion model (ComfyUI / SD).
- DFlash variant is vision-capable but the draft model's bias toward code hurts vision-task quality. Default is the right pick for vision.

---

## Frontier context (192K-262K) for whole-codebase / book-length workflows

**Best for:** Loading a whole codebase, a research paper, or a book in one shot. Multi-step reasoning over very large inputs.

**Recommended:** [`docker-compose.yml`](../compose/docker-compose.yml) (default — already at 262K).

**Why default:** You're already at the model's natural max (262K). The only reason to switch is if you need DFlash's faster code TPS and can live with 185K-200K (DFlash variants).

**Gotchas:**
- **Cold prefill at 262K = real time.** Budget 5+ minutes for the first request. Make this your async/cron pre-warm step, not an interactive UX.
- **Recall degrades past 100K.** This is universal across LLMs at this size class. Verify on your actual workload.
- **Multi-tenant at full 262K** caps at 2 streams (default) or 4 streams (turbo). If you have 10 users, KV pool fragmentation hurts; consider lowering `max-model-len` and serving more concurrent at smaller ctx.

**Limitations:**
- 262K is the model's hard ceiling. Larger workloads need a different model class.
- Recall quality drops noticeably between 150K and 262K. Use the upper range only when the workload genuinely needs it.

**Tuning levers:**
- Aggressive prefix caching is the entire game at this ctx size. Architect requests so the first N tokens are stable across many queries.
- Consider DFlash text-only at 200K instead — modest ctx loss, substantial TPS gain, simpler memory profile.

---

## Advanced mode — DFlash tuning, P67 experiments, contributing

**Best for:** Folks running spec-decode bench experiments, validating Sandermage's PRs, contributing back to upstream / Genesis.

The `docker-compose.p67-bench.yml`, `docker-compose.p67-bench-dflash.yml`, and `docker-compose.p67-bench-35b-dflash.yml` composes are designed for cross-rig validation of vllm#40914 (P67 / synthetic-args trick) on dual-card hardware. They're not user-facing — research artifacts.

If you're doing this kind of work:
1. Read [docs/INTERNALS.md](INTERNALS.md) for what each Genesis patch does
2. Use `bench.sh` with multiple runs (`RUNS=10 WARMUPS=3`) for tight CV measurements
3. Capture VRAM, GPU util, and AL alongside TPS in your reports
4. Cross-link to Sandermage's MODELS.md for A5000-class baselines

**Things to try:**
- **Different DFlash N values** — N=3, N=5, N=7. We landed on N=5 for code; the sweet spot may shift with model updates.
- **Genesis P67 / P78 enabled** — `GENESIS_ENABLE_P67=1 GENESIS_ENABLE_P78=1` before booting. Bench-only; not on the production composes.
- **Mixing turbo + DFlash** — possible in principle (TurboQuant KV + DFlash draft) but currently untested by us. Genesis v7.14 + DFlash interactions need verification.

**Things that won't work:**
- TP=4 on this hardware — only 2 cards.
- Pipeline parallelism (PP) — vLLM supports PP but it's overhead on a 27B model that fits in TP=2 + fp8. Don't bother on this scale.

---

## When something is wrong

- **Container dies at boot with `GPTQ_MARLIN_MIN_THREAD_N (64) > out_features`** — vllm#40361 patch didn't apply. Check `/opt/ai/vllm-src/` exists and is mounted in the compose. Re-clone if missing.
- **Container dies during DFlash boot** — vllm#40334 dtype mismatch. Verify compose has `--dtype bfloat16`. Don't change to fp16 or fp32.
- **Tool calls return `<tool_call>` as plain text** — Turbo variant only. Genesis v7.14 didn't apply. Check `Genesis Results: 27 applied`.
- **OOM during prefill at 60K+** — Turbo variant on 24 GB cards. Verify compose has `gpu-memory-utilization=0.85` (not 0.92). Or switch to default fp8 variant.
- **Per-stream TPS lower than the README claims** — `bench.sh` with 3+ warmups + 5 measured runs first. Run-to-run variance is 5%. If consistently low, check NVIDIA driver version (need 580.x+ for CUDA 13).

If none match, open an issue with `docker logs vllm-qwen36-27b-dual 2>&1 | tail -200` + `nvidia-smi` output.
