# Qwen3.6-27B on dual RTX 3090

> ## 🚚 Active development moved to **[noonghunna/club-3090](https://github.com/noonghunna/club-3090)**
>
> The single-card and dual-card recipes have been consolidated into a single model-agnostic repo organized by inference engine (vLLM / llama.cpp / SGLang) rather than by card count.
>
> - **For new setups:** use [club-3090](https://github.com/noonghunna/club-3090) — the dual-card configs live at `models/qwen3.6-27b/vllm/compose/docker-compose.dual*.yml`.
> - **Existing setups keep working:** this repo isn't deleted; its files are unchanged. But future bug fixes, improvements, and new model support land in club-3090.
> - **New issues:** please file at [club-3090/issues](https://github.com/noonghunna/club-3090/issues). This repo's issue tracker remains open for ongoing conversations.
>
> The original README content is preserved below for users following external links.

---

**Run Qwen3.6-27B at full 262K context with 4-stream concurrency on 2× consumer 24 GB RTX 3090.** Drop-in OpenAI-compatible API, full multimodal feature set, no cloud bills.

---

## TL;DR — what you'll get

- A 27-billion-parameter model running on **2× RTX 3090** (48 GB combined VRAM) at **TP=2**
- **OpenAI-compatible API** on `http://localhost:8010` — point any OpenAI SDK at it
- **All the features** working: chat, vision, tool calling, streaming, reasoning mode
- **Full 262K context** (Qwen3.6's natural max) with vision enabled
- **Real concurrent serving** — 2-4 streams at full ctx (turbo variant supports 4)
- **~71-89 TPS single-stream** (narr / code), **~257 TPS aggregate at 4 streams**
- **DFlash spec-decode option**: 78 / 128 TPS single-stream (fastest path for code)

**First time here?** → [**Quick start**](#quick-start).
**Coming from the single-card repo and want to know what changes?** → [What this gives you](#what-this-gives-you-over-the-single-card-project).
**Hit an error?** → [Troubleshooting](#troubleshooting).
**Don't know what TP / NVLink / DFlash mean?** → [Glossary](#glossary).

---

## Will this work for you?

| You'll need | Notes |
|---|---|
| 2× NVIDIA RTX 3090 (24 GB each) | Other Ampere/Ada cards work too (4090, A6000, A5000) at SM 8.0+. Mixed-card setups are untested by us. |
| ~30 GB free disk | Model 18 GB + patched vLLM source clone + Docker layers. |
| Linux (Ubuntu 22.04+ tested) | Same caveats as single-card: vLLM is Linux + CUDA only. WSL2 untested. |
| Docker + NVIDIA Container Toolkit | If `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi` shows both GPUs, you're good. |
| NVIDIA driver 580.x+ | For CUDA 13 runtime in vLLM nightly. `nvidia-smi`. |
| ATX power supply with capacity for ~460W combined | At 230W per card cap. Most modern ATX PSUs handle this comfortably. |
| **NVLink bridge NOT required** | The stack disables NVLink-specific NCCL features. PCIe-only is the tested path. If you have NVLink it'll be unused (and that's fine). |
| Patched vLLM source clone | `vllm#40361` (Marlin pad-sub-tile-n) hasn't landed upstream yet — we volume-mount a local fork. One-time `git clone` during setup. |

**You're probably fine if:** you've successfully run a single-card stack before, you have two slots that fit 3090s with airflow, and you have ~30 minutes for setup.

**Try the [single-card repo](https://github.com/noonghunna/qwen36-27b-single-3090) first if:** you've never run a local LLM, you only have one 3090, or you're not sure dual-card concurrent serving is what you actually need (per-stream TPS isn't much faster than single-card; the win is multi-tenant aggregate throughput).

---

## How this is built (one-paragraph version)

We use [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) (the same INT4 quant as the single-card repo) on **vLLM nightly** with **TP=2** (tensor parallelism splits the model across both cards). The default ships with **fp8 KV cache + MTP n=3 spec-decode**, which gets you full 262K context with 2× concurrency. A **turbo variant** swaps fp8 KV for TurboQuant 3-bit + Genesis v7.14 patches to unlock 4× concurrency at the same ctx. A **DFlash variant** uses [Luce DFlash N=5 spec-decode](https://github.com/luce-spec/dflash-vllm) for the fastest per-stream code TPS (128 vs 89 default). All variants currently need [`vllm#40361`](https://github.com/vllm-project/vllm/pull/40361) (Marlin pad-sub-tile-n) volume-mounted as a fork until it lands upstream.

> 📖 **Companion repo:** [qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090) — same model on a single 3090.
> 🐛 **Upstream PR (dependency):** [vllm-project/vllm#40361](https://github.com/vllm-project/vllm/pull/40361) — drops out when this lands.

**Philosophy:** this stack tracks `vllm:nightly` rather than pinning to a tested digest. Because we use fp8 KV (not TurboQuant) by default, we sidestep most of the anchor-drift surface the single-card project has to manage. We add patches only when nightly actively breaks something for our config.

---

## Status at a glance

Four configurations, all functional and end-to-end verified (10K / 30K / 60K / 90K recall ladder passes for each).

| Variant | Context | Single-stream TPS (narr / code) | Concurrency at 262K | Vision | Niche |
|---|---|---|---|---|---|
| **Default** (`docker-compose.yml`) — fp8 + MTP | **262K** | **71 / 89** | 2.36× | ✅ | Best per-stream TPS · general use |
| **Turbo** (`docker-compose.turbo.yml`) — TurboQuant_3bit_nc + MTP + Genesis v7.14 | **262K** | 58 / 69 | **4.59×** | ✅ | **9× the KV pool · 4-stream serving** |
| **DFlash** (`docker-compose.dflash.yml`) — DFlash N=5 + vision | 185K | **78 / 128** | 1× | ✅ | Fastest single-stream code with vision |
| **DFlash text-only** (`docker-compose.dflash-noviz.yml`) — DFlash N=5 + 200K | 200K | 77 / 124 | 1× | ❌ | Fastest single-stream max-ctx |

All variants depend on **vllm#40361** (Marlin pad-sub-tile-n) volume-mounted from local source. Turbo additionally depends on **Genesis v7.14 patches** for the [#40880](https://github.com/vllm-project/vllm/issues/40880) MTP × TurboQuant × cudagraph fix (P64/P65/P66/P68/P69 enabled via env opts).

### What this gives you over the single-card project

- **Full 262K context** — fp8 default fits the model's natural max with 2.36× concurrency. No `max-model-len` compromise.
- **Real concurrent serving** — fp8 default supports 2 concurrent at full ctx; turbo variant unlocks 4 concurrent at full 262K (9× the KV pool).
- **DFlash spec-decode at 78/128 TPS** — both vision and no-vision variants validated. Single-stream peak performance.
- **All multimodal features** — vision, tools, thinking, recall, streaming, MTP n=3 / DFlash N=5 spec-decode (your pick), prefix caching, chunked prefill.

### What this does NOT give you

- **Higher single-stream TPS.** Single-stream narrative is ~68 TPS, vs single-card's ~66 TPS — basically flat. On Ampere PCIe-only (no NVLink), TP=2 allreduce overhead nearly cancels the memory-bandwidth doubling for batch=1 decode. **If you only care about one-user-at-a-time chat, the single-card project is just as fast.** TP=2's win is concurrent throughput, not per-request latency.

### What this requires that single-card doesn't

- **2× RTX 3090** (or compatible Ampere SM 8.6+ cards with 24 GB each)
- **vLLM PR #40361 patched source** at `/opt/ai/vllm-src/` (or wherever — set `VLLM_SRC_DIR`). Without this, vLLM crashes during model load on TP=2 with a `GPTQ_MARLIN_MIN_THREAD_N (64) > out_features` error. See `patches/README.md`.
- ~30 GB free disk (model + Marlin source checkout)

### TurboQuant + Genesis v7.14 — now an option

The Turbo variant (`docker-compose.turbo.yml`) loads TurboQuant_3bit_nc KV with Genesis v7.14 patches enabled. v7.14 (released 2026-04-25) closes [#40880](https://github.com/vllm-project/vllm/issues/40880) (MTP × TurboQuant × cudagraph corruption) via P65's cudagraph downgrade for spec-decode. The fp8 default doesn't load Genesis at all — it doesn't need to.

**Per-stream cost of the Turbo variant:** ~25% TPS regression vs fp8 (P65 forces `cudagraph_mode=PIECEWISE` for spec-decode → eager continuation). **Concurrency win:** 4.59× streams at full 262K vs fp8's 2.36×, so aggregate throughput exceeds fp8 above ~3 concurrent streams.

---

## Requirements

- **GPUs:** 2× NVIDIA RTX 3090 (24 GB, Ampere sm_86). PCIe-only is fine — no NVLink bridge required (compose disables NVLink-dependent NCCL features).
- **Driver:** 580.x or newer (for CUDA 13 runtime in the vLLM nightly image).
- **Disk:** ~30 GB (model weights + patched vLLM source clone).
- **Software:**
  - Docker with NVIDIA Container Toolkit
  - `git`, `curl`, `sha256sum` (setup script uses them)
  - `hf` CLI *or* `huggingface-cli` (install: `pip install 'huggingface-hub[hf_transfer]'`)

No system Python required.

---

## Quick start

```bash
# 1. Clone this repo
git clone https://github.com/noonghunna/qwen36-dual-3090.git
cd qwen36-dual-3090

# 2. Clone the patched vLLM source (Marlin pad fork — needed until vllm#40361 lands)
sudo mkdir -p /opt/ai && sudo chown $USER /opt/ai
git clone -b marlin-pad-sub-tile-n https://github.com/noonghunna/vllm.git /opt/ai/vllm-src

# 3. Download + SHA-verify the model (~20 GB, 10-30 min) + clone Genesis patches (~50 MB)
#    setup.sh now handles both: model download with SHA verification AND
#    cloning Sandermage/genesis-vllm-patches into ./patches/genesis (used by
#    the .turbo and .p67-bench composes). Set SKIP_GENESIS=1 to skip if you only
#    need the default fp8 / DFlash composes.
bash scripts/setup.sh

# 4. Start the server
cd compose && docker compose up -d

# 5. Watch it come up (~3 min for cold compile across both cards)
docker logs -f vllm-qwen36-27b-dual
# Wait for "Application startup complete"

# 6. Sanity test
curl -sf http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround",
       "messages":[{"role":"user","content":"Capital of France?"}],
       "max_tokens":30}'

# 7. Run the benchmark
cd .. && bash scripts/bench.sh
```

The stack serves on `http://localhost:8010/v1/*` as a drop-in OpenAI-compatible endpoint.

### What success looks like

If everything booted correctly, `docker logs vllm-qwen36-27b-dual` should show:

```
INFO ...  [marlin-patch] applying pad-sub-tile-n diff to /usr/local/lib/python3.12/dist-packages/vllm/...
INFO ...  Tensor parallelism initialized across 2 GPUs (TP=2)
INFO ...  Available KV cache memory: 5.x GiB per GPU (262K-token pool)
INFO ...  Application startup complete.
```

`nvidia-smi` should show two processes, one on each GPU, each using ~22-23 GB VRAM (mirror copy of TP-split layers + KV pool half).

A successful curl returns the standard OpenAI completion shape (Paris answer for the sanity test).

For thorough verification (all 10 functional checks — server, Genesis patches if turbo, tool calling, streaming, thinking, long-context recall, tool-prefill safety, output quality, MTP acceptance):
```bash
bash scripts/verify-full.sh
```

---

## Why dual-card on the same model

Single-card serves Qwen3.6-27B at ~85 TPS peak in 22.8 GB VRAM, but only one request at a time (`max_num_seqs=1` because weights eat most of the VRAM). Dual-card splits the weights across two cards, which gets you:

- **Concurrent serving budget grows** from `max_num_seqs=1` to `max_num_seqs=8` because the per-card weight footprint halves (9 GB instead of 18 GB) → 15 GB free per card for KV pool + workspace.
- **Aggregate throughput scales nicely with concurrency:**
  - 1 stream:  71 TPS
  - 2 streams: 119 TPS
  - 4 streams: 257 TPS
  - 8 streams: 385 TPS

  At 8 concurrent the per-stream TPS drops to ~50, but you're serving 8 users simultaneously. That's ~4.5× the multi-user capacity of single-card.
- **No context-vs-feature tradeoffs** — vision + tools + 64K + MTP + recall all fit comfortably. (Single-card has to drop vision to get past 20K.)

What dual-card does NOT buy you on this model:

- **Single-stream latency improvement.** TP=2 allreduce on PCIe-only Ampere costs almost as much as the memory-bandwidth doubling saves. Single-stream narrative comes in at ~68 TPS — basically the same as single-card. If your workload is "one user, low concurrency, want fastest replies," stay on single-card.

The cost is needing two cards and the Marlin pad patch. Once vllm#40361 lands, the patch dependency drops and this is just `docker compose up`.

> ### A note on NVLink
>
> **All numbers in this repo are measured PCIe-only — no NVLink bridge.** The RTX 3090 supports a third-party NVLink bridge (~112 GB/s), but we don't run one.
>
> Each TP=2 decode step requires an all-reduce after the QKV+output projection. On PCIe-4 (32 GB/s) at batch=1 this all-reduce takes a non-trivial fraction of the per-layer time, which is why dual-card single-stream TPS on this stack lands flat against single-card. With an NVLink bridge the all-reduce should drop substantially, and we'd expect single-stream TPS to scale closer to the memory-bandwidth doubling — plausibly ~100-130 TPS narrative range, though we haven't measured. Multi-stream throughput (4-8 concurrent) is bound by compute and VRAM rather than interconnect, so NVLink wouldn't change those much.
>
> If you have an NVLink bridge for your 3090s and run this recipe on it, we'd love to hear the numbers — the gap between PCIe-only and NVLink on this exact config isn't well documented anywhere we've seen.

---

## When to pick TurboQuant (Turbo variant)

The Turbo variant unlocks 4-stream concurrency at full 262K (KV pool: 1.5M tokens vs fp8's 168K — 9× more). It applies Genesis v7.14 patches to fix the [#40880](https://github.com/vllm-project/vllm/issues/40880) bug class that previously made TurboQuant + MTP + cudagraph silently corrupt tool-call output.

**Pick Turbo if:**
- You're serving 4+ concurrent agents at long ctx (RAG batch jobs, multi-user team workloads)
- KV pool dominance matters (long-running sessions with prefix-cached state)

**Stay on fp8 default if:**
- Single-stream latency / TPS matters more than concurrency
- You don't routinely hit 4+ concurrent
- Per-stream code TPS (89 vs 69) is your main metric

**Cost of Turbo:** P65's cudagraph downgrade (FULL → PIECEWISE for spec-decode) costs ~25% per-stream TPS. Crossover with fp8 happens around 3 concurrent streams; above that Turbo wins on aggregate.

We had to also adjust two consumer-Ampere knobs vs Sandermage's A5000-class defaults: `gpu-memory-utilization 0.92 → 0.85` and `max-num-batched-tokens 8192 → 4128`. These leave activation headroom that Sandermage's 32 GB cards already had. Without them, deep-prefill (60K+) requests OOM. See `compose/docker-compose.turbo.yml` for the working config.

---

## Pick a compose variant

Only one container can bind a given port at a time — `docker compose down` (with the right `-f`) before switching.

```bash
# Default: fp8 + MTP n=3 + 262K + vision + 2 streams (port 8010)
cd compose && docker compose up -d

# Turbo: TurboQuant_3bit_nc + MTP n=3 + Genesis v7.14 + 4-stream concurrency (port 8011)
cd compose && docker compose -f docker-compose.turbo.yml up -d

# DFlash with vision: N=5 + 185K + vision + 1 stream (port 8012, fastest single-user)
cd compose && docker compose -f docker-compose.dflash.yml up -d

# DFlash text-only: N=5 + 200K, no vision + 1 stream (port 8013, fastest max-ctx)
cd compose && docker compose -f docker-compose.dflash-noviz.yml up -d
```

### Decision tree

- **Multi-tenant or many concurrent agents at long ctx?** → **Turbo** (4-stream at 262K, 9× KV pool)
- **General use, vision matters, balanced TPS + ctx?** → **Default fp8** (best per-stream, 2 streams, 262K)
- **Solo user, code-heavy, want max single-stream TPS?** → **DFlash with vision** (78/128 TPS, 185K)
- **Need >185K ctx and don't need vision?** → **DFlash text-only** (200K, no vision)

---

## Configuration notes

### Tensor parallelism

`--tensor-parallel-size 2` splits the model across both cards. Allreduce happens on PCIe (no NVLink). To keep it stable on consumer Ampere:

- `--disable-custom-all-reduce` — vLLM's custom kernel assumes NVLink-class bandwidth; fall back to NCCL.
- `NCCL_P2P_DISABLE=1` — disables NCCL's GPU-to-GPU P2P transfers (uses host bounce buffer instead). Counterintuitively faster on PCIe-only setups because it avoids contention.
- `NCCL_CUMEM_ENABLE=0` — disables NCCL's CUDA memory pool, which has compatibility issues on this stack.

### Speculative decoding

MTP n=3, same as the single-card project. Acceptance rates are similar (~92/81/67% per position on warmed traffic).

### Concurrency

`--max-num-seqs 8` × `--max-num-batched-tokens 8192` lets you actually saturate the cards with concurrent requests. Single-card's `max_num_seqs=1` is a single-user serving config; dual-card's 8 is closer to a small team's worth.

### Power cap

Both cards run at default 230W cap on our stack. For ~+10% mean TPS:

```bash
sudo nvidia-smi -pl 330 -i 0
sudo nvidia-smi -pl 330 -i 1
```

---

## Benchmarking

```bash
bash scripts/bench.sh
```

The bench script reports three metrics per run + summary stats across all measured runs:

- **`wall_TPS`** = `completion_tokens / wall_time` — user-perceived total speed (includes prefill cost)
- **`decode_TPS`** = `completion_tokens / (wall_time − TTFT)` — pure model decode rate (excludes prefill)
- **`TTFT`** — time to first token, captured via streaming
- **Summary**: mean / std / **CV** / min / max for each TPS metric across N measured runs (default 5, 3 warmups; override via `RUNS=N WARMUPS=M bash scripts/bench.sh`)

CV (coefficient of variation) is the headline stability metric — Sandermage's [vllm#40914](https://github.com/vllm-project/vllm/pull/40914) cross-rig validation case hinges on it. Spec-decode configs that drop CV below 5% are noticeably more predictable for SLA-sensitive use.

You can also run `bash scripts/verify-full.sh --bench` to do "verify + measure" in one command — runs all correctness checks first (server up, tool calls, recall ladder, etc.) and then the bench if everything passes.

The script now has **10 functional checks** (was 7): server, Genesis patches applied, basic completion, tool calling, streaming, thinking, long-context needle, **(8) tool-response prefill OOM** (multi-turn payload with a ~25K-token mock tool message — catches activation-memory peak crashes), **(9) output quality / cascade detection** (2K-token essay scanned for `<tool_call>` inline cascade and repetitive degeneracy), **(10) MTP acceptance length threshold** (parses SpecDecoding metrics from logs, asserts mean AL ≥ 2.0). The new tests inherit from the single-3090 [#1 prefill-OOM investigation](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1) — TP=2 with fp8 KV (the default here) gives much wider safety margins than single-card TQ3 KV does, but the same bug classes can theoretically fire under extreme workloads, and the tests are cheap insurance.

Same canonical prompts as the single-card project (800-word essay, quicksort code), so numbers are directly comparable.

### Measured numbers (1× RTX 3090 each at 230W cap, vLLM nightly digest 9bba4628)

**Single-stream:**

| Workload | Run 1 | Run 2 | Run 3 | Notes |
|---|---|---|---|---|
| Narrative (800-word essay, 1000 tok) | 66.9 | 68.0 | 70.5 | mean 68 TPS |
| Code (quicksort, ~650 tok) | 87.1 | 90.3 | — | mean 89 TPS |

**Concurrent (narrative, 800-word essay, 1000 tok per stream):**

| Concurrency | Per-stream TPS | Aggregate TPS | Wall time |
|---|---|---|---|
| 1 | 71 | 71 | 14.1s |
| 2 | 60 | 119 | 16.7s |
| 4 | 67 | **257** | 15.5s |
| 8 | 49 | **385** | 20.7s |

At 4 concurrent, per-stream TPS holds nicely. At 8 concurrent, per-stream drops to ~50 but aggregate hits 385 — that's the regime where dual-card pulls meaningfully ahead of single-card's hard ~85 TPS cap.

**Long-context generation (262K config, prefix-cache warm):**

| Loaded ctx | Cold prefill | Generation TPS (300-tok output) |
|---|---|---|
| 100K input | ~101s (one-time) | 36 TPS |
| 200K input | ~217s (one-time) | 28 TPS |

Cold prefill is the dominant cost at long context — vLLM chunked-prefills at 8192 tok/chunk, so 200K takes ~25 chunks. Once prefilled, subsequent requests against the same prefix hit the cache and only pay generation cost. **Practical UX**: long-context use cases want either pre-warmed prefix caches or to keep sessions alive. Cold-start at 200K is genuinely slow; prefix-cached follow-ups feel snappy.

**Long-context recall:** verify-full.sh's needle ladder passes at 10K / 30K / 60K / 90K. Recall at deeper depths (180K / 240K) is plausible but not yet measured systematically.

---

## Repo layout

```
qwen36-dual-3090/
├── README.md                              (this file — start here)
├── CHANGELOG.md                           dated history
├── LICENSE                                Apache-2.0
├── docs/
│   ├── INTERNALS.md                       deep technical: Marlin patch mechanics, TP=2 allreduce, DFlash N=5
│   └── USE_CASES.md                       per-workload guides: multi-tenant, frontier ctx, RAG, code, vision
├── patches/
│   ├── README.md                          Marlin pad PR #40361 dependency notes
│   └── genesis/                           cloned by setup.sh → Sandermage/genesis-vllm-patches
├── compose/
│   ├── docker-compose.yml                 DEFAULT — fp8 + MTP n=3 + 262K + vision (port 8010)
│   ├── docker-compose.turbo.yml           Turbo — TurboQuant_3bit_nc + Genesis v7.14 + 4-stream (port 8011)
│   ├── docker-compose.dflash.yml          DFlash — N=5 + 185K + vision (port 8012)
│   ├── docker-compose.dflash-noviz.yml    DFlash text-only — N=5 + 200K (port 8013)
│   ├── docker-compose.p67-bench.yml       BENCH — 27B + MTP + Genesis v7.48 P67/P78
│   ├── docker-compose.p67-bench-dflash.yml      BENCH — 27B + DFlash N=5 + Genesis v7.48
│   └── docker-compose.p67-bench-35b-dflash.yml  BENCH — 35B-A3B + DFlash + Genesis v7.48
└── scripts/
    ├── setup.sh                           model SHA-verify + clones genesis-vllm-patches
    ├── verify.sh                          quick smoke test (~10 sec)
    ├── verify-full.sh                     full functional test — 10 checks (~3 min)
    └── bench.sh                           canonical TPS bench (wall_TPS / decode_TPS / TTFT / CV)
```

> 📜 **Want the engineering depth?** [docs/INTERNALS.md](docs/INTERNALS.md) covers Marlin pad mechanics, TP=2 allreduce, DFlash, P67.
> 📚 **Specific workload?** [docs/USE_CASES.md](docs/USE_CASES.md) has per-use-case configs and gotchas.
> 📅 **What changed when?** [CHANGELOG.md](CHANGELOG.md) is the dated log.
> 🔬 **Trying llama.cpp / SGLang instead of vLLM?** Engine comparison and quick recipes lives in the [single-card repo's docs/engines/](https://github.com/noonghunna/qwen36-27b-single-3090/tree/master/docs/engines).

---

## Glossary

| Term | What it means |
|---|---|
| **TP=2 / tensor parallelism** | Splits each model layer's weights across both GPUs; layers compute together, results combined via NCCL all-reduce. Doubles effective VRAM (48 GB total) but adds inter-GPU communication cost per forward step. |
| **NVLink** | NVIDIA's high-bandwidth GPU-to-GPU interconnect (~600 GB/s on 3090). 3090s have a connector but it's unused if no bridge is installed. We disable NVLink-dependent NCCL features and run PCIe-only. |
| **Allreduce** | The collective op TP uses to combine partial results from each GPU. On PCIe-only consumer Ampere this is ~3-5× slower than NVLink, so single-stream TPS gain from TP=2 is modest (~5%). |
| **Concurrent streams** | Multiple users/agents serving simultaneously. KV pool is shared; each stream gets a slice. The dual stack supports 2-4 concurrent at full 262K (depending on variant). Aggregate throughput scales near-linearly to ~4 streams. |
| **DFlash / DFlash N=5** | A custom 5-token draft model from z-lab specialized for Qwen3.6, optimized for code workloads. Replaces MTP n=3 with an external small model that runs in parallel; nets 78/128 vs 71/89 TPS narr/code. |
| **MTP** | Multi-Token Prediction — built-in spec-decode head that ships with Qwen3.6. Default for the fp8 + Turbo variants. |
| **Marlin pad-sub-tile-n** | A specific bug in vLLM's Marlin INT4 kernel where output features < 64 cause a crash on TP=2 (the TP-split makes some Lorbus tensors thinner than the kernel's minimum). Our [PR #40361](https://github.com/vllm-project/vllm/pull/40361) pads them to the kernel-min before dispatch. |
| **TurboQuant** | A 3-bit KV cache compression scheme. Smaller per-token KV → can fit more tokens at the same mem-util → more concurrent streams in the KV pool. We use it in the Turbo variant. |
| **fp8 / fp8_e5m2** | An 8-bit float KV cache format. Larger per-token bytes than TurboQuant but dodges several upstream bugs that affect TQ. The default variant uses fp8 — simpler, fewer patches, full feature support. |
| **Prefix cache** | When two requests share a leading prompt, vLLM serves the second from cache (skips re-prefill). Especially useful for long-document workflows where users iterate against the same loaded doc. |

---

## Upstream status

- **[vllm-project/vllm#40361](https://github.com/vllm-project/vllm/pull/40361)** — Marlin pad-sub-tile-n. **OPEN, MERGEABLE**, labeled `bug`, awaiting maintainer review. When this lands, drop the `/opt/ai/vllm-src/` mounts from the compose and just use upstream nightly.
- **[vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831)** — TurboQuant × spec-decode cudagraph bug (closed for ngram path via Sander's v7.13). Default fp8 variant dodges it; Turbo variant relies on Genesis v7.14 P65 to work around it for the MTP path.
- **[vllm-project/vllm#40880](https://github.com/vllm-project/vllm/issues/40880)** — MTP × TurboQuant × cudagraph corruption. Root-caused by Sandermage 2026-04-25; v7.14 patches (P64/P65/P66/P68/P69) fix it as a workaround. Proper fix awaits a custom Triton multi-query kernel (P67 design only). The Turbo variant of this repo applies v7.14 + the necessary consumer-Ampere knobs.
- **[vllm-project/vllm#40334](https://github.com/vllm-project/vllm/pull/40334)** — DFlash `combine_hidden_states` dtype mismatch fix. **OPEN, NOT MERGED**. Workaround in our DFlash variants: `--dtype bfloat16` (matches the draft's training dtype, sidesteps the fp32→fp16 cast).

---

## Credits

- **Qwen team** (@Alibaba_Qwen) — base model + MTP head architecture
- **Lorbus** — AutoRound INT4 quant with preserved BF16 `mtp.fc`
- **[Sandermage](https://github.com/Sandermage)** — Genesis v7.14 patch tree (P65 cudagraph downgrade, P66 capture-size filter, P64 streaming MTP, P68/P69 long-ctx tool adherence) that unblocks TurboQuant + MTP for the Turbo variant. Root-caused [#40880](https://github.com/vllm-project/vllm/issues/40880) and shipped a fix in 36 hours.
- **z-lab** — DFlash draft model for Qwen3.6-27B
- **vLLM project** — TurboQuant infrastructure and the Marlin kernel we patched
- **Intel AutoRound** — quantization framework

Our contribution: the Marlin pad patch (PR #40361), this reproducible recipe with four variants for different workloads, and the [companion single-card project](https://github.com/noonghunna/qwen36-27b-single-3090).

---

## License

Apache 2.0. Do what you want with it.
