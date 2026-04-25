# Qwen3.6-27B on dual RTX 3090

**A validated recipe for serving Qwen3.6-27B on 2× consumer 24 GB RTX 3090** at TP=2 — full OpenAI API, vision, tool calling, streaming, MTP speculative decoding, real concurrent serving, all verified end-to-end via `scripts/verify-full.sh`.

Based on [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) via **vLLM nightly** (latest dev branch) with MTP n=3 + fp8_e5m2 KV cache. Currently depends on one patch — **vLLM PR #40361** (Marlin pad-sub-tile-n) — volume-mounted from a local fork until it lands upstream.

**Philosophy:** this stack tracks `vllm:nightly` rather than pinning to a tested digest. Because we use fp8 KV (not TurboQuant), we sidestep most of the anchor-drift surface the single-card project has to manage. We add patches only when nightly actively breaks something for our config.

> 📖 **Companion repo:** [qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090) — same model on a single 3090.
> 🐛 **Open upstream PR (dependency):** [vllm-project/vllm#40361](https://github.com/vllm-project/vllm/pull/40361) — drops out as a dependency when this lands.

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

# 3. Download + SHA-verify the model (~20 GB, 10-30 min)
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
├── README.md                              (this file)
├── LICENSE                                Apache-2.0
├── .gitignore
├── patches/
│   └── README.md                          Marlin pad PR #40361 dependency notes
├── compose/
│   ├── docker-compose.yml                 DEFAULT — fp8 + MTP n=3 + 262K + vision (port 8010)
│   ├── docker-compose.turbo.yml           Turbo — TurboQuant_3bit_nc + Genesis v7.14 + 4-stream (port 8011)
│   ├── docker-compose.dflash.yml          DFlash — N=5 + 185K + vision (port 8012)
│   └── docker-compose.dflash-noviz.yml    DFlash text-only — N=5 + 200K (port 8013)
└── scripts/
    ├── setup.sh                           download + SHA verify model + check Marlin patch
    ├── verify.sh                          quick smoke test (~10 sec)
    ├── verify-full.sh                     full functional test incl. needle ladder (~3 min)
    └── bench.sh                           canonical TPS bench
```

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
