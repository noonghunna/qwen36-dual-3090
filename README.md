# Qwen3.6-27B on dual RTX 3090

**A validated recipe for serving Qwen3.6-27B on 2× consumer 24 GB RTX 3090** at TP=2 — full OpenAI API, vision, tool calling, streaming, MTP speculative decoding, real concurrent serving, all verified end-to-end via `scripts/verify-full.sh`.

Based on [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) via **vLLM nightly** (latest dev branch) with MTP n=3 + fp8_e5m2 KV cache. Currently depends on one patch — **vLLM PR #40361** (Marlin pad-sub-tile-n) — volume-mounted from a local fork until it lands upstream.

**Philosophy:** this stack tracks `vllm:nightly` rather than pinning to a tested digest. Because we use fp8 KV (not TurboQuant), we sidestep most of the anchor-drift surface the single-card project has to manage. We add patches only when nightly actively breaks something for our config.

> 📖 **Companion repo:** [qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090) — same model on a single 3090.
> 🐛 **Open upstream PR (dependency):** [vllm-project/vllm#40361](https://github.com/vllm-project/vllm/pull/40361) — drops out as a dependency when this lands.

---

## Status at a glance

One configuration today, fully functional.

| Variant | Context | Single-stream TPS (narr / code) | Concurrent throughput | Vision | Caveats |
|---|---|---|---|---|---|
| **Default** (`docker-compose.yml`) | 64K | 68 / 89 | **257 TPS at 4 concurrent**, **385 TPS at 8 concurrent** | ✅ | Depends on vllm#40361 patch (volume-mounted) |

### What this gives you over the single-card project

- **Real concurrent serving** — 8 simultaneous requests at ~50 TPS each = **385 TPS aggregate system throughput**, vs single-card's ~85 TPS hard cap (max_num_seqs=1). About 4.5× the multi-user capacity.
- **64K context** instead of 20K with vision (or 75K text-only) — fp8 KV at TP=2 has 3× the headroom.
- **No TurboQuant complications** — sidesteps [vllm#40831](https://github.com/vllm-project/vllm/issues/40831) entirely. Full cudagraph + torch.compile speed.
- **More headroom for future variants** — DFlash draft, larger context, etc. would all fit comfortably.

### What this does NOT give you

- **Higher single-stream TPS.** Single-stream narrative is ~68 TPS, vs single-card's ~66 TPS — basically flat. On Ampere PCIe-only (no NVLink), TP=2 allreduce overhead nearly cancels the memory-bandwidth doubling for batch=1 decode. **If you only care about one-user-at-a-time chat, the single-card project is just as fast.** TP=2's win is concurrent throughput, not per-request latency.

### What this requires that single-card doesn't

- **2× RTX 3090** (or compatible Ampere SM 8.6+ cards with 24 GB each)
- **vLLM PR #40361 patched source** at `/opt/ai/vllm-src/` (or wherever — set `VLLM_SRC_DIR`). Without this, vLLM crashes during model load on TP=2 with a `GPTQ_MARLIN_MIN_THREAD_N (64) > out_features` error. See `patches/README.md`.
- ~30 GB free disk (model + Marlin source checkout)

### What's NOT enabled (intentionally)

- **TurboQuant KV** — skipped to dodge [vllm#40831](https://github.com/vllm-project/vllm/issues/40831). fp8_e5m2 has plenty of headroom at TP=2 for 64K, no functional caveats.
- **Genesis patches** — only relevant when using TurboQuant on hybrid models. Not loaded.
- **`patch_tolist_cudagraph.py`** (from the single-card project) — fixes a TurboQuant-specific crash. Not loaded.

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

## Why we skip TurboQuant on this stack

The [single-card project's long-context variant](https://github.com/noonghunna/qwen36-27b-single-3090/blob/main/compose/docker-compose.longctx-experimental.yml) uses TurboQuant 3-bit KV to fit 125K context on a single 3090 with vision. That hits [vllm#40831](https://github.com/vllm-project/vllm/issues/40831) (CUDA graph capture × spec-decode produces degenerate token loops) and ships a `cudagraph_mode=NONE` workaround at -60% TPS.

On dual-card we don't need TurboQuant — fp8_e5m2 KV at TP=2 has enough headroom for 64K context, and the bug doesn't fire on fp8. Full cudagraph + torch.compile speed, no workaround. The 65K vs 125K context tradeoff is worth it for the speed and simplicity.

If you specifically need >64K context on dual-card, TurboQuant would re-introduce the same `cudagraph_mode=NONE` workaround pattern from the single-card long-context variant. Not yet attempted on this repo.

---

## Pick a compose variant

Currently one variant. More may follow as we benchmark different configs.

```bash
cd compose && docker compose up -d
```

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

---

## Repo layout

```
qwen36-dual-3090/
├── README.md                        (this file)
├── LICENSE                          Apache-2.0
├── .gitignore
├── patches/
│   └── README.md                    Marlin pad PR #40361 dependency notes
├── compose/
│   └── docker-compose.yml           DEFAULT — Lorbus + TP=2 + MTP + fp8 + vision, 64K
└── scripts/
    ├── setup.sh                     download + SHA verify model + check Marlin patch
    ├── verify.sh                    quick smoke test (~10 sec)
    ├── verify-full.sh               full functional test (~3 min)
    └── bench.sh                     canonical TPS bench
```

---

## Upstream status

- **[vllm-project/vllm#40361](https://github.com/vllm-project/vllm/pull/40361)** — Marlin pad-sub-tile-n. **OPEN, MERGEABLE**, labeled `bug`, awaiting maintainer review. When this lands, drop the `/opt/ai/vllm-src/` mounts from the compose and just use upstream nightly.
- **[vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831)** — TurboQuant × spec-decode cudagraph bug. We dodge it entirely by using fp8 KV. Not relevant to this stack.

---

## Credits

- **Qwen team** (@Alibaba_Qwen) — base model + MTP head architecture
- **Lorbus** — AutoRound INT4 quant with preserved BF16 `mtp.fc`
- **vLLM project** — TurboQuant infrastructure and the Marlin kernel we patched
- **Intel AutoRound** — quantization framework

Our contribution: the Marlin pad patch (PR #40361), this reproducible recipe, and the [companion single-card project](https://github.com/noonghunna/qwen36-27b-single-3090).

---

## License

Apache 2.0. Do what you want with it.
