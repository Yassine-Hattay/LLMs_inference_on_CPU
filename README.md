# CPU-Only LLM Inference Benchmarking Harness

Benchmark scripts used to collect the data for **"Toward Hardware-Adaptive Pareto Fronts for CPU-Only LLM Inference: An Empirical Study of `llama.cpp` Across Four Heterogeneous Devices"** (Yassine, Master's Thesis, 2026).

These scripts drive [`llama.cpp`](https://github.com/ggerganov/llama.cpp)'s `llama-bench`, `llama-perplexity`, and `llama-server` binaries through a controlled, single-variable-at-a-time sweep of thread count, quantization format, KV-cache precision, Flash Attention, context depth, and a few system-level knobs (CPU pinning, NUMA, mmap, concurrency, speculative decoding) — with everything left un-offloaded (`-ngl 0`) so every measurement is pure CPU throughput.

An interactive calculator implementing the roofline model derived from this data is at **https://yassine-hattay.github.io/LLMs_inference_on_CPU/index.html**. The full write-up, including every figure and the derivation of the model, is the thesis itself.

## Repository contents

| File | Machine | Role |
|---|---|---|
| `Laptop (Ryzen 5600H).sh` | AMD Ryzen 5 5600H @ 3.30 GHz (6C/12T, Zen 3), 16 GB DDR4 | mid-range, compute-rich |
| `Desktop (Pentium G5420).sh` | Intel Pentium Gold G5420 @ 3.80 GHz (2C/4T), 4 GB DDR4 | low-end, bandwidth-starved |
| `Jetson Nano.sh` | Jetson Nano, ARM Cortex-A57 rev1 (4C/4T), 3.9 GB LPDDR4 | embedded, bandwidth-starved |
| `Laptop (Ryzen 5600H) data.txt` | Ryzen 5600H | consolidated raw output collected by `Laptop (Ryzen 5600H).sh` |
| `Desktop (Pentium G5420) data.txt` | Pentium G5420 | consolidated raw output collected by `Desktop (Pentium G5420).sh` |
| `Jetson Nano data.txt` | Jetson Nano | consolidated raw output collected by `Jetson Nano.sh` |

Each `... data.txt` file is a plain-text concatenation of every `.jsonl` file the corresponding script produced in its `bench_results/<host>-<arch>/` directory, one after another under a `FILE: <name> [JSONL]` header — it's a human-readable export for sharing/inspection, not itself something the harness reads back in. The [Data format](#data-format) section below documents the schema in detail and describes exactly what's present in each of these three files as uploaded.

A fourth machine, a Raspberry Pi 3B (Cortex-A53, 0.9 GB LPDDR2), was part of the original hardware fleet but its run never produced a single usable row — there is no script or data file for it here.

All three scripts are variations on the same harness — resumable, JSONL-emitting, phase-based — adapted to what each machine could actually survive. `Laptop (Ryzen 5600H).sh` and `Desktop (Pentium G5420).sh` run the full phase set (A–M); `Jetson Nano.sh` only runs Phases A and B, since the Nano's 4 GB of RAM ruled out the larger models and the server-based phases used elsewhere.

## Prerequisites

1. **A built `llama.cpp`.** Each script expects to be run from a directory containing `./build/bin/llama-bench` (required), plus `./build/bin/llama-perplexity` and `./build/bin/llama-server` (required only for the phases that use them — see the [phase reference](#phase-reference) below). Build with GPU offload disabled/unused; every phase here passes `-ngl 0` explicitly regardless.
2. **A WikiText-2 test file** at `./wiki.test.raw`, used by the perplexity phase. This is the standard WikiText-2 raw test split used throughout the quantization literature.
3. **GGUF model files.** See [Models directory](#models-directory-and-a-path-inconsistency-to-know-about) below — the exact filenames each script looks for are listed there.
4. **Standard Linux tooling:** `bash`, `curl`, and `jq`-free JSON parsing via `grep`/`cut` (already handled in-script). `Desktop (Pentium G5420).sh` and `Laptop (Ryzen 5600H).sh` additionally need `taskset` and `numactl` for Phase F (`util-linux` and `numactl` packages respectively) — if these aren't installed, Phase F will fail for those two sub-tests but the rest of the script continues.
5. **Python + pandas**, only if you want to load the `.jsonl` output afterward (not required to run the scripts themselves).

## Models directory (and a path inconsistency to know about)

Each script resolves models relative to its own working directory, and **the three scripts don't agree on the relative path**:

- `Jetson Nano.sh` looks for models at `../models`
- `Desktop (Pentium G5420).sh` and `Laptop (Ryzen 5600H).sh` look for models at `../../models`

This reflects how each machine's directory tree actually happened to be laid out during data collection, not a deliberate convention — if you're re-running these scripts, check the `MODELS_DIR=` line near the top of the script and edit it (it's a plain variable, not overridable via environment variable) to point at wherever your `.gguf` files live, relative to wherever you launch the script from. `BIN_DIR` and `WIKITEXT` are always resolved relative to the current directory (`./build/bin`, `./wiki.test.raw`), so in practice each script expects to be launched from inside (or symlinked next to) your `llama.cpp` checkout.

Filenames the scripts look for (case-sensitive, must match exactly — a missing file is silently skipped rather than erroring, so double-check names if a phase produces fewer rows than expected):

| Filename | Type | Total / Active Params | Used by |
|---|---|---|---|
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Dense | 0.5B | desktop, laptop (Phase C) |
| `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | Dense | 1.5B | all three (jetson: Phase A; desktop/laptop: Phase C, K1 draft model) |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Dense | 7.6B | all three (main model for most phases) |
| `Qwen2.5-7B-Instruct-Q4_0.gguf` | Dense | 7.6B | desktop, laptop (Phase A) |
| `Qwen2.5-7B-Instruct-Q3_K_M.gguf` | Dense | 7.6B | desktop, laptop (Phase A) |
| `Qwen2.5-7B-Instruct-IQ4_XS.gguf` | Dense | 7.6B | desktop only (Phase A) |
| `Qwen2.5-7B-Q8_0.gguf` | Dense | 7.6B | desktop only (Phase A) |
| `Qwen3-8B-Q4_K_M.gguf` | Dense | 8B | desktop, laptop (Phase C) |
| `meta-llama-3.1-8b-instruct-q4_0.gguf` | Dense | 8B | desktop, laptop (Phase C) |
| `phi-4-IQ4_NL.gguf` | Dense | 14B | desktop, laptop (Phase C) |
| `Qwen1.5-MoE-A2.7B-Chat.IQ4_NL.gguf` | MoE | 14B total / 2.7B active | desktop, laptop (Phases C, D) |
| `DeepSeek-V2-Lite.IQ4_NL.gguf` | MoE | 16B total / 2.4B active | desktop, laptop (Phases C, D) |
| `DeepSeek-Coder-V2-Lite-Base.IQ4_NL.gguf` | MoE | 16B total / 2.4B active | desktop, laptop (Phase C) |

You don't need every file to run a script — each phase checks for the presence of the models it needs and quietly skips the ones that aren't there. The MoE rows exist in the harness for completeness; **no MoE phase reached completion on any machine** in the original data collection (see below), so treat them as untested plumbing rather than validated functionality.

## Running a script

These are bash scripts, so on Windows they need to be run from WSL or Git Bash rather than PowerShell/cmd. Because the filenames contain spaces and parentheses, always quote them:

```bash
cd /path/to/llama.cpp                    # wherever your build/bin/ and wiki.test.raw live
cp "/path/to/Desktop (Pentium G5420).sh" .   # or "Laptop (Ryzen 5600H).sh" / "Jetson Nano.sh"
chmod +x "Desktop (Pentium G5420).sh"
./"Desktop (Pentium G5420).sh"
```

By default the thread sweep is derived from the machine's own core count: `{1, 2, 4, 6, 8, $(nproc)}` for the desktop/laptop scripts, `{1, 2, 4, $(nproc)}` for the Jetson script, deduplicated and sorted. Override it with an environment variable if you want a different sweep (e.g., on a 4-core Raspberry Pi):

```bash
THREADS_LIST="1,2,4" ./"Desktop (Pentium G5420).sh"
```

Note that `$(nproc)` reports *logical* processors, so on an SMT/hyperthreaded chip (like the 6C/12T Ryzen) the sweep's top value can exceed the physical core count — that gap is itself the point of Phase B's thread scaling measurements (see Section 3.1 of the thesis).

Each script is safe to `Ctrl-C` and re-run at any time — see [Resumability](#resumability) below.

## Phase reference

Every phase isolates one axis of `llama.cpp`'s configuration space while holding the rest fixed. Section references are to the thesis.

| Phase | What it sweeps | Holds fixed | In which script(s) |
|---|---|---|---|
| **A** — Quantization Pareto | Model / quant format | threads swept, `-p 512 -n 128` | jetson, desktop |
| **B** (`fa0`, `fa1`, `quantized`, `ppl`) — KV-cache & attention | `-ctk`/`-ctv` (f16 / q8_0 / q4_0 / iq4_nl), `-fa` on/off, context depth (0 / 3584 / 15872) | model = Qwen2.5-7B-Q4_K_M, threads swept | all three |
| **C** — Cross-model baseline | Model (full staged roster incl. MoE) | quant fixed per model, threads swept | desktop, laptop |
| **D** — Flash Attention, dense vs. MoE | `-fa` 0/1, model (dense vs. MoE), context depth | threads swept | desktop, laptop |
| **F** — CPU isolation & pinning | default / `taskset` (4 cores) / `taskset` (all cores) / `numactl --interleave` | model, quant fixed | desktop, laptop |
| **G** — Memory mapping | `-mmp` on/off (note: this is `mmap`, **not** `mlock` — `llama-bench` has no `--mlock` flag) | model, quant fixed | desktop, laptop |
| **I** — Multi-process contention | number of concurrent `llama-server` instances (1/2/4) | 2 threads/server, via HTTP `/completion` | desktop, laptop |
| **K1** — Speculative decoding | thread count, with/without a 1.5B draft model (`-md`) | via `llama-server` HTTP `/completion` | desktop, laptop |
| **L** — Micro-batch sweep | `-ub` (128/256/512/1024/2048) | threads swept | desktop, laptop |
| **M** — Context-depth knee | context depth (0/512/1536/3584/7680/15872) × `-ctk`/`-ctv` (f16, q8_0) | `-fa 1`, threads swept | desktop, laptop |

Three tools do all the actual measuring, and it's worth knowing which is which since the output format differs:

- **`llama-bench`** (Phases A, B, C, D, F, G, L, M) runs a fixed workload for `$REPS` repetitions and emits one JSON object per line (`-o jsonl`) with mean and standard deviation tokens/sec.
- **`llama-perplexity`** (Phase B's `_ppl` step) runs the model over `wiki.test.raw` in chunks and logs a perplexity score to a `.log` file — not JSONL.
- **`llama-server`** (Phases I, K1) is the actual inference engine exposed over HTTP; these two phases launch a real server process and issue `/completion` requests against it, appending one CSV row per configuration. This is structurally different from the other phases — a single live measurement per configuration, not an averaged repetition — so treat Phase I/K1 numbers with that caveat if you do get any.

## Resumability

Every phase is designed to survive being killed mid-run — deliberately, since these scripts ran unattended for hours on hardware that wasn't dedicated benchmarking infrastructure (a laptop that gets closed, a desktop used for other things, a Jetson on an unreliable supply). The mechanism:

- Each thread value (or model, or config) gets **its own output file**, so a crash invalidates only the one measurement in flight, not the whole phase.
- Before launching a measurement, the script checks the **row count already present** in that file against how many rows a complete run would produce — not just whether the file exists. A short file is recognized as partial and is fully re-run from scratch (`llama-bench` itself has no internal checkpointing, so a partial file can't be trusted to have consistent, comparable rows and isn't appended to).
- The two server-driven phases (I, K1) use a CSV instead: a header is written only if the file is missing or empty, and each configuration is checked against existing rows by exact key match before being attempted, so a restart picks up at the next untried key.

This means you can simply re-run the same script after an interruption (or on a schedule/cron) and it will only do the work that isn't already done — no separate "resume" flag needed.

One caveat baked into the current scripts: `phase_a` and `phase_c` in `Desktop (Pentium G5420).sh` use `grep -c "$m"` against the model's own filename to figure out how many rows already exist for that specific model in a shared multi-model output file. If a filename is a substring of another filename in your model set, this count will be wrong — none of the filenames in the table above collide this way, but it's worth knowing if you add new models.

## Data format

Every `llama-bench`-driven phase emits **JSON Lines**: one JSON object per line, no wrapping array, no trailing commas between lines. Each line is one *complete measurement* — a single (model, quantization, thread count, context depth, KV-cache setting, prompt/generation length) configuration, already averaged over `$REPS` repetitions internally (`$REPS` = 5 on the desktop and laptop, 3 on the Jetson). The three uploaded `.txt` files are a plain-text concatenation of these `.jsonl` files (see [Repository contents](#repository-contents)); strip the `FILE: ... [JSONL]` / `[EMPTY FILE]` header lines and each block below one is directly parseable as JSONL.

Every row has the same 38 columns. The ones that actually vary across rows in this dataset, and what they mean, are:

| Column | Type | Meaning |
|---|---|---|
| `cpu_info` | string | CPU model string reported by the OS (e.g. `"AMD Ryzen 5 5600H with Radeon Graphics"`, `"Intel(R) Pentium(R) Gold G5420 CPU @ 3.80GHz"`, `"ARMv8 Processor rev 1 (v8l)"`) |
| `model_filename` | string | Path to the `.gguf` file as passed to `-m`, relative to wherever the script was run (`../../models/...` on desktop/laptop, `../models/...` on Jetson — see the path note above) |
| `model_type` | string | Human-readable model/quant description parsed from the GGUF metadata, e.g. `"qwen2 7B Q4_K - Medium"` |
| `model_size` | int | Model file size on disk, in bytes |
| `model_n_params` | int | Total parameter count |
| `n_threads` | int | Thread count for this measurement — the swept variable in Phase A/B |
| `type_k`, `type_v` | string | KV-cache quantization type for keys/values (`f16`, `q8_0`, `q4_0`, `iq4_nl`) — the swept variable in Phase B's quantized sub-phase |
| `flash_attn` | bool (or `-1`) | Flash Attention on/off. Note: every row in the desktop data has `flash_attn: -1` rather than `true`/`false` — this is `llama-bench`'s "auto" sentinel, not a data-entry error, and shows up because Phase A doesn't pass `-fa` explicitly on that machine. The Jetson and laptop data use proper `true`/`false` values throughout. |
| `n_prompt`, `n_gen` | int | Which of the two workload types this row measures: `(512, 0)` = prompt processing (512-token prompt, 0 generated), `(0, 128)` = generation (0-token prompt, 128 generated). Every phase in this dataset produces exactly these two row-types per configuration, never both nonzero in the same row. |
| `n_depth` | int | Context depth prepended before the timed prompt/generation (0, 3584, or 15872 tokens in Phase B) |
| `test_time` | string | ISO-8601 UTC timestamp of when that specific measurement was run — useful for spotting the repeated-run artifact described below |
| `avg_ns`, `avg_ts` | int, float | Mean latency (nanoseconds) and mean throughput (tokens/second) across the `$REPS` repetitions — **`avg_ts` is the number used throughout the thesis** |
| `stddev_ns`, `stddev_ts` | int, float | Standard deviation of the same across repetitions |
| `samples_ns`, `samples_ts` | list[int], list[float] | The raw per-repetition values `avg_ns`/`avg_ts` were computed from — length equals `$REPS` for that machine |

The remaining columns (`build_commit`, `build_number`, `gpu_info`, `backends`, `n_batch`, `n_ubatch`, `cpu_mask`, `cpu_strict`, `poll`, `n_gpu_layers`, `n_cpu_moe`, `split_mode`, `main_gpu`, `no_kv_offload`, `devices`, `tensor_split`, `tensor_buft_overrides`, `use_mmap`, `use_direct_io`, `embeddings`, `no_op_offload`, `no_host`, `fit_target`, `fit_min_ctx`) are `llama-bench`'s standard fields and are constant across every row in this dataset (`n_gpu_layers: 0` throughout, confirming pure-CPU execution; `build_commit: "unknown"` throughout, since these builds weren't tagged with a git commit at build time).

Loading example:

```python
import pandas as pd
df = pd.read_json("phaseB_f16_fa1_t4.jsonl", lines=True)
df["workload"] = df.apply(lambda r: "prefill" if r.n_prompt > 0 else "generation", axis=1)
```

### What's actually in each uploaded file

**`Jetson Nano data.txt`** — 2 files merged, but only 1 has data:
- `phaseA_quant.jsonl`: **12 rows**, fully complete and clean — two models (`Qwen2.5-7B-Instruct-Q4_K_M`, `Qwen2.5-1.5B-Instruct-Q4_K_M`) × threads `{1, 2, 4}` × `{prompt, generation}`, exactly 12/12 expected rows with no duplicates or gaps.
- `phaseB_f16_fa0_t1.jsonl`: **empty** (0 bytes). None of Phase B produced data on the Jetson, consistent with the thesis's own accounting.

**`Desktop (Pentium G5420) data.txt`** — 1 file merged:
- `phaseA_quant.jsonl`: **33 rows**, and it's a clear illustration of the "Design 1" resumability failure the thesis describes (Section 2.6). Breaking it down by model:
  - `Q4_0`: fully complete — 5 threads × 2 workloads = 10/10 rows, one clean run per configuration, all from 2026-07-24.
  - `Q3_K_M`: 9/10 rows — threads 1, 2, 4, 6 are complete (prompt + generation), but thread 8 only has the prompt-processing row; the generation row at thread=8 is missing entirely. All from 2026-07-27.
  - `Q4_K_M`: this is the one the thesis flags by name. It never got past thread=4, and what it does have is duplicated: thread=1 prompt-processing was run 4 separate times (2026-07-14, twice on 07-15, once on 07-16) and thread=1 generation was run 6 separate times across the same three days; thread=2 has 1 prompt-processing row and 2 generation rows; thread=4 has a single prompt-processing row and no generation row at all. Threads 6 and 8 never appear. This is the exact "same low-thread-count measurement re-measured on every restart, higher threads never reached" failure mode Design 2 (used in `Laptop (Ryzen 5600H).sh` and later folded back into `Desktop (Pentium G5420).sh`) was built to fix — if you're using this file, average or dedupe the `Q4_K_M` thread=1 rows by hand, or prefer `Q4_0`/`Q3_K_M` for anything that assumes one row per configuration.

**`Laptop (Ryzen 5600H) data.txt`** — 13 files merged, all Phase B:
- `phaseB_f16_fa0_t{1,2,4,6,8}.jsonl` and `phaseB_f16_fa1_t{1,2,4,6,8}.jsonl`: **6 rows each, 60 rows total** — fully complete (3 context depths × 2 workloads, one clean measurement each, no duplicates) across all 5 threads and both Flash Attention settings.
- `phaseB_quantized_fa1_t1_q8_0_q8_0.jsonl` and `phaseB_quantized_fa1_t1_q8_0_q4_0.jsonl`: **6 rows each** — fully complete, but only at thread=1.
- `phaseB_quantized_fa1_t1_q8_0_iq4_nl.jsonl`: **4 rows** — incomplete; the 15872-context-depth pair (prompt and generation) never ran, so only depths 0 and 3584 are present.
- No other quantized-KV pairings (of the 9 possible `{q8_0, q4_0, iq4_nl}` × `{q8_0, q4_0, iq4_nl}` combinations), no perplexity logs, and no Phase C onward are present in this file — consistent with the thesis's statement that the laptop run stopped after 3 of 9 KV-format pairings.

## Data coverage: what each script actually finished

Not every phase that ran actually finished before time ran out during the original data collection. This is the ground truth as reflected in the three uploaded data files above — if you're comparing your own re-run against the published results, this is what "the published results" actually consist of:

| Machine | Completed | Not run / incomplete |
|---|---|---|
| Laptop (Ryzen 5600H) | Phase B, f16 KV, `fa=0`/`1`, all 3 depths, full 5-thread sweep (60/60 rows, no duplicates) | Phase B quantized-KV: 2 of 9 pairings fully complete, 1 of 9 partial (missing one context depth), remaining 6 not run. Perplexity, Phase C–M: not run |
| Desktop (Pentium G5420) | `Q4_0` quant sweep (full, 5 threads, 10/10 rows) | `Q3_K_M`: 9/10 rows (missing thread=8 generation). `Q4_K_M`: repeatedly restarted from thread=1 across 3 separate days, never reached thread 6 or 8 — see the duplication note above. Phase B–M: not run |
| Jetson Nano | Phase A, both target models, 3-thread sweep (complete, 12/12 rows, no duplicates) | Phase B (f16): file present but empty, 0 rows |
| Raspberry Pi 3B | Nothing | No script or data file exists for this machine |

In other words: `Jetson Nano.sh`, `Laptop (Ryzen 5600H).sh`, and `Desktop (Pentium G5420).sh` all *implement* the full multi-phase design described above, but none of the three machines actually got through all of it — the scripts are the full harness; the data in these three files is a subset of what they're capable of producing. Finishing the remaining phases (especially Phases C–M on the laptop and desktop, and Phase B on the Jetson) is listed as near-term future work in the thesis (Section 5.3.1).

## Output

All `llama-bench`-driven phases write `.jsonl` to `./bench_results/$(hostname)-$(uname -m)/` when you run the scripts yourself. See [Data format](#data-format) above for the full column reference and a breakdown of what's actually in the three sample data files included in this repo. Perplexity phases write `.log` files (grep for `"final estimate"`); Phases I and K1 write `.csv` files directly.

## Citation

If you use this harness or its data, please cite the thesis:

```
Yassine. "Toward Hardware-Adaptive Pareto Fronts for CPU-Only LLM Inference:
An Empirical Study of llama.cpp Across Four Heterogeneous Devices."
Master's Thesis, 2026.
```