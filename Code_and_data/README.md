# CPU-Only LLM Inference Benchmarking Harness

Benchmark scripts used to collect the data for **"Optimizing CPU-Only LLM Inference: Parameter Sweeps, Hardware Bottlenecks, and Performance Modeling"** (Yassine Hattay, Master's Thesis, ENSI, 2026).

These scripts drive [`llama.cpp`](https://github.com/ggerganov/llama.cpp)'s `llama-bench`, `llama-perplexity`, and `llama-server` binaries through a controlled, single-variable-at-a-time sweep of thread count, quantization format, KV-cache precision, Flash Attention, context depth, and system-level knobs (CPU pinning, NUMA, mmap, concurrency, speculative decoding) — with everything left un-offloaded (`-ngl 0`) so every measurement reflects pure CPU throughput.

An interactive calculator implementing the roofline model derived from this data is available at **https://yassine-hattay.github.io/LLMs_inference_on_CPU/index.html**. The full write-up, including all figures and the mathematical derivation of the model, is contained within the thesis itself.

## Repository Structure

```text
Code_and_data/
├── data_collection.sh             # Master unified benchmarking harness script
├── Desktop (Pentium G5420) data.txt # Consolidated raw output collected on Intel Pentium G5420
├── Laptop (Ryzen 5600H) data.txt   # Consolidated raw output collected on AMD Ryzen 5 5600H
├── Jetson Nano data.txt           # Consolidated raw output collected on NVIDIA Jetson Nano
├── LICENSE-DATA                   # License terms for dataset and harness code
└── README.md                      # Documentation (this file)

```

| File | Target Machine Specs | Role / Contents |
| --- | --- | --- |
| `data_collection.sh` | Portable (Linux/POSIX environment) | Parametric driver script running Phases A–M. Auto-detects core topology or accepts manual overrides. |
| `Laptop (Ryzen 5600H) data.txt` | AMD Ryzen 5 5600H @ 3.30 GHz (6C/12T, Zen 3), 16 GB DDR4 | Consolidated raw benchmark output (mid-range, compute-rich baseline). |
| `Desktop (Pentium G5420) data.txt` | Intel Pentium Gold G5420 @ 3.80 GHz (2C/4T), 4 GB DDR4 | Consolidated raw benchmark output (low-end, bandwidth-starved baseline). |
| `Jetson Nano data.txt` | ARM Cortex-A57 rev1 (4C/4T), 3.9 GB LPDDR4 | Consolidated raw benchmark output (embedded, low-bandwidth baseline). |

Each `... data.txt` file is a plain-text concatenation of every `.jsonl` file produced by the harness in its `bench_results/<host>-<arch>/` directory, concatenated under a `FILE: <name> [JSONL]` header block. It provides a single, human-readable export for inspection and downstream parsing. See [Data format](https://www.google.com/search?q=%23data-format) for schema details.

*Note on hardware fleet:* A fourth machine, a Raspberry Pi 3B (Cortex-A53, 0.9 GB LPDDR2), was included in the original hardware testing fleet; however, memory constraints prevented it from completing a usable sweep row, so no output file is included for it.

## Prerequisites

1. **Built `llama.cpp` Binaries:**
The script expects to run relative to a `llama.cpp` build containing `./build/bin/llama-bench` (required for primary benchmarks), plus `./build/bin/llama-perplexity` and `./build/bin/llama-server` (required for perplexity and server-driven phases). Ensure GPU offloading is not required; every execution phase passes `-ngl 0` explicitly.
2. **WikiText-2 Test Dataset:**
A standard WikiText-2 raw test split file located at `./wiki.test.raw` (required for Phase B perplexity validation).
3. **GGUF Model Weights:**
Place required `.gguf` model files in the target models directory. Path resolution can be set dynamically using environment variables (see [Models Directory & Paths](https://www.google.com/search?q=%23models-directory-and-path-configuration)).
4. **Standard Linux Utilities:**
`bash`, `curl`, `grep`, `cut`. For Phase F (CPU isolation and NUMA testing), `taskset` (`util-linux`) and `numactl` are required. If missing, Phase F sub-tests will register a failure and the harness will proceed to subsequent phases.
5. **Python 3 + pandas:**
Recommended for loading and analyzing generated `.jsonl` data exports.

## Models Directory and Path Configuration

By default, `data_collection.sh` resolves binary, model, and dataset paths relative to the current working directory. You can override these defaults via environment variables at execution time without modifying script source code:

| Environment Variable | Default Path | Description |
| --- | --- | --- |
| `BIN_DIR` | `./build/bin` | Path to compiled `llama.cpp` executables. |
| `MODELS_DIR` | `../models` | Directory containing `.gguf` weight files. |
| `WIKITEXT` | `./wiki.test.raw` | Path to WikiText-2 evaluation file. |

### Model Roster Requirements

Model filenames are case-sensitive and must match the exact names below. If a model file is missing, the harness gracefully skips the dependent phase without halting execution:

| Filename | Type | Total / Active Params | Phase Usage |
| --- | --- | --- | --- |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Dense | 0.5B | Phase C |
| `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | Dense | 1.5B | Phase A (Jetson), Phase C, Phase K1 (Draft Model) |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Dense | 7.6B | Primary model for core parameters and KV-cache sweeps |
| `Qwen2.5-7B-Instruct-Q4_0.gguf` | Dense | 7.6B | Phase A |
| `Qwen2.5-7B-Instruct-Q3_K_M.gguf` | Dense | 7.6B | Phase A |
| `Qwen2.5-7B-Instruct-IQ4_XS.gguf` | Dense | 7.6B | Phase A |
| `Qwen2.5-7B-Q8_0.gguf` | Dense | 7.6B | Phase A |
| `Qwen3-8B-Q4_K_M.gguf` | Dense | 8B | Phase C |
| `meta-llama-3.1-8b-instruct-q4_0.gguf` | Dense | 8B | Phase C |
| `phi-4-IQ4_NL.gguf` | Dense | 14B | Phase C |
| `Qwen1.5-MoE-A2.7B-Chat.IQ4_NL.gguf` | MoE | 14B total / 2.7B active | Phase C, Phase D |
| `DeepSeek-V2-Lite.IQ4_NL.gguf` | MoE | 16B total / 2.4B active | Phase C, Phase D |
| `DeepSeek-Coder-V2-Lite-Base.IQ4_NL.gguf` | MoE | 16B total / 2.4B active | Phase C |

*Note on MoE Models:* MoE phase definitions are implemented in the harness for architecture evaluations. Due to memory bandwidth and execution limits on the tested hardware, MoE phases did not reach full completion in the published raw dataset.

## Execution Guide

Execute `data_collection.sh` from within or adjacent to your `llama.cpp` installation directory. On Windows systems, run via WSL2 or Git Bash.

```bash
# Navigate to working llama.cpp directory
cd /path/to/llama.cpp

# Run harness with default paths
bash /path/to/Code_and_data/data_collection.sh

```

### Customizing Environment and Sweeps

Thread sweeps auto-detect logical core count (`nproc`), constructing a deduplicated, sorted array (e.g., `{1, 2, 4, 6, 8, $(nproc)}`). You can manually specify thread lists or custom model paths:

```bash
# Specify explicit thread sweep and custom models directory
THREADS_LIST="1,2,4" MODELS_DIR="/mnt/storage/models" bash data_collection.sh

```

Because `$(nproc)` counts logical hyperthreads (e.g., 12 threads on 6C/12T Ryzen), sweeping beyond physical core count allows measuring retrograde thread-scaling behaviors on bandwidth-bound workloads (Section 3.1 of thesis).

## Phase Reference

Each phase isolates a single operational axis while fixing remaining execution variables.

| Phase | Swept Variables | Fixed Parameters |
| --- | --- | --- |
| **A — Quantization Pareto** | Model & Quantization Format | Thread sweep, `-p 512 -n 128` |
| **B — KV-Cache & Attention** | `-ctk`/`-ctv` (`f16`, `q8_0`, `q4_0`, `iq4_nl`), `-fa` (0/1), Context Depth (0, 3584, 15872) | Model: `Qwen2.5-7B-Instruct-Q4_K_M`, Thread sweep |
| **C — Cross-Model Baseline** | Model architecture (Dense vs. MoE roster) | Fixed quantization per model, Thread sweep |
| **D — Flash Attention Scaling** | `-fa` (0/1), Dense vs. MoE, Context Depth | Thread sweep |
| **F — CPU Isolation & Pinning** | Default OS scheduler, `taskset` physical cores, `taskset` all cores, `numactl --interleave` | Fixed model and quantization |
| **G — Memory Mapping** | `-mmp` (mmap enabled/disabled) | Fixed model and quantization |
| **I — Multi-Process Contention** | Concurrent `llama-server` instances (1, 2, 4) | 2 threads/server via HTTP `/completion` |
| **K1 — Speculative Decoding** | Thread count, Draft model active/inactive (`-md` 1.5B) | HTTP `/completion` benchmarks |
| **L — Micro-Batch Sweep** | Micro-batch size `-ub` (128, 256, 512, 1024, 2048) | Thread sweep |
| **M — Context-Depth Knee** | Context Depth (0–15872) × `-ctk`/`-ctv` (`f16`, `q8_0`) | `-fa 1`, Thread sweep |

### Execution Drivers

1. **`llama-bench`:** Drives Phases A, B, C, D, F, G, L, and M. Runs fixed workloads for `$REPS` repetitions (5 on desktop/laptop, 3 on Jetson) and outputs JSON Lines (`-o jsonl`).
2. **`llama-perplexity`:** Drives Phase B validation (`_ppl`). Computes perplexity over `wiki.test.raw` and logs raw estimates to `.log` files.
3. **`llama-server`:** Drives Phases I and K1. Launches an HTTP server process, fires batch `/completion` requests, and records latency statistics to CSV files.

## Resumability Mechanism

The harness incorporates atomic output logging to sustain unattended long-duration benchmark runs across system restarts or interruptions:

* **File Granularity:** Each distinct thread count, quantization, or hardware configuration writes to its own specific output file in `./bench_results/$(hostname)-$(uname -m)/`.
* **Completion Verification:** Before executing a benchmark step, the harness checks existing row counts against expected repetition outputs. Partial or interrupted outputs are recognized as incomplete and re-executed cleanly from scratch.
* **Server CSV Checks:** CSV-based phases (I, K1) verify key existence prior to firing HTTP requests, appending new configuration lines seamlessly upon resumption.

Re-running `data_collection.sh` skips previously completed configurations and resumes at the exact point of interruption.

## Data Format

JSON Lines files (`.jsonl`) produced by `llama-bench` contain one JSON object per completed evaluation configuration. The raw uploaded `.txt` files combine these individual JSONL outputs into single files per machine.

### Key Schema Fields

Every output record contains 38 fields. Primary analytical fields used in the thesis include:

| Field | Type | Description |
| --- | --- | --- |
| `cpu_info` | string | CPU model description reported by OS kernel. |
| `model_filename` | string | Relative file path to the executed GGUF model binary. |
| `model_type` | string | Architecture type and quantization tag parsed from GGUF metadata. |
| `model_n_params` | int | Total parameter count of the model. |
| `n_threads` | int | Thread allocation count for the benchmark run. |
| `type_k`, `type_v` | string | Data types utilized for KV-cache keys and values (`f16`, `q8_0`, `q4_0`, `iq4_nl`). |
| `flash_attn` | bool / int | Flash Attention state (`1` = enabled, `0` = disabled, `-1` = auto/unspecified). |
| `n_prompt` | int | Length of input prompt evaluated (`512` = prompt processing test, `0` = generation test). |
| `n_gen` | int | Length of generated output tokens (`128` = token generation test, `0` = prefill test). |
| `n_depth` | int | Pre-filled context depth preceding timed evaluation (0, 3584, 15872). |
| `avg_ts` | float | **Primary Metric:** Mean generation speed in tokens/second across repetitions. |
| `stddev_ts` | float | Standard deviation of throughput across repetitions. |
| `samples_ts` | list[float] | Array of raw per-repetition throughput values. |

*Note on Execution Control:* Fields such as `n_gpu_layers` remain set to `0` across all records, confirming CPU-bound execution without discrete hardware acceleration.

### Quick Data Loading Example

```python
import pandas as pd

# Load JSONL benchmark output
df = pd.read_json("bench_results/host-x86_64/phaseB_f16_fa1_t4.jsonl", lines=True)

# Categorize execution phase
df["phase_type"] = df.apply(lambda row: "prefill" if row.n_prompt > 0 else "generation", axis=1)

# Extract core throughput metrics
summary = df[["model_type", "n_threads", "n_depth", "phase_type", "avg_ts", "stddev_ts"]]
print(summary.head())

```

## Published Dataset Coverage Summary

The uploaded text datasets represent completed empirical runs extracted during research collection:

| Machine Dataset | Completed Runs | Notes & Missing Coverage |
| --- | --- | --- |
| **Laptop (Ryzen 5600H)** | Phase B (`f16` KV, `fa=0/1`, context depths 0–15872, full thread sweep; 60/60 rows). Partial Phase B quantized KV. | Quantized KV completed 2 of 9 pairings cleanly, 1 partial. Subsequent phases (C–M) were not run. |
| **Desktop (Pentium G5420)** | Phase A (`Q4_0` full thread sweep; 10/10 rows). | `Q3_K_M` completed 9/10 rows. `Q4_K_M` experienced early thread restarts (detailed in Section 2.6 of thesis). Phases B–M omitted. |
| **Jetson Nano** | Phase A (both target models, full thread sweep; 12/12 rows). | Phase B `f16` initialized but yielded 0 completed records due to OOM/timeout bounds. |

Unfinished phases across the parameter space represent targets for ongoing community benchmarking using `data_collection.sh`.

## Citation

If you use this benchmarking harness or dataset in your research, please cite the Master's thesis:

```bibtex
@mastersthesis{hattay2026cpu,
  author       = {Yassine Hattay},
  title        = {Optimizing CPU-Only LLM Inference: Parameter Sweeps, Hardware Bottlenecks, and Performance Modeling},
  school       = {National School of Computer Sciences (ENSI), University of Manouba},
  year         = {2026},
  month        = {September},
  type         = {Master's Thesis}
}

```

