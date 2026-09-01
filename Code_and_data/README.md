# CPU-Only LLM Inference Benchmarking Harness

Benchmark scripts used to collect the data for **"Toward Hardware-Adaptive Pareto Fronts for CPU-Only LLM Inference: An Empirical Study of llama.cpp Across Three Heterogeneous Devices"** (Yassine Hattay, Master's Thesis, ENSI, 2026).

These scripts drive [`llama.cpp`](https://github.com/ggerganov/llama.cpp)'s `llama-bench`, `llama-perplexity`, and `llama-server` binaries through a controlled, single-variable-at-a-time sweep of thread count, quantization format, KV-cache precision, Flash Attention, context depth, and system-level knobs (CPU pinning, NUMA, mmap, concurrency, speculative decoding) — with everything left un-offloaded (`-ngl 0`) so every measurement reflects pure CPU throughput.

An interactive calculator implementing the roofline model derived from this data is available at **https://yassine-hattay.github.io/LLMs_inference_on_CPU/index.html**. The full write-up, including all figures and the mathematical derivation of the model, is contained within the thesis itself.

## Repository Structure

```text
Code_and_data/
├── data_collection.sh             # Master unified benchmarking harness script
├── laptop_amp_cpu_data.txt        # Consolidated raw output collected on AMD Ryzen 5 5600H
├── desktop_pc_results.txt         # Consolidated raw output collected on Intel Pentium G5420
├── jetson_data.txt                # Consolidated raw output collected on NVIDIA Jetson Nano
├── LICENSE-DATA                   # License terms for dataset and harness code
└── README.md                      # Documentation (this file)
```

| File | Target Machine Specs | Role / Contents |
| --- | --- | --- |
| `data_collection.sh` | Portable (Linux/POSIX environment) | Parametric driver script running isolated measurement sweeps. Auto-detects core topology or accepts manual overrides. |
| `laptop_amp_cpu_data.txt` | AMD Ryzen 5 5600H @ 3.30 GHz (6C/12T, Zen 3), 16 GB DDR4 | Consolidated raw benchmark output (mid-range, compute-rich baseline). |
| `desktop_pc_results.txt` | Intel Pentium Gold G5420 @ 3.80 GHz (2C/4T), 4 GB DDR4 | Consolidated raw benchmark output (low-end, bandwidth-starved baseline). |
| `jetson_data.txt` | ARM Cortex-A57 rev1 (4C/4T), 3.9 GB LPDDR4 | Consolidated raw benchmark output (embedded, low-bandwidth baseline). |

Each `..._data.txt` file is a plain-text concatenation of every `.jsonl` file produced by the harness in its `bench_results/<host>-<arch>/` directory, concatenated under a `FILE: <name> [JSONL]` header block. It provides a single, human-readable export for inspection and downstream parsing. See [Data Format](#data-format) for schema details.

*Note on hardware fleet:* A fourth machine, a Raspberry Pi 3B (Cortex-A53, 0.9 GB LPDDR2), was included in the original hardware testing fleet; however, memory constraints prevented it from completing a usable sweep row, so no output file is included for it.

## Prerequisites

1. **Built `llama.cpp` Binaries:**
   The script expects to run relative to a `llama.cpp` build containing `./build/bin/llama-bench` (required for primary benchmarks), plus `./build/bin/llama-perplexity` and `./build/bin/llama-server` (required for perplexity and server-driven sweeps). Ensure GPU offloading is not required; every execution sweep passes `-ngl 0` explicitly.
2. **WikiText-2 Test Dataset:**
   A standard WikiText-2 raw test split file located at `./wiki.test.raw` (required for perplexity validation sweeps).
3. **GGUF Model Weights:**
   Place required `.gguf` model files in the target models directory. Path resolution can be set dynamically using environment variables (see [Models Directory & Paths](#models-directory-and-path-configuration)).
4. **Standard Linux Utilities:**
   `bash`, `curl`, `grep`, `cut`. For CPU isolation and NUMA testing, `taskset` (`util-linux`) and `numactl` are required. If missing, those specific sub-tests will register a failure and the harness will proceed to subsequent sweeps.
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

Model filenames are case-sensitive and must match the exact names below. If a model file is missing, the harness gracefully skips the dependent sweep without halting execution:

| Filename | Type | Total / Active Params | Sweep Usage |
| --- | --- | --- | --- |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Dense | 0.5B | `sweep_model_comparison` |
| `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | Dense | 1.5B | `sweep_model_comparison`, `sweep_speculative` (Draft), Jetson model-size comparison |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Dense | 7.6B | Primary model for core parameters and KV-cache sweeps |
| `Qwen2.5-7B-Instruct-Q4_0.gguf` | Dense | 7.6B | `sweep_model_comparison` (Pentium quantization sweep) |
| `Qwen2.5-7B-Instruct-Q3_K_M.gguf` | Dense | 7.6B | `sweep_model_comparison` (Pentium quantization sweep) |
| `Qwen3-8B-Q4_K_M.gguf` | Dense | 8B | `sweep_model_comparison` |
| `meta-llama-3.1-8b-instruct-q4_0.gguf` | Dense | 8B | `sweep_model_comparison` |
| `phi-4-IQ4_NL.gguf` | Dense | 14B | `sweep_model_comparison` |
| `Qwen1.5-MoE-A2.7B-Chat.IQ4_NL.gguf` | MoE | 14B total / 2.7B active | `sweep_model_comparison`, `sweep_flashattn` |
| `DeepSeek-V2-Lite.IQ4_NL.gguf` | MoE | 16B total / 2.4B active | `sweep_model_comparison`, `sweep_flashattn` |
| `DeepSeek-Coder-V2-Lite-Base.IQ4_NL.gguf` | MoE | 16B total / 2.4B active | `sweep_model_comparison` |

*Note on MoE Models:* MoE model definitions are implemented in the harness for architecture evaluations. Due to memory bandwidth and execution limits on the tested hardware, MoE sweeps did not reach full completion in the published raw dataset.

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

## Sweep Reference

Each sweep isolates a single operational axis while fixing remaining execution variables.

| Sweep Category | Swept Variables | Fixed Parameters | Script Function |
| --- | --- | --- | --- |
| **KV-Cache & Attention** | `-ctk`/`-ctv` (`f16`, `q8_0`, `q4_0`, `iq4_nl`), `-fa` (0/1), Context Depth (0, 3584, 15872) | Model: `Qwen2.5-7B-Instruct-Q4_K_M`, Thread sweep | `sweep_kv_fa0`, `sweep_kv_fa1`, `sweep_kv_quantized` |
| **Perplexity Validation** | KV format (`f16`, `q8_0`, `q4_0`) | Model, context, threads | `sweep_perplexity` |
| **Model Comparison** | Model architecture (Dense vs. MoE roster) | Fixed quantization per model, Thread sweep | `sweep_model_comparison` |
| **Flash Attention Scaling** | `-fa` (0/1), Context Depth | Thread sweep | `sweep_flashattn` |
| **CPU Isolation & Pinning** | Default OS scheduler, `taskset` physical cores, `taskset` all cores, `numactl --interleave` | Fixed model and quantization | `sweep_cpu_pinning` |
| **Memory Mapping** | `-mmp` (mmap enabled/disabled) | Fixed model and quantization | `sweep_mmap` |
| **Multi-Process Contention** | Concurrent `llama-server` instances (1, 2, 4) | 2 threads/server via HTTP `/completion` | `sweep_contention` |
| **Speculative Decoding** | Thread count, Draft model active/inactive (`-md` 1.5B) | HTTP `/completion` benchmarks | `sweep_speculative` |
| **Micro-Batch Sweep** | Micro-batch size `-ub` (128, 256, 512, 1024, 2048) | Thread sweep | `sweep_microbatch` |
| **Context-Depth Knee** | Context Depth (0–15872) × `-ctk`/`-ctv` (`f16`, `q8_0`) | `-fa 1`, Thread sweep | `sweep_context_knee` |

### Execution Drivers

1. **`llama-bench`:** Drives KV-cache, model comparison, flash attention, micro-batch, and context-depth sweeps. Runs fixed workloads for `$REPS` repetitions and outputs JSON Lines (`-o jsonl`).
2. **`llama-perplexity`:** Drives perplexity validation sweeps. Computes perplexity over `wiki.test.raw` and logs raw estimates to `.log` files.
3. **`llama-server`:** Drives contention and speculative decoding sweeps. Launches an HTTP server process, fires batch `/completion` requests, and records latency statistics to CSV files.

## Resumability Mechanism

The harness incorporates atomic output logging to sustain unattended long-duration benchmark runs across system restarts or interruptions:

* **File Granularity:** Each distinct thread count, quantization, or hardware configuration writes to its own specific output file in `./bench_results/$(hostname)-$(uname -m)/`.
* **Completion Verification:** Before executing a benchmark step, the harness checks existing row counts against expected repetition outputs. Partial or interrupted outputs are recognized as incomplete and re-executed cleanly from scratch.
* **Server CSV Checks:** CSV-based sweeps (contention, speculative decoding) verify key existence prior to firing HTTP requests, appending new configuration lines seamlessly upon resumption.

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
df = pd.read_json("bench_results/host-x86_64/sweep_kv_fa1_t4.jsonl", lines=True)

# Categorize execution phase
df["sweep_type"] = df.apply(lambda row: "prefill" if row.n_prompt > 0 else "generation", axis=1)

# Extract core throughput metrics
summary = df[["model_type", "n_threads", "n_depth", "sweep_type", "avg_ts", "stddev_ts"]]
print(summary.head())
```

## Published Dataset Coverage Summary

The uploaded text datasets represent completed empirical runs extracted during research collection:

| Machine Dataset | Completed Runs | Notes & Missing Coverage |
| --- | --- | --- |
| **Laptop (Ryzen 5600H)** | KV-cache and attention sweeps (`f16` KV, `fa=0/1`, all 3 depths, full 5-thread sweep). | Quantized-KV: 3/9 pairings, thread=1 only. Perplexity, model-comparison sweeps: not run. |
| **Desktop (Pentium G5420)** | `Q4_0` quantization sweep (full, 5 threads). `Q3_K_M` (11/12 rows). | `Q4_K_M`: repeatedly restarted, never reached thread > 4. Other sweeps: not run. |
| **Jetson Nano** | Model-size comparison sweep (both target models, 3-thread sweep). | KV-cache and attention sweeps: 0 rows, file present but empty. |

Unfinished sweeps across the parameter space represent targets for ongoing community benchmarking using `data_collection.sh`.

## Citation

If you use this benchmarking harness or dataset in your research, please cite the Master's thesis:

```bibtex
@mastersthesis{hattay2026cpu,
  author       = {Yassine Hattay},
  title        = {Toward Hardware-Adaptive Pareto Fronts for CPU-Only LLM Inference: An Empirical Study of \texttt{llama.cpp} Across Three Heterogeneous Devices},
  school       = {National School of Computer Sciences (ENSI), University of Manouba},
  year         = {2026},
  month        = {September},
  type         = {Master's Thesis}
}
```
