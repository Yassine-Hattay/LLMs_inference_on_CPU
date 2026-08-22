#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
# UNIVERSAL PARETO SWEEP SCRIPT (Jetson Nano Optimized)
# ==============================================================================
# Tailored for Jetson Nano (4GB RAM, 4-core ARM64).
# Only tests the 2 available models to save time and prevent OOM.
# ==============================================================================

BIN_DIR="./build/bin"
MODELS_DIR="../models"
OUT_DIR="./bench_results/$(hostname)-$(uname -m)"
WIKITEXT="./wiki.test.raw"
PPL_CHUNKS=20
REPS=3

# Jetson Nano has 4 cores, cap at 4 threads
THREADS_LIST="${THREADS_LIST:-$(printf '%s\n' 1 2 4 "$(nproc)" | sort -nu | paste -sd,)}"

QWEN_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
QWEN_1_5B="$MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

mkdir -p "$OUT_DIR"
BENCH="$BIN_DIR/llama-bench"
PPL="$BIN_DIR/llama-perplexity"

if [[ ! -x "$BENCH" ]]; then echo "!! llama-bench not found. Build first."; exit 1; fi
if [[ ! -f "$WIKITEXT" ]]; then echo "!! $WIKITEXT missing. Download it first."; exit 1; fi

log() { echo -e "\n=== $1 ===\n"; }

skip_if_complete() {
  local out="$1" expected="$2" label="$3"
  local actual=0
  [[ -f "$out" ]] && actual=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [[ "$actual" -ge "$expected" ]]; then
    echo "✅ SKIP (complete: $actual/$expected rows): $label"
    return 0
  elif [[ "$actual" -gt 0 ]]; then
    echo "⚠️ RE-RUNNING (partial: $actual/$expected rows): $label"
  else
    echo "🚀 STARTING: $label"
  fi
  return 1
}

# PHASE A: Quantization Pareto (Only test available models)
phase_a() {
  log "PHASE A: Quantization Pareto"
  local out="$OUT_DIR/phaseA_quant.jsonl"
  local models=( "Qwen2.5-7B-Instruct-Q4_K_M.gguf" "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" )
  local model_args=()
  for m in "${models[@]}"; do
    [[ -f "$MODELS_DIR/$m" ]] && model_args+=(-m "$MODELS_DIR/$m")
  done
  local expected=$(( ${#model_args[@]} * 3 * 2 )) # 2 models, 3 threads (1,2,4), 2 tests
  skip_if_complete "$out" "$expected" "Phase A" && return
  if [[ ${#model_args[@]} -gt 0 ]]; then
    "$BENCH" "${model_args[@]}" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$out"
  fi
}

# PHASE B: KV Cache & Attention (Per-pairing split for safety)
phase_b_fa0() {
  log "PHASE B (fa=0): f16 baseline"
  local expected=6
  for t in 1 2 4; do
    local out="$OUT_DIR/phaseB_f16_fa0_t${t}.jsonl"
    skip_if_complete "$out" "$expected" "fa=0 t=$t" && continue
    "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 -ctk f16 -ctv f16 -fa 0 -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
  done
}

phase_b_fa1() {
  log "PHASE B (fa=1): f16 baseline"
  local expected=6
  for t in 1 2 4; do
    local out="$OUT_DIR/phaseB_f16_fa1_t${t}.jsonl"
    skip_if_complete "$out" "$expected" "fa=1 t=$t" && continue
    "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 -ctk f16 -ctv f16 -fa 1 -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
  done
}

phase_b_quantized() {
  log "PHASE B (quantized, fa=1): per-pairing"
  local expected=6
  local kv_types=(q8_0 q4_0 iq4_nl)
  for t in 1 2 4; do
    for ctk in "${kv_types[@]}"; do
      for ctv in "${kv_types[@]}"; do
        local out="$OUT_DIR/phaseB_quantized_fa1_t${t}_${ctk}_${ctv}.jsonl"
        skip_if_complete "$out" "$expected" "quantized t=$t ctk=$ctk ctv=$ctv" && continue
        "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 -ctk "$ctk" -ctv "$ctv" -fa 1 -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
      done
    done
  done
}

echo "🚀 STARTING JETSON NANO PARETO SWEEP"
echo "🖥️ Hostname: $(hostname)"
echo "🧠 Threads: $THREADS_LIST"
echo "📂 Output: $OUT_DIR"
echo "================================================================"

phase_a
phase_b_fa0
phase_b_fa1
phase_b_quantized

echo -e "\n🏆 ALL PHASES COMPLETE. Results saved in $OUT_DIR/"
