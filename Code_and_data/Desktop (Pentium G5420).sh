#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
# UNIVERSAL PARETO SWEEP SCRIPT v6.2 (Per-Model Resumable & Bug-Free)
# ==============================================================================
# Fixes the "0\n0" arithmetic error by safely handling grep -c exit codes.
# ==============================================================================

BIN_DIR="./build/bin"
MODELS_DIR="../../models"
OUT_DIR="./bench_results/$(hostname)-$(uname -m)"
WIKITEXT="./wiki.test.raw"
PPL_CHUNKS=20
REPS=5

THREADS_LIST="${THREADS_LIST:-$(printf '%s\n' 1 2 4 6 8 "$(nproc)" | sort -nu | paste -sd,)}"

QWEN_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
QWEN_05B="$MODELS_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
QWEN_1_5B="$MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

mkdir -p "$OUT_DIR"
BENCH="$BIN_DIR/llama-bench"
PPL="$BIN_DIR/llama-perplexity"
SERVER="$BIN_DIR/llama-server"

if [[ ! -x "$BENCH" ]]; then
  echo "!! llama-bench not found at $BIN_DIR. Build first."
  exit 1
fi
if [[ ! -f "$WIKITEXT" ]]; then
  echo "!! $WIKITEXT missing. Download it first."
  exit 1
fi

log() { echo -e "\n=== $1 ===\n"; }

wait_for_server() {
  local port="$1" timeout="${2:-90}" waited=0
  until curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"status":"ok"'; do
    sleep 1; waited=$((waited+1))
    if [[ $waited -ge $timeout ]]; then return 1; fi
  done
  return 0
}

IFS=',' read -ra _TVALS <<< "$THREADS_LIST"
NUM_THREADS=${#_TVALS[@]}

# ==============================================================================
# PHASE A: Quantization Pareto (PER-MODEL RESUMABLE)
# ==============================================================================
phase_a() {
  log "PHASE A: Quantization Pareto (Per-Model Resumable)"
  local out="$OUT_DIR/phaseA_quant.jsonl"
  local models=(
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    "Qwen2.5-7B-Instruct-Q4_0.gguf"
    "Qwen2.5-7B-Instruct-Q3_K_M.gguf"
    "Qwen2.5-7B-Instruct-IQ4_XS.gguf"
    "Qwen2.5-7B-Q8_0.gguf"
  )
  
  local existing_models=()
  local total_expected=0
  for m in "${models[@]}"; do
    if [[ -f "$MODELS_DIR/$m" ]]; then
      existing_models+=("$m")
      total_expected=$((total_expected + NUM_THREADS * 2))
    fi
  done

  local actual=0
  [[ -f "$out" ]] && actual=$(wc -l < "$out" 2>/dev/null || echo 0)
  
  if [[ "$actual" -ge "$total_expected" ]]; then
    echo "✅ SKIP (complete: $actual/$total_expected rows): Phase A"
    return 0
  elif [[ "$actual" -gt 0 ]]; then
    echo "⚠️ RESUMING Phase A (partial: $actual/$total_expected rows)"
  else
    echo "🚀 STARTING: Phase A"
  fi

  for m in "${existing_models[@]}"; do
    local model_expected=$((NUM_THREADS * 2))
    local model_actual=0
    if [[ -f "$out" ]]; then
      model_actual=$(grep -c "$m" "$out" || true)
    fi
    
    if [[ "$model_actual" -ge "$model_expected" ]]; then
      echo "  ✅ SKIP (complete): $m"
      continue
    fi
    
    if [[ "$model_actual" -gt 0 ]]; then
      echo "  ⚠️ RE-RUNNING (partial): $m (stripping incomplete rows)"
      grep -v "$m" "$out" > "${out}.tmp" && mv "${out}.tmp" "$out"
    else
      echo "  🚀 STARTING: $m"
    fi
    
    "$BENCH" -m "$MODELS_DIR/$m" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl >> "$out"
  done
}

# ==============================================================================
# PHASE B: KV Cache & Attention Physics
# ==============================================================================
phase_b_fa0() {
  log "PHASE B (fa=0): f16 baseline"
  local expected=6
  IFS=',' read -ra tvals <<< "$THREADS_LIST"
  for t in "${tvals[@]}"; do
    local out="$OUT_DIR/phaseB_f16_fa0_t${t}.jsonl"
    local actual=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [[ "$actual" -ge "$expected" ]]; then echo "  ✅ SKIP: fa=0 t=$t"; continue; fi
    [[ "$actual" -gt 0 ]] && echo "  ⚠️ RE-RUNNING (partial): fa=0 t=$t" || echo "  🚀 STARTING: fa=0 t=$t"
    "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 -ctk f16 -ctv f16 -fa 0 -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
  done
}

phase_b_fa1() {
  log "PHASE B (fa=1): f16 baseline"
  local expected=6
  IFS=',' read -ra tvals <<< "$THREADS_LIST"
  for t in "${tvals[@]}"; do
    local out="$OUT_DIR/phaseB_f16_fa1_t${t}.jsonl"
    local actual=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [[ "$actual" -ge "$expected" ]]; then echo "  ✅ SKIP: fa=1 t=$t"; continue; fi
    [[ "$actual" -gt 0 ]] && echo "  ⚠️ RE-RUNNING (partial): fa=1 t=$t" || echo "  🚀 STARTING: fa=1 t=$t"
    "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 -ctk f16 -ctv f16 -fa 1 -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
  done
}

phase_b_quantized() {
  log "PHASE B (quantized, fa=1 only)"
  local expected=54
  IFS=',' read -ra tvals <<< "$THREADS_LIST"
  for t in "${tvals[@]}"; do
    local out="$OUT_DIR/phaseB_quantized_fa1_t${t}.jsonl"
    local actual=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [[ "$actual" -ge "$expected" ]]; then echo "  ✅ SKIP: quantized t=$t"; continue; fi
    [[ "$actual" -gt 0 ]] && echo "  ⚠️ RE-RUNNING (partial): quantized t=$t" || echo "  🚀 STARTING: quantized t=$t"
    "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 -ctk q8_0,q4_0,iq4_nl -ctv q8_0,q4_0,iq4_nl -fa 1 -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
  done
}

phase_b_ppl() {
  log "PHASE B (perplexity)"
  local ppl_threads="${THREADS_LIST%%,*}"
  for ct in f16 q8_0 q4_0; do
    local log_out="$OUT_DIR/phaseB_ppl_${ct}.log"
    if [[ -s "$log_out" ]] && grep -q "final estimate" "$log_out"; then
      echo "  ✅ SKIP: perplexity $ct"
      continue
    fi
    echo "  🚀 STARTING: perplexity $ct"
    "$PPL" -m "$QWEN_7B" -f "$WIKITEXT" -c 4096 --chunks "$PPL_CHUNKS" -ctk "$ct" -ctv "$ct" -fa 1 -t "$ppl_threads" 2>&1 | tee "$log_out" | grep -i "final estimate" || true
  done
}

# ==============================================================================
# PHASE C: Cross-Architecture Baseline (PER-MODEL RESUMABLE)
# ==============================================================================
phase_c() {
  log "PHASE C: Cross-model baseline (Per-Model Resumable)"
  local out="$OUT_DIR/phaseC_all_models.jsonl"
  local models=(
    "qwen2.5-0.5b-instruct-q4_k_m.gguf" "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf"   "Qwen3-8B-Q4_K_M.gguf"
    "meta-llama-3.1-8b-instruct-q4_0.gguf" "phi-4-IQ4_NL.gguf"
    "Qwen1.5-MoE-A2.7B-Chat.IQ4_NL.gguf" "DeepSeek-V2-Lite.IQ4_NL.gguf"
    "DeepSeek-Coder-V2-Lite-Base.IQ4_NL.gguf"
  )
  
  local existing_models=()
  local total_expected=0
  for m in "${models[@]}"; do
    if [[ -e "$MODELS_DIR/$m" ]]; then
      existing_models+=("$m")
      total_expected=$((total_expected + NUM_THREADS * 2))
    fi
  done

  local actual=0
  [[ -f "$out" ]] && actual=$(wc -l < "$out" 2>/dev/null || echo 0)
  
  if [[ "$actual" -ge "$total_expected" ]]; then
    echo "✅ SKIP (complete: $actual/$total_expected rows): Phase C"
    return 0
  elif [[ "$actual" -gt 0 ]]; then
    echo "⚠️ RESUMING Phase C (partial: $actual/$total_expected rows)"
  else
    echo "🚀 STARTING: Phase C"
  fi

  for m in "${existing_models[@]}"; do
    local model_expected=$((NUM_THREADS * 2))
    local model_actual=0
    if [[ -f "$out" ]]; then
      model_actual=$(grep -c "$m" "$out" || true)
    fi
    
    if [[ "$model_actual" -ge "$model_expected" ]]; then
      echo "  ✅ SKIP (complete): $m"
      continue
    fi
    
    if [[ "$model_actual" -gt 0 ]]; then
      echo "  ⚠️ RE-RUNNING (partial): $m (stripping incomplete rows)"
      grep -v "$m" "$out" > "${out}.tmp" && mv "${out}.tmp" "$out"
    else
      echo "  🚀 STARTING: $m"
    fi
    
    "$BENCH" -m "$MODELS_DIR/$m" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl >> "$out"
  done
}

# ==============================================================================
# PHASE D: Flash Attention on/off (PER-MODEL RESUMABLE)
# ==============================================================================
phase_d() {
  log "PHASE D: Flash Attention on/off (Per-Model Resumable)"
  local out="$OUT_DIR/phaseD_flashattn.jsonl"
  local models=(
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf" "DeepSeek-V2-Lite.IQ4_NL.gguf"
    "Qwen1.5-MoE-A2.7B-Chat.IQ4_NL.gguf"
  )
  
  local existing_models=()
  local total_expected=0
  for m in "${models[@]}"; do
    if [[ -f "$MODELS_DIR/$m" ]]; then
      existing_models+=("$m")
      total_expected=$((total_expected + 3 * 2 * NUM_THREADS * 2))
    fi
  done

  local actual=0
  [[ -f "$out" ]] && actual=$(wc -l < "$out" 2>/dev/null || echo 0)
  
  if [[ "$actual" -ge "$total_expected" ]]; then
    echo "✅ SKIP (complete: $actual/$total_expected rows): Phase D"
    return 0
  elif [[ "$actual" -gt 0 ]]; then
    echo "⚠️ RESUMING Phase D (partial: $actual/$total_expected rows)"
  else
    echo "🚀 STARTING: Phase D"
  fi

  for m in "${existing_models[@]}"; do
    local model_expected=$((3 * 2 * NUM_THREADS * 2))
    local model_actual=0
    if [[ -f "$out" ]]; then
      model_actual=$(grep -c "$m" "$out" || true)
    fi
    
    if [[ "$model_actual" -ge "$model_expected" ]]; then
      echo "  ✅ SKIP (complete): $m"
      continue
    fi
    
    if [[ "$model_actual" -gt 0 ]]; then
      echo "  ⚠️ RE-RUNNING (partial): $m (stripping incomplete rows)"
      grep -v "$m" "$out" > "${out}.tmp" && mv "${out}.tmp" "$out"
    else
      echo "  🚀 STARTING: $m"
    fi
    
    "$BENCH" -m "$MODELS_DIR/$m" -p 512 -n 128 -d 0,3584,15872 -fa 0,1 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl >> "$out"
  done
}

# ==============================================================================
# PHASE F, G, I, K1, L, M (Standard Resumption)
# ==============================================================================
phase_f() {
  log "PHASE F: CPU isolation & pinning"
  local o1="$OUT_DIR/phaseF_default.jsonl"
  local exp1=$((NUM_THREADS * 2))
  local act1=$(wc -l < "$o1" 2>/dev/null || echo 0)
  [[ "$act1" -ge "$exp1" ]] && echo "  ✅ SKIP: default" || { echo "  🚀 STARTING: default"; "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$o1"; }

  local o2="$OUT_DIR/phaseF_taskset_0-3.jsonl"
  local act2=$(wc -l < "$o2" 2>/dev/null || echo 0)
  [[ "$act2" -ge 2 ]] && echo "  ✅ SKIP: taskset 0-3" || { echo "  🚀 STARTING: taskset 0-3"; taskset -c 0-3 "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t 4 -ngl 0 -r "$REPS" -o jsonl > "$o2" 2>/dev/null; }

  local o3="$OUT_DIR/phaseF_taskset_all.jsonl"
  local act3=$(wc -l < "$o3" 2>/dev/null || echo 0)
  [[ "$act3" -ge 2 ]] && echo "  ✅ SKIP: taskset all" || { echo "  🚀 STARTING: taskset all"; taskset -c "0-$(( $(nproc)-1 ))" "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$(nproc)" -ngl 0 -r "$REPS" -o jsonl > "$o3" 2>/dev/null; }

  local o4="$OUT_DIR/phaseF_numa_interleave.jsonl"
  local exp4=$((NUM_THREADS * 2))
  local act4=$(wc -l < "$o4" 2>/dev/null || echo 0)
  [[ "$act4" -ge "$exp4" ]] && echo "  ✅ SKIP: numa" || { echo "  🚀 STARTING: numa"; numactl --interleave=all "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$o4" 2>/dev/null; }
}

phase_g() {
  log "PHASE G: mmap on/off"
  local o1="$OUT_DIR/phaseG_mmap_on.jsonl"
  local exp1=$((NUM_THREADS * 2))
  local act1=$(wc -l < "$o1" 2>/dev/null || echo 0)
  [[ "$act1" -ge "$exp1" ]] && echo "  ✅ SKIP: mmap on" || { echo "  🚀 STARTING: mmap on"; "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$o1"; }

  local o2="$OUT_DIR/phaseG_mmap_off.jsonl"
  local exp2=$((NUM_THREADS * 2))
  local act2=$(wc -l < "$o2" 2>/dev/null || echo 0)
  [[ "$act2" -ge "$exp2" ]] && echo "  ✅ SKIP: mmap off" || { echo "  🚀 STARTING: mmap off"; "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -mmp 0 -o jsonl > "$o2"; }
}

phase_i() {
  log "PHASE I: multi-process contention"
  local csv="$OUT_DIR/phaseI_contention.csv"
  [[ ! -s "$csv" ]] && echo "num_processes,port,predicted_per_second" > "$csv"
  local port_base=8080
  for n in 1 2 4; do
    if grep -q "^$n," "$csv" 2>/dev/null; then echo "  ✅ SKIP: n=$n"; continue; fi
    echo "  🚀 STARTING: n=$n concurrent server(s)"
    pkill -f llama-server 2>/dev/null; sleep 2
    for i in $(seq 1 "$n"); do
      port=$((port_base + i))
      "$SERVER" -m "$QWEN_7B" -ngl 0 -t 2 -c 2048 --port "$port" > /dev/null 2>&1 &
    done
    for i in $(seq 1 "$n"); do wait_for_server $((port_base + i)) 90 || echo "!! server on port $((port_base+i)) failed"; done
    for i in $(seq 1 "$n"); do
      port=$((port_base + i))
      speed=$(curl -s -X POST "http://127.0.0.1:$port/completion" -H "Content-Type: application/json" -d '{"prompt":"Write a story.","n_predict":64}' | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2)
      echo "$n,$port,${speed:-NA}" >> "$csv"
    done
    pkill -f llama-server 2>/dev/null; sleep 2
  done
}

phase_k1_speculative() {
  log "PHASE K1: speculative decoding x threads"
  local csv="$OUT_DIR/phaseK1_speculative.csv"
  [[ ! -s "$csv" ]] && echo "threads,config,predicted_per_second" > "$csv"
  IFS=',' read -ra thread_vals <<< "$THREADS_LIST"
  for t in "${thread_vals[@]}"; do
    if grep -q "^$t," "$csv" 2>/dev/null; then echo "  ✅ SKIP: t=$t"; continue; fi
    echo "  🚀 STARTING: Speculative decoding t=$t"
    pkill -f llama-server 2>/dev/null; sleep 2
    "$SERVER" -m "$QWEN_7B" -ngl 0 -t "$t" -c 4096 --port 8080 > /dev/null 2>&1 &
    if wait_for_server 8080 90; then
      speed=$(curl -s -X POST "http://127.0.0.1:8080/completion" -H "Content-Type: application/json" -d '{"prompt":"Write code.","n_predict":128}' | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2)
      echo "$t,baseline_no_draft,${speed:-NA}" >> "$csv"
    fi
    pkill -f llama-server 2>/dev/null; sleep 2

    "$SERVER" -m "$QWEN_7B" -md "$QWEN_1_5B" -ngl 0 -t "$t" -c 4096 --port 8080 > /dev/null 2>&1 &
    if wait_for_server 8080 120; then
      speed=$(curl -s -X POST "http://127.0.0.1:8080/completion" -H "Content-Type: application/json" -d '{"prompt":"Write code.","n_predict":128}' | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2)
      echo "$t,with_1.5B_draft,${speed:-NA}" >> "$csv"
    fi
    pkill -f llama-server 2>/dev/null; sleep 2
  done
}

phase_l() {
  log "PHASE L: micro-batch sweep"
  local out="$OUT_DIR/phaseL_microbatch.jsonl"
  local expected=$((5 * NUM_THREADS * 2))
  local actual=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [[ "$actual" -ge "$expected" ]]; then echo "✅ SKIP: Phase L"; return 0; fi
  [[ "$actual" -gt 0 ]] && echo "⚠️ RE-RUNNING (partial): Phase L" || echo "🚀 STARTING: Phase L"
  "$BENCH" -m "$QWEN_7B" -p 2048 -n 128 -ub 128,256,512,1024,2048 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$out"
}

phase_m() {
  log "PHASE M: context-depth knee detection"
  local out="$OUT_DIR/phaseM_context_knee.jsonl"
  local expected=$((4 * 6 * NUM_THREADS * 2))
  local actual=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [[ "$actual" -ge "$expected" ]]; then echo "✅ SKIP: Phase M"; return 0; fi
  [[ "$actual" -gt 0 ]] && echo "⚠️ RE-RUNNING (partial): Phase M" || echo "🚀 STARTING: Phase M"
  "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,512,1536,3584,7680,15872 -fa 1 -ctk f16,q8_0 -ctv f16,q8_0 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$out"
}

# ==============================================================================
# EXECUTION
# ==============================================================================
echo "🚀 STARTING UNIVERSAL PARETO SWEEP v6.2"
echo "🖥️ Hostname: $(hostname)"
echo "🧠 Threads: $THREADS_LIST"
echo "📂 Output: $OUT_DIR"
echo "================================================================"

phase_a
phase_b_fa0
phase_b_fa1
phase_b_quantized
phase_b_ppl
phase_c
phase_d
phase_f
phase_g
phase_i
phase_k1_speculative
phase_l
phase_m

echo -e "\n🏆 ALL PHASES COMPLETE. Results saved in $OUT_DIR/"