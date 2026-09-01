#!/usr/bin/env bash
set -uo pipefail
BIN_DIR="./build/bin"
MODELS_DIR="../../models"
OUT_DIR="./bench_results/$(hostname)-$(uname -m)"
WIKITEXT="./wiki.test.raw"
PPL_CHUNKS=20
REPS=5

# Threads is now a SWEPT axis, not a fixed value — this is deliberate: the
# optimal thread count on this machine will not transfer to a Raspberry Pi or
# any other CPU, so it needs to be measured on every machine you run this on,
# not assumed. Override via env, e.g. THREADS_LIST="1,2,4" on a 4-core Pi.
THREADS_LIST="${THREADS_LIST:-$(printf '%s\n' 1 2 4 6 8 "$(nproc)" | sort -nu | paste -sd,)}"

QWEN_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
QWEN_1_5B="$MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

mkdir -p "$OUT_DIR"

BENCH="$BIN_DIR/llama-bench"
PPL="$BIN_DIR/llama-perplexity"
SERVER="$BIN_DIR/llama-server"

[[ -x "$BENCH" ]] || { echo "!! llama-bench not found"; exit 1; }
[[ -f "$WIKITEXT" ]] || { echo "!! $WIKITEXT missing"; exit 1; }

log() { echo -e "\n=== $1 ===\n"; }

wait_for_server() {
    local port="$1" timeout="${2:-90}" waited=0
    until curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 1; waited=$((waited+1))
        if [[ $waited -ge $timeout ]]; then return 1; fi
    done
    return 0
}

# ==============================================================================
# Resume support v2: checks ROW COUNT, not just "file exists"
# ==============================================================================
skip_if_complete() {
    local out="$1" expected="$2" label="$3"
    local actual=0
    [[ -f "$out" ]] && actual=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [[ "$actual" -ge "$expected" ]]; then
        echo "SKIP (complete: $actual/$expected rows): $label -> $out"
        return 0
    elif [[ "$actual" -gt 0 ]]; then
        echo "RE-RUNNING FROM SCRATCH (partial: $actual/$expected rows -- llama-bench can't"
        echo "  resume mid-sweep, so this thread's file is rebuilt clean, not appended to): $label -> $out"
    fi
    return 1
}

# ==============================================================================
# KV-CACHE AND ATTENTION SWEEPS
# Split into 3 independently-resumable sweeps.
# ==============================================================================
sweep_kv_fa0() {
    log "KV-CACHE SWEEP (fa=0): f16 baseline, per-thread so interruptions cost 1 thread, not all"
    local expected=6   # 3 depths x (prompt,gen)
    IFS=',' read -ra tvals <<< "$THREADS_LIST"
    for t in "${tvals[@]}"; do
        local out="$OUT_DIR/sweep_kv_fa0_t${t}.jsonl"
        skip_if_complete "$out" "$expected" "fa=0 t=$t" && continue
        "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 \
            -ctk f16 -ctv f16 -fa 0 -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
    done
}

sweep_kv_fa1() {
    log "KV-CACHE SWEEP (fa=1): f16 baseline, per-thread"
    local expected=6
    IFS=',' read -ra tvals <<< "$THREADS_LIST"
    for t in "${tvals[@]}"; do
        local out="$OUT_DIR/sweep_kv_fa1_t${t}.jsonl"
        skip_if_complete "$out" "$expected" "fa=1 t=$t" && continue
        "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 \
            -ctk f16 -ctv f16 -fa 1 -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
    done
}

sweep_kv_quantized() {
    log "KV-CACHE SWEEP (quantized, fa=1 only): q8_0/q4_0/iq4_nl, per-thread PER-PAIRING"
    local expected=6
    local kv_types=(q8_0 q4_0 iq4_nl)
    IFS=',' read -ra tvals <<< "$THREADS_LIST"
    for t in "${tvals[@]}"; do
        for ctk in "${kv_types[@]}"; do
            for ctv in "${kv_types[@]}"; do
                local out="$OUT_DIR/sweep_kv_quantized_fa1_t${t}_${ctk}_${ctv}.jsonl"
                skip_if_complete "$out" "$expected" "quantized t=$t ctk=$ctk ctv=$ctv" && continue
                "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,3584,15872 \
                    -ctk "$ctk" -ctv "$ctv" -fa 1 \
                    -t "$t" -ngl 0 -r "$REPS" -o jsonl > "$out"
            done
        done
    done
}

sweep_perplexity() {
    local ppl_threads="${THREADS_LIST%%,*}"
    for ct in f16 q8_0 q4_0; do
        local log_out="$OUT_DIR/sweep_perplexity_${ct}.log"
        skip_if_complete "$log_out" 1 "perplexity sweep ($ct)" && continue
        "$PPL" -m "$QWEN_7B" -f "$WIKITEXT" -c 4096 --chunks "$PPL_CHUNKS" \
            -ctk "$ct" -ctv "$ct" -fa 1 -t "$ppl_threads" \
            2>&1 | tee -a "$log_out" | grep -i "final estimate" || true
    done
}

# ==============================================================================
# MODEL AND QUANTIZATION SWEEPS
# These hold KV-cache and Flash Attention settings fixed and sweep the model itself.
# ==============================================================================
sweep_model_comparison() {
    local out="$OUT_DIR/sweep_model_comparison.jsonl"
    skip_if_complete "$out" 1 "model comparison sweep" && return
    log "MODEL SWEEP: cross-model baseline (threads=swept)"
    local models=(
        "qwen2.5-0.5b-instruct-q4_k_m.gguf" "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf"   "Qwen3-8B-Q4_K_M.gguf"
        "meta-llama-3.1-8b-instruct-q4_0.gguf" "phi-4-IQ4_NL.gguf"
        "Qwen1.5-MoE-A2.7B-Chat.IQ4_NL.gguf" "DeepSeek-V2-Lite.IQ4_NL.gguf"
        "DeepSeek-Coder-V2-Lite-Base.IQ4_NL.gguf"
    )
    local model_args=()
    for m in "${models[@]}"; do
        path="$MODELS_DIR/$m"
        [[ -e "$path" ]] && model_args+=(-m "$path")
    done
    [[ ${#model_args[@]} -gt 0 ]] && \
        "$BENCH" "${model_args[@]}" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" \
            -o jsonl > "$out"
}

# FIXED: -d instead of -p for context depth; fa 0/1 both valid since only f16.
sweep_flashattn() {
    local out="$OUT_DIR/sweep_flashattn.jsonl"
    skip_if_complete "$out" 1 "Flash Attention sweep" && return
    log "FLASH ATTENTION SWEEP: on/off, dense vs MoE (threads=swept)"
    local models=(
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf" "DeepSeek-V2-Lite.IQ4_NL.gguf"
        "Qwen1.5-MoE-A2.7B-Chat.IQ4_NL.gguf"
    )
    local model_args=()
    for m in "${models[@]}"; do
        path="$MODELS_DIR/$m"
        [[ -f "$path" ]] && model_args+=(-m "$path")
    done
    [[ ${#model_args[@]} -gt 0 ]] && \
        "$BENCH" "${model_args[@]}" -p 512 -n 128 -d 0,3584,15872 -fa 0,1 \
            -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$out"
}

sweep_microbatch() {
    local out="$OUT_DIR/sweep_microbatch.jsonl"
    skip_if_complete "$out" 1 "micro-batch sweep" && return
    log "MICRO-BATCH SWEEP: (threads=swept)"
    "$BENCH" -m "$QWEN_7B" -p 2048 -n 128 -ub 128,256,512,1024,2048 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$out"
}

# FIXED: -d instead of -p; dropped fa=0 entirely (q8_0 needs fa=1, and testing
# f16-at-fa=0 belongs in the KV-cache baseline sweep, not duplicated here).
sweep_context_knee() {
    local out="$OUT_DIR/sweep_context_knee.jsonl"
    skip_if_complete "$out" 1 "context-depth knee sweep" && return
    log "CONTEXT-DEPTH SWEEP: knee detection (threads=swept)"
    "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -d 0,512,1536,3584,7680,15872 \
        -fa 1 -ctk f16,q8_0 -ctv f16,q8_0 -t "$THREADS_LIST" -ngl 0 -r "$REPS" \
        -o jsonl > "$out"
}

# ==============================================================================
# SYSTEM-LEVEL AND SERVING SWEEPS
# These hold the model and quantization fixed and vary OS scheduling.
# ==============================================================================
sweep_cpu_pinning() {
    log "CPU PINNING SWEEP: isolation & pinning"
    local o1="$OUT_DIR/sweep_cpu_pinning_default.jsonl" o2="$OUT_DIR/sweep_cpu_pinning_taskset_0-3.jsonl" \
          o3="$OUT_DIR/sweep_cpu_pinning_taskset_all.jsonl" o4="$OUT_DIR/sweep_cpu_pinning_numa_interleave.jsonl"
    skip_if_complete "$o1" 1 "CPU pinning default" || \
        "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$o1"
    skip_if_complete "$o2" 1 "CPU pinning taskset 0-3" || \
        taskset -c 0-3 "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t 4 -ngl 0 -r "$REPS" -o jsonl > "$o2" 2>/dev/null
    skip_if_complete "$o3" 1 "CPU pinning taskset all" || \
        taskset -c "0-$(( $(nproc)-1 ))" "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$(nproc)" -ngl 0 -r "$REPS" -o jsonl > "$o3" 2>/dev/null
    skip_if_complete "$o4" 1 "CPU pinning numa" || \
        numactl --interleave=all "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$o4" 2>/dev/null
}

# NOTE: this toggles mmap, not mlock — llama-bench has no --mlock flag (only
# llama-server/llama-cli do). If you specifically need real mlock behavior,
# that has to be tested through llama-server instead, not llama-bench.
sweep_mmap() {
    log "MMAP SWEEP: on/off (NOT the same variable as mlock — see comment)"
    local o1="$OUT_DIR/sweep_mmap_on.jsonl" o2="$OUT_DIR/sweep_mmap_off.jsonl"
    skip_if_complete "$o1" 1 "mmap on" || \
        "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -o jsonl > "$o1"
    skip_if_complete "$o2" 1 "mmap off" || \
        "$BENCH" -m "$QWEN_7B" -p 512 -n 128 -t "$THREADS_LIST" -ngl 0 -r "$REPS" -mmp 0 -o jsonl > "$o2"
}

sweep_contention() {
    log "CONTENTION SWEEP: multi-process"
    local csv="$OUT_DIR/sweep_contention.csv"
    # Only write header if file doesn't exist or is empty
    [[ ! -s "$csv" ]] && echo "num_processes,port,predicted_per_second" > "$csv"
    local port_base=8080
    for n in 1 2 4; do
        # Check if we already have data for this process count
        if grep -q "^$n," "$csv" 2>/dev/null; then
            echo "SKIP (already have data for n=$n)"
            continue
        fi
        pkill -f llama-server 2>/dev/null; sleep 2
        for i in $(seq 1 "$n"); do
            port=$((port_base + i))
            "$SERVER" -m "$QWEN_7B" -ngl 0 -t 2 -c 2048 --port "$port" > /dev/null 2>&1 &
        done
        for i in $(seq 1 "$n"); do wait_for_server $((port_base + i)) 90 || true; done
        for i in $(seq 1 "$n"); do
            port=$((port_base + i))
            speed=$(curl -s -X POST "http://127.0.0.1:$port/completion" -H "Content-Type: application/json" -d '{"prompt":"Write a story.","n_predict":64}' | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2)
            echo "$n,$port,${speed:-NA}" >> "$csv"
        done
        pkill -f llama-server 2>/dev/null; sleep 2
    done
}

sweep_speculative() {
    log "SPECULATIVE DECODING SWEEP: x threads (llama-server takes ONE -t value,"
    log "not a list, so this loops explicitly over the same thread values as everywhere else)"
    local csv="$OUT_DIR/sweep_speculative.csv"
    # Only write header if file doesn't exist or is empty
    [[ ! -s "$csv" ]] && echo "threads,config,predicted_per_second" > "$csv"
    IFS=',' read -ra thread_vals <<< "$THREADS_LIST"
    for t in "${thread_vals[@]}"; do
        # Check if we already have data for this thread count
        if grep -q "^$t," "$csv" 2>/dev/null; then
            echo "SKIP (already have data for t=$t)"
            continue
        fi
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

# ------------------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------------------
sweep_kv_fa0
sweep_kv_fa1
sweep_kv_quantized
sweep_perplexity
sweep_model_comparison
sweep_flashattn
sweep_cpu_pinning
sweep_mmap
sweep_contention
sweep_speculative
sweep_microbatch
sweep_context_knee

echo -e "\nDone. Results are .jsonl (one JSON object per line) — read with:"
echo "  import pandas as pd; df = pd.read_json('sweep_kv_fa1_t4.jsonl', lines=True)"