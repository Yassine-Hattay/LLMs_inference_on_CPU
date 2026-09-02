# Updated Presentation Script — Thesis Defense
## Toward Hardware-Adaptive Pareto Fronts for CPU-Only LLM Inference
**Target: 20 minutes | ~2,600 words**

---

## Slide 1 — Title Slide (0:00 → 0:30, 30s)

**Visual Elements:**
- Title centered, large font
- University logos (ENSI, University of Manouba) side by side at top
- Your name and "Master's Thesis Defense" below
- Date: September 16, 2026
- Subtle background: CPU chip silhouette or abstract network graph

**Script:**
"Good afternoon, members of the jury. My thesis asks a simple but critical question: if you don't have a GPU — and most computers in the world don't — what settings actually get you the most tokens per second from a large language model, and what does it cost in quality? I studied this empirically using llama.cpp across three very different CPUs. Let me walk you through what I found."

---

## Slide 2 — The GPU-less Majority (0:30 → 1:45, 75s)

**Visual Elements:**
- Split screen: Left side shows GPU datacenter (H100 image), right side shows diverse consumer devices (laptop, desktop, Raspberry Pi, Jetson Nano)
- Large arrow pointing from right to left labeled "70× range in performance"
- Icon-based representation: 📊 Benchmark leaderboards (left) vs  Real-world devices (right)
- Key stat in bold: "CPU inference isn't an alternative — it's the ONLY path"

**Script:**
"The dominant narrative around LLM inference assumes a GPU — benchmark leaderboards, deployment guides, all of it focuses on datacenter GPUs like the H100. But look at the actual machines people own: aging office laptops, hand-me-down desktops, single-board computers like the Jetson Nano. None have a usable GPU for inference, or the GPU is too small to hold even a modest model.

For this hardware, CPU inference isn't an alternative — it's the only path. And the moment you try running something like llama.cpp on one of these machines, you're hit with a wall of flags: thread count, batch size, context depth, KV-cache precision, Flash Attention on or off, plus a dozen quantized model variants. Without a principled way to navigate that space, people are left with trial and error and forum folklore. That's the gap this thesis closes."

---

## Slide 3 — Why llama.cpp (1:45 → 3:00, 75s)

**Visual Elements:**
- Three columns with icons:
  1. 🔧 CPU Kernel Support (x86/ARM/RISC-V logos)
  2. 🎛️ Parameter Exposure (CLI flags visualization)
  3.  GGUF Ecosystem (Hugging Face logo + quantization levels 2-8 bit)
- Bottom: Timeline showing "Same script → Zen 3 laptop → Pentium → Cortex-A57"

**Script:**
"I chose llama.cpp not because it's the fastest CPU inference engine in every configuration, but because it's the most parametrically exposed one. Three reasons:

First, breadth of CPU kernel support. Its GGML backend ships hand-written SIMD kernels for x86 — that's SSE, AVX, AVX2, AVX-512, VNNI — plus ARM NEON and generic scalar fallbacks. This is precisely what let the exact same benchmark script run unmodified on a modern AMD Zen 3 laptop, a decade-old Intel Pentium, and an ARM Cortex-A57 board.

Second, dense parameter exposure. One tool — llama-bench — exposes threads, context depth, batch size, KV-cache quantization, and Flash Attention as simple command-line flags, emitting clean structured JSONL output. No other CPU inference stack gives you this many independent knobs through one stable interface.

Third, the GGUF ecosystem. Pre-quantized weights exist for essentially every model I'd want, from 2 bits to 8, in both legacy formats like Q4_0 and newer K-quants like Q4_K_M. This let me treat quantization format itself as an experimental variable, not just a fixed implementation detail."

---

## Slide 4 — Dense vs. MoE Models + Contributions (3:00 → 4:30, 90s)

**Visual Elements:**
- Left side: Diagram showing Dense model (all parameters active, full arrows)
- Right side: Diagram showing MoE model (router selecting 2 of 8 experts, sparse arrows)
- Callout box: "14B total params → only 2.7B active per token"
- Bottom: Four contribution icons with labels:
  1.  Empirical characterization
  2.  Roofline-USL model
  3. 💡 Practical recommendations
  4. 🌐 Web calculator

**Script:**
"Two model families matter here. Dense models — like Qwen or Llama — activate every parameter on every single token. Their compute and memory cost per token is fixed by total parameter count.

Mixture-of-Experts models replace that with a bank of experts and a router that activates only a handful per token. So a 14-billion-parameter model might only touch 2.7 billion parameters to generate one word.

On a GPU, where the whole model sits in fast VRAM, that distinction matters less. On a CPU at batch size one — the realistic single-user case — generation is memory-bandwidth-bound, as I'll show you directly. Under that regime, a dense model must stream its entire weight set from RAM for every token, while an MoE model only streams the experts it actually selected. That's the theoretical hook for this thesis.

My contributions are: one, an empirical characterization of hardware effects across real devices; two, a roofline-style model that explains them with calibration constants; three, an extension to the MoE case via a bytes-touched-per-token argument; and four, a public web calculator that implements it."

---

## Slide 5 — Hardware Fleet & Methodology (4:30 → 6:15, 105s)

**Visual Elements:**
- Table with color-coded rows:
  - 🟢 Ryzen 5600H: 6C/12T, 16GB DDR4, ~51 GB/s (mid-range, compute-rich)
  - 🟡 Pentium G5420: 2C/4T, 4GB DDR4, ~38 GB/s (low-end, bandwidth-starved)
  - 🟠 Jetson Nano: 4C/4T, 4GB LPDDR4, ~25 GB/s (embedded, bandwidth-starved)
  - ⚫ Pi 3B: 4C/4T, 1GB LPDDR2, ~3 GB/s (excluded — OOM failure, not shown in table)
- Right side: Resumability diagram showing "Design 1 (failed) → Design 2 (success)"
- Badge: "70× performance range" | "CV < 5%"

**Script:**
"I benchmarked three machines, chosen opportunistically from what I had access to, spanning roughly a seventy-times range in raw generation speed.

First, a mid-range AMD Ryzen 5 5600H laptop with dual-channel DDR4 — compute-rich. Second, a low-end Intel Pentium Gold desktop with a single memory channel and just 4 gigabytes of RAM — bandwidth-starved. Third, an NVIDIA Jetson Nano, an embedded ARM board with shared LPDDR4 memory. I also attempted a fourth device, a Raspberry Pi 3B, but it's excluded from every result I'll show you: its 1 gigabyte of RAM isn't enough to hold even our most aggressively quantized 7B model, so it failed with out-of-memory errors before writing a single row of data — a hardware limitation, not a measurement one.

Everything else ran through a resumable harness I built around llama-bench: three to five repetitions per configuration, coefficient of variation under 5 percent in every case I report.

The resumability was critical. My first design didn't handle interruption well — it re-ran the same measurements over and over when processes were killed. The second design fixed this with row-count-based checkpointing, because these were multi-hour runs on hardware that wasn't dedicated to benchmarking — a laptop that gets closed, a family desktop that gets used for other things. Design 2 is what's behind every number in this talk."

---

## Slide 6 — Finding 1: USL Thread Scaling (6:15 → 8:00, 105s)

**Visual Elements:**
- Left graph: Amdahl's Law curve (rise then flatten, σ=0.05)
- Right graph: USL curve (rise, peak, decline, σ=0.05, κ=0.02)
- Peak marker N* highlighted on USL curve
- Two small inset graphs below:
  - Ryzen: Prompt processing saturates at 6 threads (25 tok/s)
  - Pentium: Generation retrogrades from 0.083 to 0.057 tok/s (32% slower)
- Callout: "κg > κp always → generation more contention-sensitive"

**Script:**
"Here's the first finding, and it surprised me the most. You'd expect adding CPU threads to always help, or at worst plateau. On several machines, it doesn't — performance rises, peaks, and then gets worse. This is retrograde scaling.

Classical Amdahl's Law can't produce that hump — it only predicts rise-then-flatten. Gunther's Universal Scalability Law can, because it adds a term for threads fighting over shared memory and cache lines.

On the Ryzen laptop, prompt processing is compute-bound: it saturates right around 6 threads — exactly the chip's physical core count — even though the chip advertises 12 logical threads via hyperthreading. Going from 6 to 8 threads bought almost nothing.

Generation tells a different story: it peaks at 4 threads at 8.83 tokens per second, then drops to 6.90 at 6 threads before a small partial recovery. Generating each token means re-reading the entire model from RAM, so adding threads to a memory-bound task just adds competitors fighting over the same pipe.

On the cheap Pentium desktop this is brutal: generation falls in a straight line from 0.083 tokens per second at 1 thread down to 0.057 at 8 threads — adding threads made this workload 32 percent slower. There, single-threaded is simply optimal."

---

## Slide 7 — Finding 2: Model Size Determines Regime (8:00 → 9:45, 105s)

**Visual Elements:**
- Two-panel comparison:
  - Left: 7B model on Jetson (flat generation line: 0.041→0.039 tok/s)
  - Right: 1.5B model on Jetson (steep generation line: 0.94→3.03 tok/s, 3× speedup)
- Both show prompt processing scaling (upward slopes)
- Memory bandwidth pipe visualization:
  - 7B model: Pipe completely full at 1 thread (25.6 GB/s saturated)
  - 1.5B model: Pipe partially full, room for more threads
- Key insight box: "Bytes per token matters, not thread count"

**Script:**
"The clearest illustration of 'it depends on model size' comes from the Jetson Nano, where I ran the identical sweep on two very differently sized models.

Prompt processing speeds up with more threads for both the 7-billion-parameter model and the much smaller 1.5-billion-parameter model. That makes sense — it's compute-bound work you can split across cores regardless of model size.

Generation is where they diverge completely. The small model's generation speed scales up almost linearly with threads — from 0.94 to 1.86 to 3.03 tokens per second, more than 3 times faster going from 1 to 4 threads. The large model barely moves at all — from 0.041 to 0.038 to 0.039 tokens per second. Adding threads does essentially nothing.

The mechanism is the same memory-bandwidth story, just made vivid: generating one token means reading the whole model's weights out of RAM once. The 7B model is big enough that even a single thread already uses up all of the Jetson's roughly 25.6 gigabytes per second of memory bandwidth. There's nothing left for more threads to use. The 1.5B model is small enough that one thread doesn't saturate that same memory pipe, so extra threads still help.

This is the single clearest piece of evidence in my whole dataset that for generation speed, what matters is how many bytes of the model have to move through memory per word — not how many threads you throw at it. And that idea is exactly what motivates the Mixture-of-Experts argument later."

---

## Slide 8 — Finding 3: Flash Attention Regression (9:45 → 11:15, 90s)

**Visual Elements:**
- Three-row comparison table (depth 0, 3584, 15872):
  - Each row shows fa=0 vs fa=1 bars side by side
  - Color gradient: green (neutral) → yellow (18% slower) → red (68% slower)
- GPU vs CPU architecture diagram:
  - GPU: HBM ←→ SRAM (two-tier, Flash Attention helps)
  - CPU: L1/L2/L3 cache (unified, data already fits)
- Callout: "3× slowdown at 15K context"
- Warning icon: "Turn Flash Attention OFF on CPU"

**Script:**
"Flash Attention is a well-known GPU optimization that restructures attention math to avoid slow trips to GPU memory. It was never designed for CPUs, which manage their cache hierarchy completely differently.

My data shows it doesn't just fail to help on CPU — it actively hurts, and badly at long context.

At a short prompt, Flash Attention makes basically no difference — 7.81 versus 7.83 tokens per second for prompt processing. At a medium prompt of about 3,500 tokens, generation is already 18 percent slower — down from 3.31 to 2.73 tokens per second.

At a long prompt of about 15,800 tokens — roughly a long article — turning it on drops generation from 2.09 down to 0.67 tokens per second. That's about a third of normal speed — a 3 times slowdown.

My explanation: Flash Attention's benefit comes from avoiding slow trips to memory by doing more work per trip. On this CPU, the relevant attention data already fits comfortably inside the chip's large 16-megabyte L3 cache, so there was little slow-memory traffic to save in the first place. The restructuring itself carries a real computational cost — fewer independent operations the CPU can run in parallel internally.

On GPU that trade is a clear win. On this CPU it's a loss that gets worse the longer the prompt is.

What this means in practice: if you're running llama.cpp on a CPU with long prompts or long conversations, turn Flash Attention off. Many front-ends enable it by default, which is actively counterproductive here."

---

## Slide 9 — Finding 4: KV-Cache Quantization (11:15 → 12:45, 90s)

**Visual Elements:**
- Left: KV cache growth diagram (conversation turns → cache size)
- Right: Performance comparison at three depths:
  - Depth 0: f16/f16 (7.81) vs q8_0/q8_0 (7.12) → 9% slower
  - Depth 3584: f16/f16 (6.83) vs q8_0/iq4_nl (1.69) → 75% slower (4×)
  - Depth 15872: f16/f16 (4.59) vs q8_0/q8_0 (1.17) → 75% slower (4×)
- Dequantization overhead icon: "Extra arithmetic per access"
- Decision flowchart: "Compute-bound? → Don't quantize KV" vs "Bandwidth-bound? → Maybe quantize"

**Script:**
"Next, the KV cache — the model's running memory of the conversation so far. It grows as conversation length grows, and quantizing it is a common trick to save RAM. The intuition is that a smaller cache should also be faster.

My data says otherwise, at least here.

On the Ryzen laptop at 1 thread, every quantized-KV configuration I tested was slower than the unquantized f16 baseline — both for processing the prompt and for generating text. At the shortest context length, prompt processing fell from 7.81 down to as low as 6.37 tokens per second, and generation from 3.95 down to 3.71.

The picture gets worse, not better, as the conversation grows. At medium context length, one format was 4 times slower for prompt processing. At the longest context length we tested, another was also 4 times slower.

So why would a smaller cache ever be slower? Because reading a quantized value isn't free — the CPU must do extra arithmetic to convert it back to a usable number, a step called dequantization. On a single thread with compute cycles to spare, that arithmetic costs more time than the smaller memory footprint saves.

This configuration is compute-bound, not memory-bound, so a trick built to save memory traffic is pure overhead — it's solving a problem this configuration doesn't have.

This points to a general rule: whether KV-cache quantization helps or hurts depends on whether a given device, at a given conversation length, is limited by compute or by memory bandwidth. On the compute-rich x86 laptop tested here, it was a net loss at every conversation length. I'd predict the opposite result — a net win — on a more memory-starved device like the Jetson or Pentium at long conversation lengths, but I haven't tested that combination yet."

---

## Slide 10 — Finding 5: Quantization Format ≠ Bit-Width (12:45 → 14:15, 90s)

**Visual Elements:**
- Bar chart comparing three formats at 1 thread:
  - Q4_0: 1.78 tok/s (tallest bar, green)
  - Q4_K_M: 0.49 tok/s (medium bar, yellow)
  - Q3_K_M: 0.42 tok/s (shortest bar, red)
- Callout: "3.6× faster despite same bit-width!"
- Layout comparison diagram:
  - Q4_0: Simple, evenly-spaced blocks → Fast unpack
  - K-quants: Elaborate super-blocks → Complex unpack
- CPU instruction pipeline: Q4_0 fits neatly, K-quants cause stalls

**Script:**
"It's tempting to think of quantization purely in terms of bits per weight — fewer bits means smaller, faster. My data shows that's not the whole story. The specific layout the bits are packed into matters just as much as how many of them there are.

On the Pentium desktop, I benchmarked three quantizations of the same model at 1 thread. Q4_K_M processed the prompt at 0.49 tokens per second. Q4_0 — an older, simpler format at the exact same 4-bit width — hit 1.78 tokens per second. That's 3.6 times faster, same bit-width.

And Q3_K_M, which uses fewer bits than either — just 3 bits — was actually the slowest of the three at 0.42 tokens per second. The ordering doesn't follow bit-width at all.

My best explanation: Q4_0 uses a simple, evenly-spaced storage layout that maps neatly onto a fast, straightforward CPU instruction sequence. The newer K-quant formats pack bits more cleverly to squeeze out better quality per bit — and they do achieve better perplexity — but that requires a more complicated unpacking step every time a weight is read.

On a CPU that lacks certain modern vector instructions like AVX-512 or VNNI, that unpacking cost can outweigh the benefit of the smaller file. We present this as an observed, hardware-dependent effect rather than a fully settled explanation — confirming exactly which internal code path each format used would require instrumenting llama.cpp itself, which this study didn't do."

---

## Slide 11 — The MoE Hypothesis (14:15 → 15:30, 75s)

**Visual Elements:**
- Left: Bytes-per-token formula for Dense vs MoE:
  - Dense: B = N_params × bits/8
  - MoE: B = (N_active + N_shared) × bits/8
- Right: External validation box:
  - DeepSeek-R1/V3: 671B total, 37B active
  - 4.51 tok/s on 64-core Xeon
  - Badge: "80× larger than our staged models"
- Bottom: "Hypothesis, not result" warning label
- Arrow: "Next critical experiment"

**Script:**
"This brings me to the central hypothesis of the thesis. The bytes-per-token argument — that MoE models should be disproportionately advantaged on bandwidth-starved CPUs because they stream a fraction of their weights per token — is a hypothesis derived directly from the dense-model bandwidth mechanics I measured.

I did find one external, verifiable data point: KTransformers published a CPU-only benchmark of a 671-billion-parameter MoE model achieving 4.51 tokens per second on a 64-core server. That's directionally consistent with the hypothesis — a model that large sustaining any usable generation speed on CPU is only possible because a small fraction of its parameters are active per token.

But it's a single data point on a model roughly 80 times larger by active-parameter count than anything I staged, on server hardware nothing like my fleet. I was not able to independently verify other, more specifically on-point community claims, and I've excluded them rather than report a number I couldn't source.

Directly benchmarking MoE models against dense equivalents on constrained hardware is the single most important next experiment for this line of work."

---

## Slide 12 — Roofline-USL Model + Worked Example (15:30 → 17:15, 105s)

**Visual Elements:**
- Left: Roofline diagram with two curves:
  - USL compute curve (peaks then declines)
  - Horizontal bandwidth ceiling line
  - Generation throughput = min(compute, bandwidth)
- Right: Calibration workflow (5 steps):
  1. Measure P₁=7.81, G₁=3.95
  2. Fit σ, κ coefficients
  3. Back out BW_eff = 41.3 GB/s
  4. Predict at N=6: T_pp=26.2, T_tg=8.12
  5. Compare: Measured 25.15, 6.90 → 4%, 18% error
- Badge: "5 calibration points → full prediction"

**Script:**
"So how do these effects fit together? I formalized them into one model.

Prompt processing is USL-scaled single-thread throughput — the same curve as the thread-scaling slide. Generation is a roofline: the lesser of that compute curve or a hard bandwidth ceiling, because whichever constraint binds first is what you observe.

For a dense model, bytes streamed per token is essentially the whole quantized model size. For MoE, it's just the selected experts plus shared layers. That's the formal version of the earlier argument: for two models of similar quality, the MoE model's throughput ceiling should be higher by roughly the ratio of total to active parameters, whenever generation is bandwidth-bound — which my data shows is the common case on constrained CPUs.

Let me walk through a concrete calibration for the Ryzen running a 7-billion-parameter model. Step 1: measure single-thread baselines — 7.81 tokens per second for prefill, 3.95 for generation. Step 2: fit the USL contention coefficients from the thread sweep. Step 3: back out an effective bandwidth of about 41.3 gigabytes per second — around 80 percent of theoretical peak, consistent with sustained bandwidth.

Step 4: predict throughput at 6 threads. The model predicts 26.2 tokens per second for prefill and 8.12 for generation. Measured values were 25.15 and 6.90 — within 4 percent and 18 percent, from just five calibration points.

Step 5: given the calibrated model, a user can now evaluate any configuration and read off predicted throughput, then compare against quality and memory to find the Pareto-optimal configuration for their needs."

---

## Slide 13 — Practical Recommendations (17:15 → 18:30, 75s)

**Visual Elements:**
- Seven recommendation cards with icons:
  1. 🚫 Flash Attention at long context → Use -fa 0
  2. 💾 Don't quantize KV on compute-rich → Use -ctk f16 -ctv f16
  3. ⚙️ Physical cores, not logical → Use -t = physical cores
  4.  Single-thread on bandwidth-starved → Use -t 1
  5. 🔧 Legacy formats on old CPUs → Q4_0 faster than Q4_K_M
  6.  MoE on bandwidth-starved (provisional) → Test on your hardware
  7. 📦 Match model size to device memory budget → Avoid OOM/thrashing
- Color coding: Red (avoid), Green (preferred), Yellow (test first)

**Script:**
"So, concretely, what should you do running llama.cpp on CPU? Seven recommendations:

One: turn Flash Attention off at long context. Many front-ends enable it by default, which is actively counterproductive here.

Two: don't quantize the KV cache on compute-rich CPUs. Use full-precision f16 unless you're on genuinely bandwidth-starved hardware.

Three: set thread count to the physical core count, not the logical count. Hyperthreads buy nothing for prefill and hurt generation.

Four: on bandwidth-starved chips like the Pentium, single-threaded generation may be optimal. Check before assuming more threads help.

Five: on legacy CPUs without AVX-512, prefer simpler formats like Q4_0 for speed. Use K-quants only if quality matters more.

Six — and I'll flag this as provisional: prefer MoE models on bandwidth-starved hardware, since the roofline model predicts they'll be disproportionately advantaged there. That one is a hypothesis to test on your own hardware, not a settled recommendation.

And seven: match model size to the device's memory budget. A model that doesn't fit in RAM alongside its KV cache will thrash or fail outright — exactly what happened with the Raspberry Pi 3B I mentioned earlier."

---

## Slide 14 — Limitations (18:30 → 19:30, 60s)

**Visual Elements:**
- Three limitation boxes with severity indicators:
  1.  Calibrated, not derived (3 devices → per-device constants)
  2.  MoE untested (hypothesis, not confirmed)
  3. 🟡 llama.cpp-specific (one engine only)
- Right side: Data coverage table (Table 5.1 from report)
- Bottom: "Honest scope bounds what we can claim"

**Script:**
"I want to be honest about scope, because that matters as much as the results. Three limitations:

First, the model is calibrated, not derived. I fit constants separately for three devices from a handful of local measurements, without regressing them against raw hardware specs in a way that predicts a new device without measuring it first. This thesis delivers a cheaper way to calibrate — roughly six measurements instead of a full grid search — not a way to skip calibration entirely.

Second, the MoE prediction, the most consequential claim here, was never confirmed on my own hardware.

Third, every finding comes from one inference engine, llama.cpp. I can't tell you whether the quantization-format anomaly reflects the hardware and the underlying math, or an artifact of how this engine dispatches its kernels.

None of this discounts the findings — every effect I've shown is a real, reproducible measurement — but it bounds what I can honestly claim."

---

## Slide 15 — Future Work (19:30 → 20:15, 45s)

**Visual Elements:**
- Two columns:
  - Near-term (next 6 months):
    - Expand hardware fleet (Pi 4/5, AVX-512, Apple Silicon)
    - Run MoE benchmarks
    - Complete perplexity sweeps
    - Test CPU pinning/NUMA
    - Validate sign-flips on bandwidth-starved
  - Longer-term (1-2 years):
    - Zero-shot predictor (20-30 devices)
    - Community calibration dataset
    - Concurrent serving model
    - Generalize bytes-per-token lens
    - Trajectory modeling
    - Performance-energy rooflines
    - Cross-engine validation

**Script:**
"Near-term, five things: expand the hardware fleet — I especially want an AVX-512 chip, an Apple Silicon Mac, and a Raspberry Pi 4 or 5. Run MoE benchmarks on the fleet. Investigate CPU pinning and NUMA effects. And test Flash Attention and KV-cache quantization specifically on the bandwidth-starved machines, to check whether the predicted sign-flip actually happens there.

Longer-term, the most ambitious direction is turning per-device calibration into a true zero-shot predictor — one that takes a CPU's raw specs and outputs its constants without ever touching the hardware. That would need on the order of twenty to thirty calibrated devices, ideally contributed by a community rather than one person's personal fleet."

---

## Slide 16 — Conclusion (20:15 → 20:45, 30s)

**Visual Elements:**
- Four key findings as icons with checkmarks:
  1. ✅ Compute-bound prefill (USL)
  2. ✅ Memory-bound generation (retrograde scaling)
  3. ✅ Quantization choices (device-dependent)
  4. ✅ Flash Attention regression (long context)
- Central model badge: "Roofline-USL (4-18% error)"
- MoE hypothesis badge: "Next critical experiment"
- Thank you + Q&A prompt

**Script:**
"To close: three consumer and embedded CPUs were enough to recover four concrete, reproducible effects governing CPU-only LLM inference — compute-bound prefill that follows a scalability law, memory-bound generation that can get slower as you add threads, quantization choices whose speed impact flips depending on the device, and a Flash Attention regression at long context.

I formalized these into a roofline model that predicts measured Ryzen throughput within 4 to 18 percent, and converted it into concrete, device-aware advice.

The model's most interesting prediction — that MoE models should be structurally favored on exactly this kind of hardware — remains the clearest next step.

Thank you. I'm happy to take questions."

---

## Delivery Notes

**Total: ~2,600 words → 19-20 minutes at 2.2 w/s**

**Visual emphasis:**
- Use color coding consistently (green=good, red=bad, yellow=caution)
- Include actual data points on all graphs
- Use icons and diagrams to break up text
- Highlight key numbers in callout boxes
- Show before/after comparisons side-by-side

**Pacing tips:**
- Numbers take longer to read — practice slides 6, 7, 8, 12
- Built-in pause points: after slides 4, 10, 13
- If running long, compress slides 7 and 15
- If interrupted, slides 7 and 15 are safest to skip

**Q&A preparation:**
- Know why USL fits better than Amdahl
- Be ready to explain dequantization overhead
- Have the MoE hypothesis defense ready
- Know the difference between Q4_0 and Q4_K_M layouts