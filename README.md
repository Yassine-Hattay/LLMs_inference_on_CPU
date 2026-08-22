# LLMs Inference on CPU

Code, data, and thesis report for **"Toward Hardware-Adaptive Pareto Fronts for CPU-Only LLM Inference: An Empirical Study of `llama.cpp` Across Four Heterogeneous Devices"** (Yassine, Master's Thesis, 2026).

- 📄 **Report:** [`thesis.pdf`](./thesis.pdf)
- 🧮 **Interactive calculator:** https://yassine-hattay.github.io/LLMs_inference_on_CPU/index.html — implements the roofline model from the thesis so you can explore predicted throughput for your own hardware.
- 💻 **Code + data:** [`Code_and_data/`](./Code_and_data) — the benchmarking scripts and raw results for each of the three machines. See [`Code_and_data/README.md`](./Code_and_data/README.md) for how to run them and a full breakdown of the data format.

## Repository structure

```
LLMs_inference_on_CPU/
├── thesis.pdf                              the full report
├── LICENSE                                 MIT license (code)
└── Code_and_data/
    ├── README.md                           how to run the scripts + data format reference
    ├── LICENSE-DATA                        CC-BY-4.0 license (data + report)
    ├── Desktop (Pentium G5420).sh          benchmark script — Intel Pentium Gold G5420
    ├── Desktop (Pentium G5420) data.txt    raw results collected by that script
    ├── Laptop (Ryzen 5600H).sh             benchmark script — AMD Ryzen 5 5600H
    ├── Laptop (Ryzen 5600H) data.txt       raw results collected by that script
    ├── Jetson Nano.sh                      benchmark script — Jetson Nano (ARM Cortex-A57)
    └── Jetson Nano data.txt                raw results collected by that script
```

## License

The benchmarking scripts (`.sh` files) are MIT-licensed — see [`LICENSE`](./LICENSE). The benchmark data (`.txt` files) and the thesis report are CC-BY-4.0-licensed — see [`Code_and_data/LICENSE-DATA`](./Code_and_data/LICENSE-DATA).

## Citation

```
Yassine. "Toward Hardware-Adaptive Pareto Fronts for CPU-Only LLM Inference:
An Empirical Study of llama.cpp Across Four Heterogeneous Devices."
Master's Thesis, 2026.
```
