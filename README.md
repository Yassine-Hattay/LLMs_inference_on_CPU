# LLMs Inference on CPU

Code, data, presentation, and thesis report for **"Optimizing CPU-Only LLM Inference: Parameter Sweeps,
Hardware Bottlenecks, and Performance Modeling"** (Yassine Hattay, Master's Thesis, 2026).

- 📄 **Report:** [`Presentation_and_report/thesis.pdf`](./Presentation_and_report/thesis.pdf)
- 📊 **Presentation:** [`Presentation_and_report/CPU_LLM_Inference_Defense.pptx`](./Presentation_and_report/CPU_LLM_Inference_Defense.pptx) and [`Presentation_and_report/Presentation_Script.md`](./Presentation_and_report/Presentation_Script.md)
- 🧮 **Interactive calculator:** https://yassine-hattay.github.io/LLMs_inference_on_CPU/index.html — implements the roofline model from the thesis so you can explore predicted throughput for your own hardware.
- 💻 **Code + data:** [`Code_and_data/`](./Code_and_data/) — the benchmarking script and raw results for each of the three machines. See [`Code_and_data/README.md`](./Code_and_data/README.md) for how to run them and a full breakdown of the data format.

## Repository structure

```text
LLMs_inference_on_CPU/
├── README.md
├── LICENSE                                 # MIT license (code)
├── Presentation_and_report/
│   ├── thesis.pdf                          # The full Master's thesis report
│   ├── CPU_LLM_Inference_Defense.pptx      # Presentation slides
│   └── Presentation_Script.md              # Speaker script for the defense
└── Code_and_data/
    ├── README.md                           # How to run the scripts + data format reference
    ├── LICENSE-DATA                        # CC-BY-4.0 license (data + report)
    ├── data_collection.sh                  # Data Collection script
    ├── Desktop (Pentium G5420) data.txt    # Raw results collected by that script
    ├── Laptop (Ryzen 5600H) data.txt       # Raw results collected by that script
    └── Jetson Nano data.txt                # Raw results collected by that script

```

## License

The benchmarking scripts (`.sh` files) are MIT-licensed — see [`LICENSE`](./LICENSE). The benchmark data (`.txt` files) and the thesis report are CC-BY-4.0-licensed — see [`Code_and_data/LICENSE-DATA`](./Code_and_data/LICENSE-DATA).

## Citation

```
Yassine. "Toward Hardware-Adaptive Pareto Fronts for CPU-Only LLM Inference:
An Empirical Study of llama.cpp Across Four Heterogeneous Devices."
Master's Thesis, 2026.
```
