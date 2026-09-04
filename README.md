# LLMs Inference on CPU

Code and data for **"Optimizing CPU-Only LLM Inference: Parameter Sweeps, Hardware Bottlenecks, and Performance Modeling"** (Yassine Hattay, Master's Thesis, 2026).

- 🧮 **Interactive calculator:** https://yassine-hattay.github.io/LLMs_inference_on_CPU/index.html — implements the roofline model from the thesis so you can explore predicted throughput for your own hardware.
- 💻 **Code + data:** [`Code_and_data/`](./Code_and_data/) — the benchmarking script and raw results for each of the three machines. See [`Code_and_data/README.md`](./Code_and_data/README.md) for how to run them and a full breakdown of the data format.

## Repository structure

```text
LLMs_inference_on_CPU/
├── README.md
├── LICENSE                                 # MIT license (code)
└── Code_and_data/
    ├── README.md                           # How to run the scripts + data format reference
    ├── LICENSE-DATA                        # CC-BY-4.0 license (data)
    ├── data_collection.sh                  # Data collection script
    ├── Desktop (Pentium G5420) data.txt    # Raw results collected by that script
    ├── Laptop (Ryzen 5600H) data.txt       # Raw results collected by that script
    └── Jetson Nano data.txt                # Raw results collected by that script