# R-Latplan: Learning Reliable PDDL Models for Classical Planning from Visual Data

> A Neuro-Symbolic Learning Framework for Visually Grounded and Executable Planning Models

![ICTAI 2024](https://img.shields.io/badge/IEEE-ICTAI_2024-blue)
![Neuro-Symbolic AI](https://img.shields.io/badge/AI-Neuro--Symbolic-brightgreen)
![License](https://img.shields.io/github/license/aymeric75/R-latplan)

## 📜 Abstract

**R-Latplan** is a novel framework that learns **reliable PDDL (Planning Domain Definition Language)** action models directly from **noisy image observations** without any expert supervision or manually annotated symbolic states. It extends the original **Latplan** system by introducing a transition identifier function that anchors learned transitions to **real-world agent actions**, ensuring **executability, interpretability, and robustness**.

This repository contains the official implementation and experimental benchmarks presented in our ICTAI 2024 paper:  
**"Learning Reliable PDDL Models for Classical Planning from Visual Data"**  
by Aymeric Barbin, Federico Cerutti, Alfonso Gerevini

[[Read the Paper (PDF)]](./Learning_Reliable_PDDL_Models_for_Classical_Planning_from_Visual_Data.pdf)

---

## 🚀 Highlights

- 🔗 **Reliable Symbolic Learning**: Learns executable action models directly linked to high-level agent capabilities.
- 🧠 **Neuro-Symbolic Backbone**: Integrates Variational AutoEncoders with symbolic reasoning.
- 🖼️ **Visual Data Only**: Trained from image traces without human-labelled annotations.
- 🧪 **Robustness to Noise**: Handles mislabeled transitions and noisy observations.
- 🧩 **Domain-Independent**: Validated on classic planning benchmarks (Hanoi, Blocksworld, Sokoban).

---

## 📂 Project Structure
```bash
R-latplan/
├── models/                # Core neural networks (SAE, APPLY, REGRESS)
├── data/                  # Scripts for generating visual traces (from PDDLGym)
├── planner/               # PDDL domain generation + Fast Downward integration
├── evaluation/            # Benchmarking experiments (RQ1–RQ3)
├── utils/                 # Helper functions (image processing, DFA construction)
├── configs/               # Hyperparameter settings for each experiment
├── examples/              # Trained models and generated PDDL domains
├── README.md              # This file
└── paper.pdf              # The published paper
````

## 🧪 Experiments & Results

We designed five experiments across three benchmark domains—**Hanoi**, **Blocksworld**, and **Sokoban**—to evaluate R-Latplan's reliability and robustness:

| Research Question | Objective |
|-------------------|-----------|
| **RQ1** | Does R-Latplan produce visually and semantically reliable PDDL actions even under noise? |
| **RQ2** | Can classical planners (e.g., Fast Downward) find optimal plans using learned models from noisy/incomplete traces? |
| **RQ3** | Is R-Latplan robust to errors in the transition identifier function? |

**Highlights from the results:**

- ✅ R-Latplan always links PDDL actions to real agent capabilities.
- ✅ It avoids hallucinations common in Latplan.
- ✅ Fast Downward solved all benchmark tasks with optimal plans.
- ✅ Robust against image noise and transition mislabeling.

📊 **Redundant Actions Correction (Exp 5)**

| Domain      | Redundant Groups Found | Ground Truth |
|-------------|------------------------|--------------|
| Hanoi       | 72                     | 72           |
| Blocksworld | 14                     | 14           |
| Sokoban     | 207                    | 207          |



---


## ▶️ Running the Experiments

### Step 1: Generate Visual Traces
```bash
python data/generate_traces.py --domain blocksworld
````
### Step 2: Train R-Latplan
```bash
python train.py --domain blocksworld --config configs/blocksworld.json
````
### Step 3: Generate PDDL Domain
```bash
python planner/generate_pddl.py --model outputs/blocksworld_model.pt
````
### Step 4: Solve with Fast Downward
```bash
python planner/run_planner.py --domain outputs/blocksworld.pddl --problem problems/blocksworld/p01.pddl
````


---

## 📜 Citation

If you use this code or build upon this work, please cite the following paper:

```bibtex
@inproceedings{barbin2024rlatplan,
  title={Learning Reliable PDDL Models for Classical Planning from Visual Data},
  author={Barbin, Aymeric and Cerutti, Federico and Gerevini, Alfonso Emilio},
  booktitle={36th IEEE International Conference on Tools with Artificial Intelligence (ICTAI)},
  year={2024}
}
```
---

## 👤 Authors

- **Aymeric Barbin**  
  Doctorate in Artificial Intelligence – Sapienza University of Rome  
  [aymeric.barbin@uniroma1.it](mailto:aymeric.barbin@uniroma1.it)  
  [GitHub @aymeric75](https://github.com/aymeric75)

- **Federico Cerutti**  
  University of Brescia, Italy

- **Alfonso E. Gerevini**  
  University of Brescia, Italy

