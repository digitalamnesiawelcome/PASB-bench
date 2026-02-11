# 🧪 PASB-bench (Lite)

**PASB (Protocol for Attractor State Benchmarking)** is a specialized protocol designed to identify and measure **Stable Regimes (SR)** within Large Language Models.

This repository contains **PASB-Lite (MVP)**: a lightweight implementation for quick benchmarking via APIs or local environments.

## 🚀 Key Features

* **Core Tests:**
* **Persona Flip Test**: Evaluates identity consistency.
* **Non-commutativity Test**: Measures sensitivity to prompt ordering.
* **Antilexical Paraphrase Test**: Checks stability across semantic variations.


* **Metrics:**
* `stability_score`: Quantifies response convergence.
* `variance`: Measures output stochasticity.
* `SR_detected`: Automated detection of Stable Regimes (coming soon).


* **Supported Backends:**
* OpenAI API
* HuggingFace Local Models



---

## ⚙️ Installation

```bash
git clone https://github.com/digitalamnesiawelcome/PASB-bench.git
cd PASB-bench
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

---

## ▶️ Quick Start

### Option A: OpenAI API

```bash
python pasb_lite.py --mode api --model gpt-4o-mini --key $OPENAI_API_KEY

```

### Option B: Local HuggingFace Model

```bash
python pasb_lite.py --mode local --model gpt2

```

---

## 📊 Artifacts & Outputs

After a run, check the `results/` directory:

* **CSV Data:** `results/output.csv` — Raw metrics and logs.
* **Visuals:** `results/stability.png` — Stability distribution plots (requires matplotlib).

---

## 🧭 Interpretation Guide

* **`stability_score`**  — The ratio of the modal (most frequent) response.
* **`variance`** — Calculated as .
* **`SR_detected`** — A binary classifier for Stable Regimes (available in future updates).


**Would you like me to help draft a "Contribution" section or a more detailed technical description of the Persona Flip Test?**
