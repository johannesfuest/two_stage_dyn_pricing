# 📈 TSMT Dynamic Pricing in Credit Markets

This project implements a **Multi-Task Stochastic Multi-Armed Bandit (TSMT)** algorithm for **contextual dynamic pricing** in credit markets. The goal is to learn optimal bond pricing strategies using both synthetic and real-world datasets from TRACE and LSEG.

---

## Project Highlights

- Models yield optimization with contextual features (e.g., PCA components)
- Implements 3 learning strategies:
  - **Pooled**: A single shared model across all bonds
  - **Individual**: Separate model per bond (task)
  - **Multi-task**: Task-specific models with shared initialization and regularization
- Includes both synthetic simulations and real bond market data
- Tracks regret compared to oracle pricing (Ridge-based benchmark)

---

## Script Overview

| Script                   | Description |
|--------------------------|-------------|
| `data.py`                | Runs the **complete preprocessing pipeline** (TRACE → LSEG → final merge). |
| `TRACE_data.py`          | Processes raw TRACE transaction data. |
| `LSEG_data.py`           | Loads and formats LSEG fundamentals and maps to TRACE CUSIPs. |
| `preprocessing_data.py`  | Merges TRACE + LSEG data, filters, and formats for modeling. |
| `syn_data_gen.py`        | Generates synthetic bond datasets with contextual features. |
| `tsmt_algo.py`           | Runs the TSMT algorithm on **synthetic data**. |
| `tsmt_algo_real.py`      | Runs the TSMT algorithm on **real bond data**, evaluating all strategies. |
