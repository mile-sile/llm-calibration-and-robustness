# Probing LLMs: Empirical Calibration, Generative Diversity, and the Alignment Tax

An empirical evaluation framework investigating **epistemic uncertainty**, **confidence calibration**, and **mode collapse** across Large Language Models. This project treats LLMs not merely as prompt-engineering tools, but as **objects of empirical study**—measuring how reinforcement learning from human feedback (RLHF) and temperature sampling distort model confidence, factual accuracy, and semantic diversity.

---

## Key Research Questions

1. **Epistemic Calibration:** *When an LLM expresses high confidence, does its empirical accuracy reflect that belief?*
2. **The Hard-Easy Effect:** *How does model calibration error deviate across difficulty strata (Easy, Medium, Hard, and Unanswerable/Trick questions) compared to human baseline cognition?*
3. **The Alignment Tax & Creativity:** *How does RLHF safety tuning constrain semantic diversity and induce mode collapse across varying temperature parameters ($T \in [0.0, 1.5]$)?*
4. **Adversarial Failure Modes:** *How do models handle semantic unanswerability versus confident confabulation/hallucination?*

---

## Core Findings & Experiments

### 1. Confidence Calibration & Reliability Analysis
Models were benchmarked against a 50-question ground-truth dataset spanning four difficulty categories (Easy, Medium, Hard, and Unanswerable). Confidence scores were binned into reliability diagrams and benchmarked against human baseline performance ($N=20$ paired samples).

| Category | Model Accuracy | Model Mean Confidence | Calibration Error ($|\text{Conf} - \text{Acc}|$) | Human Calibration Error |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** | 1.000 | 0.995 | **0.005** | 0.000 |
| **Medium** | 0.733 | 0.927 | **0.240** | 0.040 |
| **Hard** | 0.867 | 0.927 | **0.180** | 0.550 |
| **Unanswerable / Trick** | 0.650 | 0.850 | **0.200** | 0.190 |
| **Overall Mean Error** | — | — | **0.156** | **0.197** |

```
               [Calibration Curve: Human vs. LLM]
    1.0 ┼                                 ● (LLM: Overconfident)
        │                              ▲  
    0.8 ┼                           ▲ / (Perfect Calibration y=x)
Acc     │                        ▲   / 
    0.6 ┼                     ▲     / 
        │                  ▲       /
    0.4 ┼               ▲         /
        │            ▲           /
    0.2 ┼         ▲             /
        └─────────┴─────────────┴─────────────┴─────────
       0.4       0.6           0.8           1.0
                         Confidence Bucket
```

* **Uniform Overconfidence Bias:** While human evaluators exhibited the classic *hard-easy effect* (Lichtenstein et al., 1982)—overconfident on hard questions and well-calibrated on easy tasks—the LLM maintained **uniformly high confidence ($\ge 0.85$) across all categories**, driven by RLHF reward models penalizing hedged phrasing.
* **Unanswerability Failure Mode:** When presented with trick questions without an objective ground truth, the model frequently confabulated plausible justifications at $85\%+$ confidence rather than explicitly admitting epistemic limits.

---

### 2. Temperature Dynamics & Mode Collapse
Evaluated divergent thinking and generative originality across temperature increments:
* **Low Temperatures ($T \le 0.3$):** Showed severe **mode collapse**, converging onto repetitive, stereotypical phrasing and predictable semantic clusters.
* **Optimal Divergence ($T \in [0.7, 1.0]$):** Maintained grammatical coherence while maximizing token entropy and non-obvious associative leaps.
* **The Alignment Tax:** Safety-fine-tuned models exhibited an *over-refusal frontier*, rejecting benign ambiguous prompts under low-entropy sampling due to defensive alignment constraints.

---

## Methodology & Mathematical Formulations

* **Calibration Error ($CE_i$):**
  $$CE_i = |\text{Confidence}_i - \text{Accuracy}_i|$$
* **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  where $B_m$ represents the set of queries binned into confidence interval $m$.
* **Confidence Floor Normalization:** Confidence scores below $0.50$ are floored to $0.50$ baseline to represent pure guessing under binary decision boundaries.

---

## Project Structure

```text
├── data/
│   ├── calibration_dataset.csv       # 50 curated ground-truth questions
│   └── evaluation_results.csv        # Evaluated model outputs & fuzzy-match scores
├── notebooks/
│   ├── calibration_experiment.ipynb  # End-to-end reliability diagrams & t-tests
│   └── creativity_rubric_eval.ipynb  # Temperature scaling & divergence benchmarks
├── src/
│   ├── query_engine.py               # OpenAI & Gemini API batch inference runner
│   ├── metrics.py                    # ECE and calibration curve computations
│   └── scoring.py                    # Regex parsers and fuzzy-matching logic
└── requirements.txt
```

---

## Quick Start

### 1. Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/llm-calibration-and-robustness.git
cd llm-calibration-and-robustness
pip install -r requirements.txt
```

### 2. Run the Calibration Pipeline
```bash
python -m src.query_engine --model gpt-3.5-turbo --dataset data/calibration_dataset.csv
python -m src.metrics --input data/evaluation_results.csv --plot
```

---

## Tech Stack & Dependencies

* **Language:** Python 3.10+
* **LLM APIs:** OpenAI API (`gpt-3.5-turbo`), Google GenAI (`gemini-2.5-flash`)
* **Data Science & Visualization:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`
