# Jev Calibration Report

Independent evaluation of Jev's calibrated probabilities.

## Methodology
- 190+ labeled examples from [real data from email]
- Each example: one Noul question with known ground truth
- Jev returned probability (0-1) for each
- Compared probability bins against actual accuracy
- evaluated Jev's three question heads — **Noul** (binary probability),
**Choice** (categorical), and **Score** (ordinal) — on 192 labeled email
examples.

### Noul head (binary probability)

| Method | ECE ↓ | Brier ↓ | Accuracy |
|---|---|---|---|
| Uncalibrated | 0.1055 | 0.0582 | 0.9062 |
| Temperature scaling (T = 0.818) | 0.0930 | 0.0596 | 0.9062 |
| Isotonic regression | **0.0573** | **0.0495** | **0.9583** |

### Choice head (categorical, top-class confidence)

| Metric | Value |
|---|---|
| Multiclass ECE | 0.0490 |
| Multiclass Brier | 0.1205 |
| Accuracy | 0.9219 |
| Mean top-class confidence | 0.9709 |

### Score head (ordinal)

| Metric | Value |
|---|---|
| Mean score value (0–4) | 3.59 |
| Mean confidence | 0.839 |
| Mean top-class probability | 0.905 |
| Normalized entropy (0=peak, 1=uniform) | 0.162 |
| Monotonicity fraction | **1.000** |
| Spearman ρ vs Noul | **0.871** |

## Findings

1. **The heads are miscalibrated in opposite directions.** Noul is
   **under-confident** (mean probability 0.870 vs positive rate 0.948,
   gap −0.078) while Choice is **over-confident** (mean top-class confidence
   0.971 vs accuracy 0.922, gap +0.049). A single global calibration strategy
   cannot fix both.

2. **Temperature scaling barely helps on Noul** (ECE 0.1055 → 0.0930, Δ
   −0.013), because the miscalibration is bimodal — under-confident at low
   probabilities, over-confident at high ones. **Isotonic regression** fits a
   non-parametric monotone map and cuts ECE by **46%** (→ 0.0573) and Brier by
   15%, while *improving* accuracy from 0.906 to 0.958.

3. **The Score head is internally coherent but concentrated.** All 192
   predictions have valid, monotone cumulative distributions (monotonicity =
   1.000). Top-class probability averages 0.905 vs a uniform baseline of 0.20,
   with normalized entropy 0.162 — the model commits to a single class in most
   cases.

4. **Score and Noul agree strongly** (Spearman ρ = 0.871), suggesting the two
   heads rank examples consistently even though one outputs a probability and
   the other an ordinal rating.

5. **The reported `confidence` scalar is not the top-class probability.**
   Across the Score head, mean confidence is 0.839 while mean top-class
   probability is 0.905 — a gap of 0.066, indicating the SDK applies additional
   calibration to the confidence field.

## Limitations

- The email dataset is highly imbalanced (94.8% positive), so low-probability
  bins contain only 1–8 examples and estimates are noisy.
- Choice's top-class confidence is pinned near 1.0 for ~86% of examples, so
  multiclass ECE is dominated by a single bin.
- **No ordinal ground truth for Score**, so calibration ECE was not computed;
  the Score analysis is limited to internal consistency and cross-head
  agreement.
- Only a single 50/50 fit/eval split was used (fixed seed). K-fold
  cross-validation would give tighter confidence intervals.

## Results
- <img width="369" height="498" alt="16" src="https://github.com/user-attachments/assets/38344d6c-288d-4936-a853-a2c079c495bf" />
- <img width="378" height="546" alt="15" src="https://github.com/user-attachments/assets/4dc74f5a-9bf0-40ff-8c85-f8def2d4b12b" />
- <img width="370" height="498" alt="14" src="https://github.com/user-attachments/assets/53e8c5d3-e263-4e85-8078-036aa2eee7bf" />
- <img width="570" height="419" alt="13" src="https://github.com/user-attachments/assets/a53770f2-9b47-4d48-add3-46fe60cd437c" />
- <img width="1223" height="369" alt="11" src="https://github.com/user-attachments/assets/d0c82a49-f438-4f43-864e-52a16766213d" />
- <img width="1284" height="344" alt="12" src="https://github.com/user-attachments/assets/7d75e32b-2cb2-47d7-b529-6eb261b2cd6a" />
- <img width="387" height="559" alt="18" src="https://github.com/user-attachments/assets/aa5a6dbc-ebf9-4b3b-9166-c426d62109b8" />

## Cost
<img width="299" height="488" alt="eval" src="https://github.com/user-attachments/assets/6c28324b-50aa-4542-b23b-4432433cd74e" />

Total evaluation cost: $0.004762 (190+ examples)[New SignUp was paused due to immense demand then waitlist for typsafe.ai same for vercel.ai gateway]

## Reproduce
- TypeSafeClient(base_url="https://api.typesafe.pro",api_key="anonymous")
- python evaluate.py          
- python run_calibration.py
- python run_calibration.py 
- streamlit 
