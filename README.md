# Jev Calibration Report

Independent evaluation of Jev's calibrated probabilities.

## Methodology
- 190+ labeled examples from [real data from email]
- Each example: one Noul question with known ground truth
- Jev returned probability (0-1) for each
- Compared probability bins against actual accuracy
- Evaluated probability calibration of both **Noul** (binary) and **Choice** (categorical) heads of Jev on 192 labeled email examples.

## Findings
- Noul head (binary probability)

| Method | ECE ↓ | Brier ↓ | Accuracy |
|---|---|---|---|
| Uncalibrated | 0.1055 | 0.0582 | 0.9062 |
| Temperature scaling (T = 0.818) | 0.0930 | 0.0596 | 0.9062 |
| Isotonic regression | **0.0573** | **0.0495** | **0.9583** |

- Choice head (categorical, top-class confidence)

| Metric | Value |
|---|---|
| Multiclass ECE | 0.0490 |
| Multiclass Brier | 0.1205 |
| Accuracy | 0.9219 |
| Mean top-class confidence | 0.9709 |

- Key finding — the two heads are miscalibrated in opposite directions

| Head | Mean confidence | Accuracy | Gap | Direction |
|---|---|---|---|---|
| Noul (binary prob) | 0.870 | 0.948 | **−0.078** | under-confident |
| Choice (top-class conf) | 0.971 | 0.922 | **+0.049** | over-confident |

Noul systematically **understates** its confidence (mean probability 0.870 vs
actual positive rate 0.948), while Choice **overstates** it (mean top-class
confidence 0.971 vs accuracy 0.922). Because the miscalibration runs in
opposite directions, **a single global calibration strategy cannot fix both
heads** — each head requires its own calibrator, and the sign of the correction
differs.
- The dataset is highly imbalanced (94.8% positive), so low-confidence bins
  contain only 1–8 examples and estimates are noisy.
- The Choice head's top-class confidence is pinned near 1.0 for ~86% of
  examples, so multiclass ECE is dominated by one bin.
- Only a single 50/50 fit/eval split is used (fixed seed). K-fold
  cross-validation would give tighter confidence intervals.
- The ordinal (Score) head is not analyzed.

## Results
<img width="372" height="497" alt="1" src="https://github.com/user-attachments/assets/21bb00be-4f01-4ca5-81bf-a4f6c6117d42" />
<img width="365" height="426" alt="2" src="https://github.com/user-attachments/assets/80902d29-c766-405d-901a-c7c28cdd8827" />
<img width="545" height="190" alt="3" src="https://github.com/user-attachments/assets/a83aec95-146e-4ca4-a409-1e50aa129c33" />
<img width="535" height="599" alt="4" src="https://github.com/user-attachments/assets/c93ee9ec-9d0c-43ea-b83e-c3b61baea008" />
<img width="1302" height="291" alt="result" src="https://github.com/user-attachments/assets/953561c6-3f8f-4455-b1a4-a746fa49ccff" />

## Cost
Total evaluation cost: $0.004762 (190+ examples)[New SignUp was paused due to immense demand then waitlist for typsafe.ai same for vercel.ai gateway]

## Reproduce
- TypeSafeClient(base_url="https://api.typesafe.pro",api_key="anonymous")
- python evaluate.py         
- python run_calibration.py 
- streamlit run app.py 
