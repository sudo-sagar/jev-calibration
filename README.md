# Jev Calibration Report

Independent evaluation of Jev's calibrated probabilities.

## Methodology
- 190+ labeled examples from [real data from gmail]
- Each example: one Noul question with known ground truth
- Jev returned probability (0-1) for each
- Compared probability bins against actual accuracy

## Results
- <img width="542" height="187" alt="metrics" src="https://github.com/user-attachments/assets/44da041e-ca12-4577-9bd4-e23c5fa7ae7b" />
- <img width="1302" height="291" alt="result" src="https://github.com/user-attachments/assets/85f1ee96-3eb1-46fa-aef3-bf1f3a7744a3" />

## Findings
- At probability 0.9, Jev was correct X% of the time
- At probability 0.5, Jev was correct Y% of the time
- <img width="532" height="609" alt="analysis" src="https://github.com/user-attachments/assets/756bb528-7c39-4928-80bd-557c6d06bf95" />
- <img width="495" height="577" alt="Calibration" src="https://github.com/user-attachments/assets/9979d824-c43b-4026-9928-7577ac750357" />






## Cost
Total evaluation cost: $0.004762 (190+ examples)[New SignUp was paused due to immense demand then waitlist for typsafe.ai same for vercel.ai gateway]

## Reproduce
- TypeSafeClient(base_url="https://api.typesafe.pro",api_key="anonymous")
- python evaluate.py
- python app.py
