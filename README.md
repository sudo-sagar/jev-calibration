# Jev Calibration Report

Independent evaluation of Jev's calibrated probabilities.

## Methodology
- 190+ labeled examples from [real data from gmail]
- Each example: one Noul question with known ground truth
- Jev returned probability (0-1) for each
- Compared probability bins against actual accuracy

## Results
[Insert calibration_plot.png]

## Findings
- At probability 0.9, Jev was correct X% of the time
- At probability 0.5, Jev was correct Y% of the time

## Cost
Total evaluation cost: $0 (190+ examples)[New SignUp was paused due to immense demand then waitlist for typsafe.ai same for vercel.ai gateway]

## Reproduce
TypeSafeClient(base_url="https://api.typesafe.pro",api_key="anonymous")
python evaluate.py
python calibration.py
python app.py
