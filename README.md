# Jev Calibration Report

Independent evaluation of Jev's calibrated probabilities.

## Methodology
- 190+ labeled examples from [real data from gmail]
- Each example: one Noul question with known ground truth
- Jev returned probability (0-1) for each
- Compared probability bins against actual accuracy

## Results
<img width="808" height="428" alt="image" src="https://github.com/user-attachments/assets/02325c18-0032-4231-a467-49f7badd5eb0" />
<img width="897" height="955" alt="image" src="https://github.com/user-attachments/assets/c3b0b00a-ac80-4c1b-8457-9eb835680824" />

## Findings
- At probability 0.9, Jev was correct X% of the time
- At probability 0.5, Jev was correct Y% of the time
- <img width="940" height="456" alt="image" src="https://github.com/user-attachments/assets/c0c708d2-b789-4609-b29c-981d7217af8a" />

## Cost
Total evaluation cost: $0 (190+ examples)[New SignUp was paused due to immense demand then waitlist for typsafe.ai same for vercel.ai gateway]

## Reproduce
TypeSafeClient(base_url="https://api.typesafe.pro",api_key="anonymous")
python evaluate.py
python app.py
python calibration.py
python app.py
