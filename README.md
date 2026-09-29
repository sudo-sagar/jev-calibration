# Jev Calibration Report

Independent evaluation of Jev's calibrated probabilities.

## Methodology
- 190+ labeled examples from [real data from gmail]
- Each example: one Noul question with known ground truth
- Jev returned probability (0-1) for each
- Compared probability bins against actual accuracy

## Results
<img width="532" height="609" alt="analysis" src="https://github.com/user-attachments/assets/756bb528-7c39-4928-80bd-557c6d06bf95" />

## Findings
- At probability 0.9, Jev was correct X% of the time
- At probability 0.5, Jev was correct Y% of the time
- <img width="499" height="554" alt="Calibration" src="https://github.com/user-attachments/assets/d3d643e0-c357-4498-a623-5f5c05bf0874" />
- <img width="1302" height="291" alt="result" src="https://github.com/user-attachments/assets/b93acf8a-95be-4afc-8f72-a8a6c6776048" />





## Cost
Total evaluation cost: $0.004762 (190+ examples)[New SignUp was paused due to immense demand then waitlist for typsafe.ai same for vercel.ai gateway]

## Reproduce
- TypeSafeClient(base_url="https://api.typesafe.pro",api_key="anonymous")
- python evaluate.py
- python app.py
