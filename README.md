# Mercor-Analytics

## Overview

This project predicts the likelihood of users accepting hypothetical job offers using anonymized data from Mercor's platform. It provides a data-driven approach to identify users most likely to accept job offers, enabling more targeted and efficient recruitment efforts.

## Features
- Defines hypothetical jobs (e.g., Machine Learning Engineer, Full-Stack Developer)
- Calculates a confidence score (0-1) for each user based on:
  - Skill match
  - Experience match
  - Pay attractiveness
  - Location compatibility
- Provides both Python and SQL implementations
- Outputs summary statistics and top candidates

## Files
- `job_acceptance_analysis.md`: Full methodology, scoring logic, code samples, and summary insights
- `run_analysis.py`: Python script to run the analysis on your data
- `job_acceptance_results.csv`: (Generated) Results of the analysis for all users

## Setup
1. **Install dependencies**
   ```sh
   pip install pandas numpy
   ```
2. **Place your user data CSV** (e.g., `sql_6_2025-02-22T1512.csv`) in the project directory.

## Usage
Run the analysis script:
```sh
python run_analysis.py
```
- The script will output summary statistics and save results to `job_acceptance_results.csv`.

## Methodology
- See `job_acceptance_analysis.md` for full details on the scoring system, job definitions, and example results.

## Deliverables
- Hypothetical job definitions
- Scoring methodology
- Python and SQL code
- Results summary and recommendations

## License
MIT (add a LICENSE file if needed)

---
*Created for the Mercor Analytics Engineer Task.* 