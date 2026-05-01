# Bayesian Reinforcement Learning: PTSD Learning Dynamics

This repository contains a Bayesian computational psychiatry analysis pipeline designed to understand how Reinforcement Learning (RL) parameters fluctuate longitudinally in a population of PTSD patients, and how these fluctuations are driven by daily symptom severity.

## 1. Population
The dataset tracks PTSD patients across a longitudinal design. Each participant completed up to 6 separate sessions. Along with performing a behavioral task in each session, they also completed a daily symptom questionnaire (PCL) measuring 8 distinct symptom clusters (e.g., Avoidance, Emotional Numbing, Negative Affect, Dysphoric Arousal). 

## 2. Behavioral Task
In each session, subjects completed a two-domain probabilistic decision-making task:
- **Stimuli**: Subjects are repeatedly forced to choose between 2 faces.
- **Domains**: Trials are split into a **Win** domain (TrialType = 'W') and a **Lose** domain (TrialType = 'L').
- **Contingencies**: In each domain, there is a "better" and a "worse" option:
  - The **better option** yields a favorable outcome (e.g., winning 30 points, or losing only 10 points) 75% of the time, and an unfavorable outcome (winning 10, or losing 30) 25% of the time.
  - The **worse option** has the exact opposite contingency (25% favorable / 75% unfavorable).
- **Session Resets**: Because new faces are introduced at the start of each session, Q-values for the stimuli are strictly reset to 0 at the start of every session.

## 3. Analysis Progression
The goal of this analysis was to move from standard cross-sectional parameter estimation to a dynamic, symptom-driven longitudinal model. We progressed through three distinct modeling phases, fitting 3 competing architectures (Single Alpha, Valence-dependent, and Domain-dependent) in each phase.

### Phase 1: Global Hierarchical Models
We started by fitting the dataset with a standard hierarchical structure, assuming each subject's learning rate ($\\alpha$) and inverse temperature ($\\beta$) were constant across all 6 sessions. This helped us establish a baseline for whether patients learn differently from positive vs. negative prediction errors (Valence model) or in Win vs. Lose domains (Task model).
* **Code**: `bayesian_rl_model_comparison.ipynb`

### Phase 2: Session-Varying Models
Because patients' clinical states fluctuate day to day, a static learning rate is biologically implausible. We updated the PyTensor graphs to allow $\\alpha$ to vary across sessions while remaining statistically tied to the individual using a random-intercept structure:
`alpha(sub, ses) = invlogit( baseline_intercept(sub) + session_effect(ses) )`
This correctly modeled the covariance between sessions (i.e., Day 1 learning rate is correlated with Day 6 learning rate) without needing an underpowered $6 \\times 6$ covariance matrix.
* **Code**: `bayesian_rl_session_comparison.ipynb`

### Phase 3: Linear Symptom-Driven Models
Finally, we replaced the random session effect with a targeted linear regression driven by the patient's exact daily symptom severity (from the PCL). 
`alpha(sub, ses) = invlogit( a(sub) + b(sub) * normalized_symptom(sub, ses) )`
Using the `numpyro` JAX backend for rapid sampling, the winning models now output the posterior distribution for the group-level slope ($b$), allowing us to directly quantify hypotheses like: *"Does increased emotional numbing significantly depress learning rates in the Win domain?"*
* **Code**: `bayesian_rl_symptom_comparison.ipynb`

## 4. Execution Pipeline
To fully replicate the results from scratch, execute the Python scripts in the `code/` directory in the following order:

1. **`python build_symptoms.py`**
   - Parses the raw Excel data, handles trailing spaces, extracts the questionnaire columns, and collapses them into 1 row per participant-session.
   - Calculates the sums for the 8 specific symptom clusters + Total Symptoms.
   - Saves the clean data to `data/symptoms_table.csv`.

2. **`bayesian_rl_model_comparison.ipynb`**
   - Run this notebook to fit and compare the Phase 1 static global models.

3. **`bayesian_rl_session_comparison.ipynb`**
   - Run this notebook to fit and compare the Phase 2 session-varying models.

4. **`bayesian_rl_symptom_comparison.ipynb`**
   - **Note**: At the top of this generated notebook, you can change the `SELECTED_SYMPTOM_IDX` to choose which symptom cluster drives the linear slope. Run this notebook to fit the Phase 3 models and view the posterior distribution slopes.