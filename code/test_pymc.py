import pandas as pd
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import matplotlib.pyplot as plt

file_path = '../data/PSUB2_RL_data_and_questionnaire_all_subs_4.21.26.xlsx'
df = pd.read_excel(file_path)

# Filter out trials with missing choices
df = df.dropna(subset=['ChoiceKey'])

# Map stimulus IDs to integers 0, 1, 2, 3
stim_map = {'W1': 0, 'W2': 1, 'L1': 2, 'L2': 3}
df['LeftID_int'] = df['LeftID'].map(stim_map)
df['RightID_int'] = df['RightID'].map(stim_map)
df['ChosenID_int'] = df['ChosenID'].map(stim_map)

# Map ChoiceKey to 0 (Left) and 1 (Right)
df['Choice'] = df['ChoiceKey'] - 1

# Scale rewards by dividing by 30
df['RewardScaled'] = df['TrialPoints'] / 30.0

# Take one subject and one session for testing
sub = df['Sub'].unique()[0]
ses = df['Ses'].unique()[0]

df_sub = df[(df['Sub'] == sub) & (df['Ses'] == ses)].copy()
df_sub = df_sub.sort_values('Trial')

left_id = df_sub['LeftID_int'].values.astype(int)
right_id = df_sub['RightID_int'].values.astype(int)
choice = df_sub['Choice'].values.astype(int)
reward = df_sub['RewardScaled'].values
chosen_id = df_sub['ChosenID_int'].values.astype(int)

trials = len(df_sub)

def update_Q(left, right, chosen, rew, Q_t, alpha):
    # Calculate probability of choosing right
    # Softmax with Q values, handled later using beta
    
    # Q update
    Q_t_next = pt.set_subtensor(Q_t[chosen], Q_t[chosen] + alpha * (rew - Q_t[chosen]))
    
    return Q_t_next

with pm.Model() as rl_model:
    alpha = pm.Beta('alpha', alpha=1, beta=1)
    beta = pm.HalfNormal('beta', sigma=10)
    
    Q_init = pt.zeros(4)
    
    # Scan through trials
    Q_seq, updates = pytensor.scan(
        fn=update_Q,
        sequences=[
            pt.as_tensor_variable(left_id),
            pt.as_tensor_variable(right_id),
            pt.as_tensor_variable(chosen_id),
            pt.as_tensor_variable(reward)
        ],
        outputs_info=[Q_init],
        non_sequences=[alpha],
        strict=True
    )
    
    # Q_seq gives Q-values AT THE END of each trial.
    # To get Q-values at the START of each trial to calculate probabilities,
    # we prepend Q_init and drop the last state.
    Q_start = pt.concatenate([pt.shape_padleft(Q_init), Q_seq[:-1]], axis=0)
    
    # Extract Q-values for left and right options presented on each trial
    Q_left = Q_start[pt.arange(trials), left_id]
    Q_right = Q_start[pt.arange(trials), right_id]
    
    # Probability of choosing Right (Choice == 1)
    p_right = pm.math.sigmoid(beta * (Q_right - Q_left))
    
    # Likelihood
    choice_obs = pm.Bernoulli('choice_obs', p=p_right, observed=choice)
    
    print("Model built and compiled logp.")
    # Run a quick sample to make sure it works
    idata = pm.sample(draws=100, tune=100, cores=1, progressbar=False)
    print("Sampling successful.")
