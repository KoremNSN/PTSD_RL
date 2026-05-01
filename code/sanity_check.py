import pandas as pd

file_path = '../data/PSUB2_RL_data_and_questionnaire_all_subs_4.21.26.xlsx'
df = pd.read_excel(file_path)

# Filter valid trials
df = df.dropna(subset=['ChoiceKey'])

print("--- Overall Average Points by ChosenID ---")
print(df.groupby('ChosenID')['TrialPoints'].mean().sort_values(ascending=False))

print("\n--- Value Frequencies by ChosenID ---")
for stim in ['W1', 'W2', 'L1', 'L2']:
    subset = df[df['ChosenID'] == stim]
    if len(subset) == 0:
        continue
    counts = subset['TrialPoints'].value_counts(normalize=True).sort_index()
    print(f"\nStimulus {stim} (N={len(subset)}):")
    for val, prop in counts.items():
        print(f"  {val:3.0f} pts: {prop*100:.1f}%")

