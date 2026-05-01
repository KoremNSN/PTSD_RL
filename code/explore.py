import pandas as pd
import numpy as np

file_path = '../data/PSUB2_RL_data_and_questionnaire_all_subs_4.21.26.xlsx'
df = pd.read_excel(file_path)

print("Columns:", df.columns.tolist())
print("\nUnique Subjects:", df['Sub'].unique())
print("Unique Sessions:", df['Ses'].unique())
print("Unique TrialTypes:", df['TrialType'].unique())
print("Unique LeftIDs:", df['LeftID'].unique())
print("Unique RightIDs:", df['RightID'].unique())
print("Unique ChoiceKeys:", df['ChoiceKey'].unique())
print("Unique ChosenIDs:", df['ChosenID'].unique())
print("Unique TrialPoints:", df['TrialPoints'].unique())

# Show a few rows of actual choices
print("\nSample Data:")
print(df[['Sub', 'Ses', 'Trial', 'TrialType', 'IdxWithinType', 'LeftID', 'RightID', 'ChoiceKey', 'ChosenID', 'TrialPoints']].head(10))
