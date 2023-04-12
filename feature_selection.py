import os
import numpy as np
import pandas as pd
import xgboost as xgb
from BorutaShap import BorutaShap

seed = 42

# Load data
df = pd.read_csv('data.csv')

# Feature candidates
INPUT_VARS = ['final_prf', 'total_emop','new_sex_2' , 'age', 'preop_wbc', 'preop_ptinr', 'preop_glu', 'preop_na',  'preop_alb', 'bmi', 'preop_hb', 
              'preop_plt', 'preop_bun', 'preop_cr', 'preop_aptt','preop_gpt', 'preop_got', 'preop_k','real_andur']

# Get the X and y data
df = df[INPUT_VARS]
df = df.dropna(axis=0)
X = df.drop(['final_prf'], axis=1)
y = df['final_prf'] 


# Set up the model
model = xgb.sklearn.XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=seed)

# Set up the feature selector
Feature_Selector = BorutaShap(model=model, 
                            importance_measure='shap',
                            classification=True
                            )

# Fit the feature selector
Feature_Selector.fit(X=X, y=y, n_trials=3000, random_state=seed, stratify=y)
