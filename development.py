import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import ml_insights as mli
import pickle

# Set random seed                
SEED = 42 
np.random.seed(SEED)

# Define variables
OUTCOME_VAR = 'final_prf'
INPUT_VARS = ['preop_alb', 'preop_cr', 'preop_glu', 'preop_wbc', 'bmi', 'preop_ptinr', 'age', 'real_andur'] # selected by feature_selection.py

# Load data
df = pd.read_csv('data.csv')

y = df[[OUTCOME_VAR]].values.flatten().astype(bool)
x = df.loc[:, INPUT_VARS].values.astype(float)

# Split data into train and test sets
x_train, x_test,  y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=SEED, stratify=y)

# Make DataFrame for train and test sets
df_train = pd.DataFrame(x_train, columns=[INPUT_VARS])
df_train['final_prf'] = y_train
df_test = pd.DataFrame(x_test, columns=[INPUT_VARS])
df_test['final_prf'] = y_test

# Check the number of samples in each set
print('{} (event {:.1f}%) training, {} testing (event {:.1f} %) samples'.format(len(df_train), np.mean(y_train) * 100 , len(df_test), np.mean(y_test) * 100))


# Set up the hyperparameter tuning
gs = RandomizedSearchCV(n_iter=10,
                        estimator=xgb.sklearn.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False),
                        n_jobs=-1,
                        verbose=10,
                        param_distributions={
                            'tree_method': ['hist'],
                            'scale_pos_weight': [1, 10, 20, 30],
                            'learning_rate': [0.01, 0.03, 0.05, 0.07], 
                            'max_depth': [3,4,5], 
                            'n_estimators': [50, 75, 100],
                            'subsample': [0.8, 1],
                            'colsample_bytree': [0.8, 1],
                            'gamma': [0.5, 0.7, 0.9]
                        }, scoring='neg_log_loss', cv=5)

# Train the model
gs.fit(x_train, y_train)

# Get and print the best hyperparameters and score
print("========= found hyperparameter =========")
print(gs.best_params_)
print(gs.best_score_)
print("========================================")

# Save the best model
gs.best_estimator_.get_booster().save_model('bestmodel.json')

# Load the best model
model = xgb.Booster()
model.load_model('bestmodel.json') # 최종모델

# Get the predictions
y_pred = model.predict(xgb.DMatrix(x_train))

# Calibrate the predictions
splinecalib = mli.SplineCalib()
splinecalib.fit(y_pred, y_train)
y_pred = splinecalib.predict(y_pred)

# Save the calibration model
with open('splinecalib.pkl', 'wb') as f:
    pickle.dump(splinecalib, f)