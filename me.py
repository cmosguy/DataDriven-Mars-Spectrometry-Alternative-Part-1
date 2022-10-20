#%%
import numpy as np
import pandas as pd
from generate_dataset import (generate_data)
from train_pipeline import (train)
from inference_pipeline import (predict) 
from cross_validation import (evaluate_cv)

feature = 'max'
interval = 100
save = 'data/savgol_features'
model = 'xgboost'
model_path = f"saved_models/{model}_{feature}_{interval}.pkl" 
save_prefix = f"{save}/{feature}{interval}"
n_splits=10
submission_file = f"submissions/mskf_{n_splits}_{model}_savgol_{feature}{interval}.csv"
#%%
#python generate_dataset.py -f max -i 100 -s data/savgol_features

trained = generate_data(True, feature, interval)
test = generate_data(False, feature, interval)
train_extention = f"{save_prefix}_train.csv"
test_extention = f"{save_prefix}_test.csv"

trained.to_csv(f"{save}/{train_extention}")
test.to_csv(f"{save}/{test_extention}")

# %%
# python train_pipeline.py -m lgbm -f max100 -s saved_models/mskf_lgbm_savgol_max100.pkl

TRAIN_LABELS = pd.read_csv("data/train_labels.csv", index_col="sample_id")
VALID_LABELS = pd.read_csv("data/val_labels.csv", index_col="sample_id") # stage 2
LABELS = pd.concat([TRAIN_LABELS, VALID_LABELS]) # stage 2


train_df = pd.read_csv(f"{save_prefix}_train.csv", header=[0], low_memory=False)
train_df.columns = train_df.iloc[0]
train_df = train_df.drop([0,1]).set_index('temp_bin', drop=True)

train(train_df=train_df, model_name=model, path=model_path, LABELS=LABELS, n_splits=n_splits)

#%%
#python inference_pipeline.py -m saved_models/mskf_lgbm_savgol_max100.pkl -f max100 -s submissions/mskf_lgbm_savgol_max100.csv

test_df = pd.read_csv(f"{save_prefix}_test.csv", header=[0], low_memory=False)
test_df.columns = test_df.iloc[0]
test_df = test_df.drop([0,1]).set_index('temp_bin', drop=True)
sub = predict(np.array(test_df), model_path)
sub.to_csv(submission_file)    

# %%
evaluate_cv(model_name=model, train=train_df)
# %%
