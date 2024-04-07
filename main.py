
"""
This script makes a ML pipeline on the Stroke data set.

We are going to skip the data analysis part with the graphics, as it is better to 
show it in the Jupyter Notebook.

The jupyter notebook can be found in the "root/notebooks" folder

All the functions are centralised in the script MLtools
"""

import ml_tools # File containing all the functions

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from imblearn.over_sampling import SMOTENC

df_raw = pd.read_csv(filepath_or_buffer="data/stroke_data.csv", sep=",", header=0)


# Data cleaning
df_proc = df_raw.copy()

# Drop the id column and the Smoking Status.
df_proc = df_proc.drop(["id"], axis = 1)
df_proc = df_proc.drop(["smoking_status"], axis = 1)

# Drop NA values on the BMI column
df_proc = df_proc.dropna()

# Erase the "other gender individuals
other_idx = df_proc[df_proc.gender == "Other"].index
df_proc = df_proc.drop(index=other_idx, axis = 0) 

# To favor automation. The data that is already hot econded in the dataset
df_proc.hypertension = df_proc.hypertension.astype(bool)
df_proc.heart_disease = df_proc.heart_disease.astype(bool)

# SPLITING DATA. We save an untouched test split

X = df_proc.drop(["stroke"], axis = 1) ## Remember that df_proc is the data after cleaning, not yet transformed
y = df_proc["stroke"]

# Produce test split
# We stratify the target variable as we do not have a balance data set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0) 
 

# Data transforming
preprocessor_train = ml_tools.one_hot_and_std_transformation(X_train)
preprocessor_test = ml_tools.one_hot_and_std_transformation(X_test)

# Transformed train and test splits
X_train_preproc = preprocessor_train.transform(X_train)
X_test_preproc = preprocessor_test.transform(X_test)

var_cols = ml_tools.get_preprocessor_col_names(preprocessor_train)

# convert it to dataFrames to increase its readability
df_transf_x_train = pd.DataFrame(X_train_preproc, columns=var_cols)
df_transf_x_test = pd.DataFrame(X_test_preproc, columns=var_cols)

# SMOTE the train split, is it has imbalanced data
# SMOTENC is capable of handling a mix of categorical and continous data
# The columns of the one hot encoding (discrete variables) go from 3 to 12

smote = SMOTENC(categorical_features=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12], sampling_strategy="minority")
print(f"X before SMOTE {X_train_preproc.shape}")
X_resampled, y_resampled = smote.fit_resample(X_train_preproc, y_train)
print(f"X after SMOTE {X_resampled.shape}")
print(f"y after SMOTE {y_resampled.shape}")

# Create a validation and training set from the SMOTED data
X_train_smote, X_val_smote, y_train_smote, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0) 


