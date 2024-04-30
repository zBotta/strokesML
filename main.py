
"""
This script makes a ML pipeline on the Stroke data set.

We are going to skip the data analysis part with the graphics, as it is better to 
show it in the Jupyter Notebook.

The jupyter notebook can be found in the "root/notebooks" folder

All the functions are centralised in the script MLtools
"""

# File with our functions and classes
from ml_tools import get_numerical_columns, get_categorical_columns, SelectorVarImpRfC  

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection._search import GridSearchCV

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline  # This pipeline is compatible with using SMOTE (the standard sklearn Pipeline is not compatible!)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

# Import raw data
df = pd.read_csv(filepath_or_buffer="data/stroke_data.csv", sep=",", header=0)
df_raw = df.copy()

# Pre-processing
df_proc = df_raw.copy()
df_proc = df_proc.drop(["id"], axis = 1)
df_proc = df_proc.drop(["smoking_status"], axis = 1)
df_proc = df_proc.drop(["ever_married"], axis = 1)
    # NA values on the BMI column
df_proc = df_proc.dropna()
    # Drop the Gender other
idx = df_proc[df_proc.gender == "Other"].index
df_proc = df_proc.drop(index=idx, axis = 0)
    # The data that is already hot econded in the dataset
df_proc.hypertension = df_proc.hypertension.astype(bool)
df_proc.heart_disease = df_proc.heart_disease.astype(bool)

# End of pre-processing

X = df_proc.drop(["stroke"], axis = 1) ## Remember that df_proc is the data after cleaning, not yet transformed
y = df_proc["stroke"]

# Define ratios, w.r.t. whole dataset.
ratio_train = 0.7
ratio_val = 0.15
ratio_test = 0.15

# Produce test split
X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, stratify=y, test_size=ratio_test, random_state=0) # We stratify the target variable as we do not have a balance data set

# Adjusts val ratio, w.r.t. remaining dataset.
ratio_remaining = 1 - ratio_test
ratio_val_adjusted = ratio_val / ratio_remaining

# Produces train and val splits.
X_train, X_val, y_train, y_val = train_test_split(X_remaining, y_remaining, stratify=y_remaining, test_size=ratio_val_adjusted, random_state=0)

# PIPELINE

# Create a pre-processor
numerical_columns = get_numerical_columns(X_train)
nominal_columns = get_categorical_columns(X_train)
        
numerical_pipeline = Pipeline([('scaler', StandardScaler())])
nominal_pipeline = Pipeline([('hot_encoder', OneHotEncoder(drop="if_binary", dtype=np.int64, handle_unknown='ignore'))]) # , sparse_output=False
        
preproc = ColumnTransformer([
            ('numerical_transformer', numerical_pipeline, numerical_columns),
            ('nominal_transformer', nominal_pipeline, nominal_columns),
        ], remainder="passthrough")

# Create a SMOTENC object
smote_nc =  SMOTENC(categorical_features=[3, 4, 5, 6, 7, 8, 9, 10, 11], sampling_strategy="minority", random_state=0)

# Pre-process, smoting, variable selection and model pipelines
mega_pipe = ImbPipeline(steps=[("preprocessor", preproc),
                               ("smoting", smote_nc),
                               ("selector", SelectorVarImpRfC()),
                               ("clf", LogisticRegression())   # virtual classifier model. All classifier models will be defined here
                              ])

# pipe_fitted = mega_pipe.fit(X_train, y_train)
# my_pipe.score(X_val, y_val)

# Grid Search Hyper-parameters
    # SVC params
c_values_svc = [0.01, 0.1, 0.5, 1, 5, 10, 20]
    # Linear Regression params
c_values_lr = [0.01, 0.1, 0.2, 0.5, 1]
    # RandomForest Classifier params
n_estimators = [5, 10, 30, 50, 100]
min_samples_leaf = [1, 2, 5, 10]
max_depths = [3, 5, 15, 30, 40]

hyper_param_rf = {"clf": [RandomForestClassifier(random_state=0)], 
                  "clf__n_estimators": n_estimators, 
                  "clf__min_samples_leaf": min_samples_leaf,
                  "clf__max_depth": max_depths
                 }

hyper_param_lr = {"clf": [LogisticRegression()],
                  "clf__solver": ["lbfgs" , "saga"], #liblinear", "newton-cg", "newton-cholesky","saga"
                  "clf__C": c_values_lr,
                  "clf__multi_class":  ["ovr", "multinomial"]
                 }

hyper_param_svc = {"clf": [SVC()],
                   "clf__C": c_values_svc,
                   "clf__kernel": ["poly"], # "linear", "rbf", "sigmoid"
                   "clf__decision_function_shape": ["ovo"] # ,"ovr"
                  }

# Using grid-search on the pipeline
search_space =[hyper_param_rf,
               hyper_param_lr,
               hyper_param_svc]

print("STARTING GRID SEARCH...")
gs = GridSearchCV(estimator=mega_pipe, param_grid=search_space, cv=3, verbose=2, error_score='raise')
gs_fitted = gs.fit(X_train, y_train)

val_score = gs_fitted.best_estimator_.score(X_val, y_val)
test_score = gs_fitted.best_estimator_.score(X_test, y_test)

print(f"END OF GRID SEARCH\nBEST ESTIMATOR {gs_fitted.best_estimator_}")

print(f"SCORE OF BEST ESTIMATOR ON TEST SET: {test_score}")
