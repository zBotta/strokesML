import numpy as np
import pandas as pd

from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection._search import BaseSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# PREPROCESSOR

# a function for getting the columns by dtype
def get_categorical_columns(df):
    """ categorical columns """
    categorical_columns_selector = make_column_selector(dtype_include= ['object'])
    categorical_columns = categorical_columns_selector(df)
    return categorical_columns


def get_numerical_columns(df):
    """ numerical columns. In our data set they are only of type float"""
    numerical_columns_selector = make_column_selector(dtype_exclude=['object','bool'])  # include only float (continous) variables
    numerical_columns = numerical_columns_selector(df)
    return numerical_columns


def get_binary_columns(df):
    """ columns that are already one hot encoded in the data set"""
    one_hot_columns_selector = make_column_selector(dtype_include= ['bool'])
    one_hot_columns = one_hot_columns_selector(df)
    return one_hot_columns


def transform_to_int(df):
    return df.astype(int)


def one_hot_and_std_transformation(df):
    """ A function that transforms a pre-cleaned data set by using:
        One Hot Encoding on the categorical variables
        Standardisation  on the numerical variables
        The already Hot econded variables are passed through with the remainder="passthrough" argument.
    """
    df = df.copy()
    
    numerical_columns = get_numerical_columns(df)
    nominal_columns = get_categorical_columns(df)
    
    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    nominal_pipeline = Pipeline([('hot_encoder', OneHotEncoder(drop="if_binary", dtype=np.int64, handle_unknown='ignore'))]) # , sparse_output=False
    
    preprocessor = ColumnTransformer([
        ('numerical_transformer', numerical_pipeline, numerical_columns),
        ('nominal_transformer', nominal_pipeline, nominal_columns),
    ], remainder="passthrough")
    
    preprocessor.fit(df)
    
    return preprocessor

def get_preprocessor_col_names(preprocessor):
    return [x.split("__")[1] for x in preprocessor.get_feature_names_out()]


def get_preprocessor_cat_names(preprocessor):
    cat_names = []
    for x in preprocessor.get_feature_names_out():
        if x.split("__")[0] == "nominal_transformer":
            cat_names.append(x.split("__")[1])
    return cat_names

class SelectorVarImpRfC( BaseSearchCV ):
    """ This class selects the variables by using a RandomForestClassifier and a 1SE criteria. 
        It can be used in a pipeline
    """

    #Class Constructor 
    def __init__( self ):
        self.sel_features = None
    
    #Method that describes what we need this fitter to do
    def fit( self, X, y = None ):
        # Create a RF model for doing the variable selection
        rfc = RandomForestClassifier(random_state=0) 
        rfc.fit(X, y)
        importances = rfc.feature_importances_
        rfc_std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)  # Standard deviation of each feature on all the calculated trees
        threshold = importances.min() + rfc_std.mean()
        imp_sel = importances > threshold
        self.sel_features = imp_sel # a boolean array with selected columns
        return self

    #Method that describes what we need this transformer to do
    # In this case, we select the columns from the variable importance threshold (see fit method)
    def transform( self, X, y = None ):
        df_X = pd.DataFrame(X)
        return df_X.loc[:, self.sel_features]  # The variable selection by columns

    def get_feature_names_out( self, X, y = None ):
        return self.sel_features