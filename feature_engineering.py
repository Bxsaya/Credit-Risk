import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import hot_encoding, label_encoding
import pandas as pd

def variance_inflation_factor(X):
    df = label_encoding(df)
    df_encoded = hot_encoding(df)
    X = df_encoded[['Age', 'Sex', 'Job', 'Credit amount', 'Duration',
       'Housing_free', 'Housing_own', 'Housing_rent',
       'Saving accounts_Unknown', 'Saving accounts_little',
       'Saving accounts_moderate', 'Saving accounts_quite rich',
       'Saving accounts_rich', 'Checking account_Unknown',
       'Checking account_little', 'Checking account_moderate',
       'Checking account_rich', 'Purpose_business', 'Purpose_car',
       'Purpose_domestic appliances', 'Purpose_education',
       'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs',
       'Purpose_vacation/others']] 
    # Add a constant column for the intercept term
    X= sm.add_constant(X)

    # Calculate VIF for each independent variable
    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF'] = [sm.OLS(X[col], X.drop(columns=[col])).fit().rsquared for col in X.columns]
    return vif

def train_test_split(df_encoded):
    df = label_encoding(df)
    df_encoded = hot_encoding(df)
    # Select the independent variables and the target variable
    X = df_encoded[['Age', 'Sex', 'Job', 'Credit amount', 'Duration',
                'Housing_free', 'Housing_own', 'Housing_rent',
                'Saving accounts_Unknown', 'Saving accounts_little',
                'Saving accounts_moderate', 'Saving accounts_quite rich',
                'Saving accounts_rich', 'Checking account_Unknown',
                'Checking account_little', 'Checking account_moderate',
                'Checking account_rich', 'Purpose_business', 'Purpose_car',
                'Purpose_domestic appliances', 'Purpose_education',
                'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs',
                'Purpose_vacation/others']]
    y = df_encoded['Risk']
    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
