import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    # Handle missing values in numerical columns
    numerical_columns = ['Age', 'Credit amount', 'Duration']
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    # Handle missing values in categorical columns
    categorical_columns = ['Sex','Housing','Saving accounts', 'Checking account', 'Purpose']
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    return df
    


def label_encoding(df):
    le = LabelEncoder()
    le_count = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1
    return df

def hot_encoding(df):
    categorical_columns = ['Sex','Housing','Saving accounts', 'Checking account', 'Purpose']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    return df_encoded