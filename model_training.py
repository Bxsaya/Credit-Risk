from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_engineering import train_test_split
from preprocessing import label_encoding, hot_encoding

def lr_training():
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
    y = df_encoded['Risk']
    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logreg = LogisticRegression(max_iter = 1000)
    logreg.fit(X_train, y_train)
    logreg_predictions = logreg.predict(X_test)

    # Calculate evaluation metrics for Logistic Regression
    logreg_accuracy = accuracy_score(y_test, logreg_predictions)
    logreg_precision = precision_score(y_test, logreg_predictions)
    logreg_recall = recall_score(y_test, logreg_predictions)
    logreg_f1 = f1_score(y_test, logreg_predictions)

def xgboost_training():
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
    y = df_encoded['Risk']
    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X_train, y_train)
    xgb_predictions = xgb_classifier.predict(X_test)

    # Calculate evaluation metrics for XGBoost
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    xgb_precision = precision_score(y_test, xgb_predictions)
    xgb_recall = recall_score(y_test, xgb_predictions)
    xgb_f1 = f1_score(y_test, xgb_predictions)