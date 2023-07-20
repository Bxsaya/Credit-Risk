import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from preprocessing import hot_encoding

def visualize_categorical_columns(df):
    # Define the list of categorical columns
    categorical_columns = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    # Calculate the number of rows and columns for the subplot grid
    num_plots = len(categorical_columns)
    num_rows = (num_plots + 1) // 2  # Round up to the nearest integer
    num_cols = min(num_plots, 2)

    # Calculate the figure size based on the number of columns
    fig_width = 6 * num_cols
    fig_height = 6 * num_rows

    # Create the subplot grid with increased size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    # Iterate over each categorical column and create a countplot in the corresponding subplot
    for i, column in enumerate(categorical_columns):
        row = i // num_cols
        col = i % num_cols

        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        sns.countplot(x=column, hue='Risk', data=df, ax=ax)

        ax.set_title(column)

        total = len(df[column])  # Total number of data points

        for p in ax.patches:
            height = p.get_height()
            percentage = height / total * 100
            ax.annotate(f'{percentage:.2f}%', (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom')

    # Adjust the spacing between subplots
    plt.tight_layout()

    plt.show()


def visualize_numerical_columns(df):
    # Define the list of numerical columns
    numerical_columns = ['Age', 'Job', 'Credit amount', 'Duration']

    # Calculate the number of rows and columns for the subplot grid
    num_plots = len(numerical_columns)
    num_rows = (num_plots + 1) // 2  # Round up to the nearest integer
    num_cols = min(num_plots, 2)

    # Calculate the figure size based on the number of columns
    fig_width = 8 * num_cols
    fig_height = 6 * num_rows

    # Create the subplot grid with increased size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    # Iterate over each numerical column and create a boxplot in the corresponding subplot
    for i, column in enumerate(numerical_columns):
        row = i // num_cols
        col = i % num_cols

        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        sns.boxplot(x=column, y='Risk', data=df, ax=ax)

        ax.set_title(column)

    # Adjust the spacing between subplots
    plt.tight_layout()

    plt.show()

    

def correlation_analysis(df):
    categorical_columns = ['Sex','Housing','Saving accounts', 'Checking account', 'Purpose']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    df2 = df_encoded[['Age', 'Sex', 'Job', 'Credit amount', 'Duration',
       'Housing_free', 'Housing_own', 'Housing_rent',
       'Saving accounts_Unknown', 'Saving accounts_little',
       'Saving accounts_moderate', 'Saving accounts_quite rich',
       'Saving accounts_rich', 'Checking account_Unknown',
       'Checking account_little', 'Checking account_moderate',
       'Checking account_rich', 'Purpose_business', 'Purpose_car',
       'Purpose_domestic appliances', 'Purpose_education',
       'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs',
       'Purpose_vacation/others']] 

    correlations = df2.corrwith(df_encoded['Risk'])
    correlations = correlations[correlations!=1]
    positive_correlations = correlations[correlations>0].sort_values(ascending=False)
    negative_correlations = correlations[correlations<0].sort_values(ascending=False)

    # Select the independent variables for correlation analysis
    # Compute the correlation matrix
    correlation_matrix = df2.corr()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')

    # Set plot title
    plt.title("Correlation Matrix of Independent Variables")

    # Display the heatmap
    plt.show()