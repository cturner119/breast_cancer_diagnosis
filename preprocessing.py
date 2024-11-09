import pandas as pd
import traceback as tb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def clean_dataframe(df: pd.DataFrame, drop_columns: tuple) -> pd.DataFrame:
    """Clean the DataFrame by dropping null values, duplicates, and specified columns."""
    try:
        for column in drop_columns:
            df = df.drop(columns=[column], errors='ignore')
        df = df.dropna(axis=1)
        df = df.drop_duplicates(keep="first")
        return df
    except Exception as e:
        print("An error occurred in clean_dataframe:")
        print(tb.format_exc())

def display_dataframe_info(df: pd.DataFrame):
    """Print summary information about the DataFrame."""
    print(f"DataFrame shape: {df.shape}")
    print(f"Null value count:\n{df.isnull().sum()}")
    print(f"Duplicate count: {df.duplicated().sum()}")
    print(f"Data types:\n{df.dtypes}")

def main():
    
    #loading csv
    df = pd.read_csv("breast_cancer_kaggle.csv")
    
    #printing dataframe information
    display_dataframe_info(df)
    
    #encoding categorical column    
    label_encoder = LabelEncoder()
    df["diagnosis_encoded"] = label_encoder.fit_transform(df["diagnosis"])
    
    # #cleaning dataframe
    df = clean_dataframe(df, drop_columns=("id", "diagnosis"))

    #saving cleaned dataframe csv file
    df.to_csv("breast_cancer_kaggle_cleaned.csv", index=False)
    
    #printing final dataframe information
    display_dataframe_info(df)
    
if __name__ == "__main__":
    main()