import pandas as pd
import traceback as tb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def encode_decode(df, column, use):
    """This function encodes or decodes a column of categorical data in a 
    data frame.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        column (pd.DataFrame): Pandas DataFrame column, None skips this step
        use (str): "encode" to encode, "decode" to decode, None skips this step

    Returns:
        pd.DataFrame: Dataframe with new encoded or decoded column
    """
    try:
        label_encoder = LabelEncoder()
        if use == "encode":
            df[f"{column}_encoded"] = label_encoder.fit_transform(df[column])
            df.attrs["label_encoder"] = label_encoder
            return df
        elif use == "decode":
            label_encoder = df.attrs["label_encoder"]
            df[f"{column}_decoded"] = label_encoder.inverse_transform(df[f"{column}_encoded"])
            return df 
        elif use == None:
            return df
        elif column == None:
            return df
        else:
            return None
        
    except: print(tb.format_exc())
    
def main():
    #loading csv
    df = pd.read_csv("breast_cancer_kaggle.csv")
    
    #encoding categorical column    
    df = encode_decode(df, column= "diagnosis", use="encode")
    
    #dropping unneeded columns
    df = df.drop(columns=["id", "diagnosis"], axis= 1)
    
    duplicates = df.duplicated().sum()
    print("duplicate sum:")
    print(duplicates)
    print(f"Null sum:\n", df.isnull().sum())
    print(f"before shape", df.shape)
    #dropping rows with null values
    df = df.dropna(axis=1)

    #dropping duplicates
    df = df.drop_duplicates(keep="first")
    
    #creating cleaned dataframe csv file
    # df.to_csv("breast_cancer_kaggle_cleaned.csv", index=False)
    
    #printing dataframe shape, null sum, duplicated sum, and data types
    print(f"cleaned shape:\n{df.shape}\n")
    print(f"null value count:\n{df.isnull().sum()}\n")
    print(f"duplicated count:{df.duplicated().sum()}\n")
    print(f"data types:\n{df.dtypes}")
    
    #plotting histogram
    df.hist(bins=11, figsize=(10,8))
    plt.tight_layout()
    plt.subplots_adjust(top=.9)
    plt.show()
    
if __name__ == "__main__":
    main()