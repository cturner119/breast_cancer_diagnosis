import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import traceback as tb

def rf_importance(file_path, plt_title):
    """runs random forest classification and plots importance graph

    Args:
        file_path (str): input csv file name
        plt_title (str): input plot title
    """
    try:
        df = pd.read_csv(file_path)
        X = df.drop("diagnosis_encoded", axis=1)
        y = df["diagnosis_encoded"]

        #train the Random Forest Classifier
        rf_model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, 
                                            n_estimators=150, random_state=42)
        rf_model.fit(X, y)

        #get the feature importances
        feature_importances = rf_model.feature_importances_

        #create a DataFrame for the features and their importance
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": feature_importances
        })

        #sort the DataFrame by importance
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        #plot the feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df)
        plt.yticks(rotation=35)
        plt.title(plt_title)
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        tb.print_exc()
        
def main(): 
    try:
        #defining datasets
        rf_importance("breast_cancer_kaggle_cleaned.csv", plt_title1)
        rf_importance("breast_cancer_kaggle_cleaned-2.csv", plt_title2)
        
        #defining plot titles
        plt_title1 = "Random Forest Feature Importance (Original Dataset)"
        plt_title2 = "Random Forest Feature Importance (Revised Dataset)"
        
    except Exception as e:
        print(f"An error occurred: {e}")
        tb.print_exc()
        
if __name__ == "__main__":
    main()