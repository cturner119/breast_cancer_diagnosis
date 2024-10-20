import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback as tb

def main():
    try:
        #reading csv
        df = pd.read_csv("breast_cancer_kaggle_cleaned.csv")
        
        #defining correlation matrix
        corr_matrix = df.corr().round(2)
        
        #creating a threshold and filtered matrix
        threshold = 0.7
        filtered_corr = corr_matrix[(corr_matrix >= threshold) | (corr_matrix <= -threshold)]

        #plotting the correlation matrix
        plt.figure(figsize=(10,8))
        sns.set_context("notebook", font_scale=.5)
        plot = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 6})
        plt.xticks(rotation=45)
        plot.set_title("Breast Cancer Diagnostic Correlation Matrix", fontsize=10)

        #plotting the filtered correlation matrix
        plt.figure(figsize=(10,8))
        sns.set_context("notebook", font_scale=.5)
        plot = sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', annot_kws={"size": 6})
        plt.xticks(rotation=45)
        plot.set_title("Strong Correlations Threshold: >.7", fontsize=10)
        plt.show()
        
        #removing highly correlated columns
        df = df.drop(df.columns[10:30], axis=1)
        df = df.drop(columns=["perimeter_mean", "area_mean", "concave points_mean", "compactness_mean"])
        corr_matrix2 = df.corr().round(2)
        filtered_corr2 = corr_matrix2[(corr_matrix2 >= threshold) | (corr_matrix2 <= -threshold)]

        #plotting the new correlation matrix
        plt.figure(figsize=(10,8))
        sns.set_context("notebook", font_scale=1)
        plot = sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', annot_kws={"size": 10})
        plt.xticks(rotation=15)
        plt.yticks(rotation=45)
        plot.set_title("Revised Breast Cancer Diagnostic Correlation Matrix", fontsize=15)
        
        #plotting the new filtered correlation matrix
        plt.figure(figsize=(10,8))
        sns.set_context("notebook", font_scale=1)
        plot = sns.heatmap(filtered_corr2, annot=True, cmap='coolwarm', annot_kws={"size": 10})
        plt.xticks(rotation=15)
        plt.yticks(rotation=45)
        plot.set_title("Revised Strong Correlations Threshold: >.7", fontsize=15)
        plt.show()
        
        #creating a revised dataset
        # df.to_csv("breast_cancer_kaggle_cleaned-2.csv", index=False)
        
    except: print(tb.format_exc())

if __name__ == "__main__":
    main()


