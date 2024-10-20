import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback as tb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
from xgboost import XGBClassifier

np.random.seed(42)

def main():
    try:
        #load dataset and defining target and features
        df = pd.read_csv("breast_cancer_kaggle_cleaned-2.csv")
        
        #defining X and y
        y = df["diagnosis_encoded"]
        X = df.drop(columns=["diagnosis_encoded"])
        
        #splitting data for testing and training
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=42)
        
        #scaling data
        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.fit_transform(Xtest)
        
        #defining classification models
        models = {
        "Logistic Regression": LogisticRegression(C=.1, penalty="l1", solver="liblinear" ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, metric="manhattan", weights="uniform"),
        "Decision Tree": DecisionTreeClassifier(max_depth=None, min_samples_leaf=4, min_samples_split=10),
        "Random Forest": RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, 
                                    n_estimators=150, random_state=42),
        "Naive Bayes": GaussianNB(var_smoothing=1e-06),
        "Support Vector Machine": SVC(kernel='linear', C=1.0, max_iter=1000),
        "LightGBM": lgb.LGBMClassifier(learning_rate=.1, max_depth=10, n_estimators=200, num_leaves=31),
        "XGBoost" : XGBClassifier(max_depth=7, learning_rate=.01, min_child_weight=1,
                                  n_estimators=150, subsample=.8 ) 
        }
        
        model_accuracy = []
        model_f1 = []
        
        for model_name, model in models.items():
            model.fit(Xtrain, ytrain)
            ypredict = model.predict(Xtest)
            
            accuracy = accuracy_score(ytest, ypredict)
            model_accuracy.append(accuracy*100)
            
            f1 = f1_score(ytest, ypredict)
            model_f1.append(f1*100)
            
            print(f"{model_name} Accuracy: {accuracy:.4f}")
            print(f"{model_name} F1 Score: {f1:.4f}")
        
        #plotting accuracy and F1 scores for all models
        bar_width = 0.35
        index = np.arange(len(models))
        plt.figure(figsize=(10, 6))

        #plotting accuracy
        bars_accuracy = plt.bar(index, model_accuracy, bar_width, label='Accuracy', color='skyblue')

        #plotting F1 Score bars
        bars_f1 = plt.bar(index + bar_width, model_f1, bar_width, label='F1 Score', color='salmon')

        #titles and labels
        plt.xlabel("Classification Models")
        plt.ylabel("Scores")
        plt.title("Classification Model Comparison: Accuracy and F1 Scores")
        plt.ylim(0, 100)  
        plt.xticks(index + bar_width / 2, models.keys(), rotation=45)
        plt.grid(axis="y")
        plt.legend()

        #annotations
        plt.bar_label(bars_accuracy, fmt='%.2f', label_type='center')
        plt.bar_label(bars_f1, fmt='%.2f', label_type='center')
        plt.tight_layout()
        plt.show()
        
    except:
        print(tb.format_exc())
        
if __name__ == "__main__":
    main()
    
    