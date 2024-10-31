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
from imblearn.over_sampling import SMOTE

def load_data(file_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset split into features and targets"""
    df = pd.read_csv(file_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, 
               random_state: int=35) -> tuple:
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train: pd.DataFrame, X_test: pd.Series) -> tuple:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  
    return X_train_scaled, X_test_scaled

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int=40) -> tuple:
    """Apply SMOTE to balance the classes."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def main():
    try:
        #load dataframe and define X and y
        X, y = load_data("breast_cancer_kaggle_cleaned-2.csv", target_column="diagnosis_encoded")
        
        #split data for training and testing
        X_train, X_test, y_train, y_test = split_data(X, y)

        #scale training data
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        
        #balance training data
        X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)
        
        #defining classification models
        models = {
        "Logistic Regression": LogisticRegression(C=.1, penalty="l2", solver="liblinear", random_state=50),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, metric="manhattan", weights="uniform"),
        "Decision Tree": DecisionTreeClassifier(max_depth=None, min_samples_leaf=4, min_samples_split=10),
        "Random Forest": RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, 
                                    n_estimators=150, random_state=40),
        "Naive Bayes": GaussianNB(var_smoothing=1e-06),
        "Support Vector Machine": SVC(kernel='linear', C=1.0, max_iter=1000),
        "LightGBM": lgb.LGBMClassifier(learning_rate=.1, max_depth=10, n_estimators=200, num_leaves=31),
        "XGBoost" : XGBClassifier(max_depth=7, learning_rate=.01, min_child_weight=1,
                                  n_estimators=150, subsample=.8 ) 
        }
        
        model_accuracy = []
        model_f1 = []
        
        for model_name, model in models.items():
            model.fit(X_train_balanced, y_train_balanced)
            y_predict = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_predict)
            model_accuracy.append(accuracy*100)
            
            f1 = f1_score(y_test, y_predict)
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
    
    