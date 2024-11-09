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

def load_data(file_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset split into features and targets"""
    try:
        df = pd.read_csv(file_path)
        y = df[target_column]
        X = df.drop(columns=[target_column])
        return X, y
    except Exception as e:
        print("An error occurred in load_data:")
        print(tb.format_exc())

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, 
               random_state: int=50) -> tuple:
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train: pd.DataFrame, X_test: pd.Series) -> tuple:
    """Scale features using StandardScaler."""
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  
        return X_train_scaled, X_test_scaled
    except Exception as e:
        print("An error occurred in scale_data:")
        print(tb.format_exc())

def main():
    
    #load dataframe and define X and y
    X, y = load_data(file_path="breast_cancer_kaggle_cleaned.csv", target_column="diagnosis_encoded")
    
    #split data for training and testing
    X_train, X_test, y_train, y_test = split_data(X, y)

    #scale training data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    #defining classification models
    models = {
    "Logistic Regression": LogisticRegression(C=.1, penalty="l2", solver="liblinear", random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, metric="manhattan", weights="uniform"),
    "Decision Tree": DecisionTreeClassifier(max_depth=None, min_samples_leaf=4, min_samples_split=10),
    "Random Forest": RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, 
                                n_estimators=150, random_state=50),
    "Naive Bayes": GaussianNB(var_smoothing=1e-06),
    "Support Vector Machine": SVC(kernel='linear', C=1.0, max_iter=1000),
    "LightGBM": lgb.LGBMClassifier(learning_rate=.1, max_depth=10, n_estimators=200, num_leaves=31),
    "XGBoost" : XGBClassifier(max_depth=7, learning_rate=.01, min_child_weight=1,
                                n_estimators=150, subsample=.8 ) 
    }
    #defining lists
    model_accuracy = []
    model_f1 = []
    
    #running all models and printing accuracy and F1-score metrics
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
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
             
if __name__ == "__main__":
    main()
    
    