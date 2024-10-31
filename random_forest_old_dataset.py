import pandas as pd
import traceback as tb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
   #load dataset and defining target and features
    df = pd.read_csv("breast_cancer_kaggle_cleaned.csv")
    
    #defining X and y
    y = df["diagnosis_encoded"]
    X = df.drop(columns=["diagnosis_encoded"])
    
    #splitting data for testing and training
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=42)
    
    #scaling data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.fit_transform(Xtest)
    
    #defining and fitting model
    clf = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, 
                                 n_estimators=150, random_state=42)
    clf.fit(Xtrain, ytrain)
    
    ypredict = clf.predict(Xtest)
    
    #defining and printing metrics
    accuracy = accuracy_score(ytest, ypredict)
    report = classification_report(ytest, ypredict)
    cm = confusion_matrix(ytest, ypredict)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print(f"confusion matrix:\n{cm}")
    
    #plotting confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=("Postive", "Negative"),
                yticklabels=("Postive", "Negative"))
    plt.xlabel(f"Predicted Labels\nDiagnosis")
    plt.ylabel(f"Diagnosis\nTrue Labels")
    plt.title("Breast Cancer Diagnosis Confusion Matrix")
    plt.show()
    
if __name__ == "__main__":
    main()
    
    