from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
import traceback as tb
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    try:
        #load dataset and defining target and features
        df = pd.read_csv("breast_cancer_kaggle_cleaned-2.csv")
        y = df["diagnosis_encoded"]
        X = df.drop(columns=["diagnosis_encoded"])
        
        #splitting data to test and training sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=42)
        
        #scaling data
        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.fit_transform(Xtest)
        
        #creating model and defining hyperparameters
        mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                    solver='adam', max_iter=1000, random_state=42)
        
        #fitting model to training data
        mlp.fit(Xtrain, ytrain)
        
        #making predictions from test data
        ypred = mlp.predict(Xtest)
        
        #defining metrics
        accuracy = accuracy_score(ytest, ypred)
        report = classification_report(ytest, ypred)
        cm = confusion_matrix(ytest, ypred)
        
        #printing metrics
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print(f"confusion matrix:\n{cm}")
        
        #plotting confusion matrix
        plt.figure(figsize=(10, 7))
        sns.set_context("notebook", font_scale=1.25)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=("Benign", "Malignant"), yticklabels=("Benign", "Malignant", ))
        plt.xlabel("Diagnosis Predicted Labels")
        plt.ylabel("Diagnosis True Labels")
        plt.title("MLP Confusion Matrix")
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        tb.print_exc()
    
if __name__ == "__main__":
    main()
    
    