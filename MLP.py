from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
import traceback as tb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

def load_data(file_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset split into features and targets"""
    df = pd.read_csv(file_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, 
               random_state: int=42) -> tuple:
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train: pd.DataFrame, X_test: pd.Series) -> tuple:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  
    return X_train_scaled, X_test_scaled

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int=42) -> tuple:
    """Apply SMOTE to balance the classes."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)

def evaluate_model(model: MLPClassifier, X_test: pd.DataFrame,
                   y_test: pd.Series) -> tuple:
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm

def plot_results(accuracy: float, report: str, cm: pd.DataFrame):
    """Plot evaluation results including accuracy, classification report, and confusion matrix."""
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print(f"confusion matrix:\n{cm}")
    
def plot_confusion_matrix(cm: pd.DataFrame, xlabel: str, ylabel: str,
                          title: str, xtick: tuple, ytick: tuple):
    """Plot the confusion matrix."""
    plt.figure(figsize=(10, 7))
    sns.set_context("notebook", font_scale=1.25)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=(xtick), 
                yticklabels=(ytick))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

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

        #creating model and defining hyperparameters
        model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                    solver='adam', max_iter=1000, random_state=40)
        
        #fitting model to training data
        model.fit(X_train_balanced, y_train_balanced)
        
        #evaluate model
        accuracy, report, cm = evaluate_model(model, X_test_scaled, y_test)

        #print metrics
        plot_results(accuracy, report, cm)

        #plot confusion matrix
        plot_confusion_matrix(cm, xlabel="Diagnosis Predicted Labels", ylabel="Diagnosis True Labels",
                              title="MLP Diagnostic Confusion Matrix",
                              xtick=("Benign", "Malignant"),
                              ytick=("Benign", "Malignant"))

        
    except Exception as e:
        print(f"An error occurred: {e}")
        tb.print_exc()
    
if __name__ == "__main__":
    main()
    
    