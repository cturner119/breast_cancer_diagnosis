import pandas as pd
import traceback as tb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

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
               random_state: int=35) -> tuple:
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

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int=40) -> tuple:
    """Apply SMOTE to balance the classes."""
    try:
        smote = SMOTE(random_state=random_state)
        return smote.fit_resample(X_train, y_train)
    except Exception as e:
        print("An error occurred in apply_smote:")
        print(tb.format_exc())


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a Random Forest Classifier and return the model."""
    try:
        model = RandomForestClassifier(max_depth=None, min_samples_leaf=1,
                                    min_samples_split=2, n_estimators=150,
                                    random_state=40)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print("An error occurred in train_random_forest:")
        print(tb.format_exc())

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame,
                   y_test: pd.Series) -> tuple:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, report, cm
    except Exception as e:
        print("An error occurred in evaluate_model:")
        print(tb.format_exc())

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

        #load dataframe and define X and y
        X, y = load_data(file_path="breast_cancer_kaggle_cleaned-2.csv", target_column="diagnosis_encoded")
        
        #split data for training and testing
        X_train, X_test, y_train, y_test = split_data(X, y)

        #scale training data
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        
        #balance training data
        X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)

        #train model
        model = train_random_forest(X_train_balanced, y_train_balanced)
        
        #evaluate model
        accuracy, report, cm = evaluate_model(model, X_test_scaled, y_test)

        #print metrics
        plot_results(accuracy, report, cm)

        #plot confusion matrix
        plot_confusion_matrix(cm, xlabel="Diagnosis Predicted Labels", ylabel="Diagnosis True Labels",
                              title="Random Forest Diagnostic Confusion Matrix",
                              xtick=("Benign", "Malignant"),
                              ytick=("Benign", "Malignant"))

if __name__ == "__main__":
    main()