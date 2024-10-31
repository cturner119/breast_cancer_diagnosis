import tensorflow as tf
import pandas as pd
import traceback as tb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE


tf.random.set_seed(42)

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

def plot_results(accuracy: float, report: str, cm: pd.DataFrame):
    """Plot evaluation results including accuracy, classification report, and confusion matrix."""
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print(f"confusion matrix:\n{cm}")

def plot_history(history):
    """Plot training and validation loss and accuracy."""
    plt.figure(figsize=(12, 5))

    #loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("CNN Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    #accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("CNN Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
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
        #loading dataset, defining target and features
        X, y = load_data(file_path="breast_cancer_kaggle_cleaned-2.csv", 
                         target_column="diagnosis_encoded")
        
        #splitting data into test and training sets
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        #scaling data
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        
        #balance training data
        X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)
        
        #creating number of classes
        class_number = np.unique(y).sum()
        
        # #reshaping for CNN
        X_train_reshaped = X_train_balanced.reshape(X_train_balanced.shape[0], X.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X.shape[1], 1)
        
        #defining 1D CNN model
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=3, activation="relu", input_shape=(X_train_reshaped.shape[1], 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(class_number, activation="sigmoid")
        ])

        #compiling the model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()
        
        #fit the model
        history = model.fit(X_train_reshaped, y_train_balanced, epochs=30, batch_size=32, 
                            validation_data=(X_test_reshaped, y_test))

        #making predictions
        y_pred_probs = model.predict(X_test_reshaped)
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        #model metrics
        test_loss, test_accuracy, = model.evaluate(X_test_reshaped, y_test)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        #printing metrics
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("Classification Report:")
        print(report)
        print(f"confusion matrix:\n{cm}")
        
        #plot training and loss history
        plot_history(history)
        
        #plot confusion matrix
        plot_confusion_matrix(cm, xlabel="Diagnosis Predicted Labels", ylabel="Diagnosis True Labels",
                              title="CNN Diagnostic Confusion Matrix",
                              xtick=("Benign", "Malignant"),
                              ytick=("Benign", "Malignant"))

    except Exception as e:
        print(f"An error occurred: {e}")
        tb.print_exc()

if __name__ == "__main__":
    main()

