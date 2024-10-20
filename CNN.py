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

np.random.seed(42)
tf.random.set_seed(42)

def load_data(file_path, target):
    """this function loads data frame and defines target and features

    Args:
        file_path (str): Input your CSV file path
        target (str): Input your target

    Returns:
        pd.dataframe: Returns a dataframe for X and y 
    """
    df = pd.read_csv(file_path)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def scale_data(X):
    """This function scales a dataframe using 
    the standard scaler.

    Args:
        X (pd.dataframe): Define dataframe to be scaled.

    Returns:
        X_scaled (nd.array): Scaled numpy array
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled 

def main():
    try:
        #loading dataset, defining target and features
        X, y = load_data(file_path="breast_cancer_kaggle_cleaned-2.csv", target="diagnosis_encoded")
        
        #scaling data
        X_scaled = scale_data(X)
        
        #creating number of classes
        class_number = np.unique(y).sum()
        
        # #reshaping for CNN
        X_reshaped = X_scaled.reshape(X.shape[0], X.shape[1], 1)
        
        #splitting data into test and training sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
        
        #defining 1D CNN model
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=3, activation="relu", input_shape=(Xtrain.shape[1], 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(class_number, activation="sigmoid")
        ])

        #compiling the model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()
        
        #fit the model
        history = model.fit(Xtrain, ytrain, epochs=30, batch_size=32, 
                            validation_data=(Xtest, ytest))

        #making predictions
        y_pred_probs = model.predict(Xtest)
        ypred = (y_pred_probs > 0.5).astype(int)
        
        #model metrics
        test_loss, test_accuracy, = model.evaluate(Xtest, ytest)
        report = classification_report(ytest, ypred)
        cm = confusion_matrix(ytest, ypred)
        
        #printing metrics
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("Classification Report:")
        print(report)
        print(f"confusion matrix:\n{cm}")
        
        #plotting accuracy & loss 
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
        
        #plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.set_context("notebook", font_scale=1.25)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=("Benign", "Malignant"), yticklabels=("Benign", "Malignant", ))
        plt.xlabel("Diagnosis Predicted Labels")
        plt.ylabel("Diagnosis True Labels")
        plt.title("CNN Confusion Matrix")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        tb.print_exc()

if __name__ == "__main__":
    main()

