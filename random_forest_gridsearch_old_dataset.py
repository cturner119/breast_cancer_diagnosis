import pandas as pd
import traceback as tb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset split into features and targets"""
    df = pd.read_csv(file_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, 
               random_state: int=50) -> tuple:
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train: pd.DataFrame, X_test: pd.Series) -> tuple:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  
    return X_train_scaled, X_test_scaled

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame,
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

def main():
    try:
        #load dataframe and define X and y
        X, y = load_data("breast_cancer_kaggle_cleaned.csv", target_column="diagnosis_encoded")
        
        #split data for training and testing
        X_train, X_test, y_train, y_test = split_data(X, y)

        #scale training data
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

        #defining random forest classifier
        rf = RandomForestClassifier(random_state=50)

        #defining the parameter grid
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

        #defining gridsearch parameters for random forest 
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

        #fitting model
        grid_search.fit(X_train_scaled, y_train)

        #finding and printing best parameters
        best_params = grid_search.best_params_
        print(f"\nBest Parameters:\n{best_params}")

        #best model
        best_model = grid_search.best_estimator_
        
        #evaluate model
        accuracy, report, cm = evaluate_model(best_model, X_test_scaled, y_test)

        #print metrics
        plot_results(accuracy, report, cm)

    except Exception as e:
        print("An error occurred:")
        print(tb.format_exc())

if __name__ == "__main__":
    main()