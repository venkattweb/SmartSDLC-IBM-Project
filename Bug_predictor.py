# bug_predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_bug_model():
    """
    Dummy bug predictor.
    Uses a small dataset where 'lines_of_code' and 'complexity' decide if bug-prone.
    """
    data = {
        "lines_of_code": [100, 500, 200, 50, 800, 300],
        "complexity": [1, 5, 2, 1, 7, 3],
        "bug": [0, 1, 0, 0, 1, 0]  # 1 = bug-prone, 0 = safe
    }
    df = pd.DataFrame(data)

    X = df[["lines_of_code", "complexity"]]
    y = df["bug"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Bug Predictor Accuracy:", accuracy_score(y_test, y_pred))

    return model


if __name__ == "__main__":
    model = train_bug_model()
    sample = [[400, 6]]  # 400 lines, complexity 6
    print("Prediction for new module:", model.predict(sample))
