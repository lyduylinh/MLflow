import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import joblib
from mlflow.models.signature import infer_signature

mlflow.set_experiment("Default")

df = pd.read_csv("data/classification.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_acc = 0
best_model = None

for depth in [4, 7, 9]:
    for n in [100, 150, 200]:
        with mlflow.start_run():
            model = RandomForestClassifier(
                n_estimators=n, max_depth=depth, random_state=42
            )
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))

            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", acc)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_train.iloc[[0]],
                signature=signature,
            )

            if acc > best_acc:
                best_acc = acc
                best_model = model

# Lưu model tốt nhất
os.makedirs("best_model", exist_ok=True)
joblib.dump(best_model, "best_model/best_model.pkl")
print(f"Mô hình tốt nhất đã lưu (accuracy={best_acc:.4f})")