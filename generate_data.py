from sklearn.datasets import make_classification
import pandas as pd
import os

# Tạo thư mục data nếu chưa có
os.makedirs("data", exist_ok=True)

# Sinh dữ liệu
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
df["target"] = y

# Lưu ra file CSV
df.to_csv("data/classification.csv", index=False)

print("Dữ liệu đã được tạo và lưu tại data/classification.")