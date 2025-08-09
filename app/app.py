from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("./best_model/best_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            features = [float(request.form[f"feature_{i}"]) for i in range(20)]
            result = model.predict([features])[0]
            prediction = "Positive (1)" if result == 1 else "Negative (0)"
        except:
            prediction = "Lỗi: Dữ liệu nhập không hợp lệ!"
    return render_template("index.html", prediction=prediction)

# 🔽 THÊM PHẦN NÀY ĐỂ FLASK CHẠY ĐƯỢC
if __name__ == "__main__":
    app.run(debug=True)