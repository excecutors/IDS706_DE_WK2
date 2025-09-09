# IDS706 Week 2 Data Engineering Mini Project

## Project Overview

This project demonstrates a basic end-to-end data analysis workflow using **AWS S3**, **Polars**, and machine learning models (Linear Regression and XGBoost). The dataset contains gold prices and related financial market data (SPX, GLD, USO, SLV, EUR/USD) from 2015–2025.

We:

* Pulled the dataset from **AWS S3** directly into memory
* Inspected and explored the data
* Applied filtering and grouping
* Visualized key relationships
* Trained and evaluated ML models (Linear Regression and XGBoost)

---

## Data Import from AWS S3

```python
import boto3
import polars as pl
from io import StringIO

BUCKET = "kaggle-gold-dataset"
KEY = "gold_data_2015_25.csv"
REGION = "us-east-2"

s3 = boto3.client("s3", region_name=REGION)
obj = s3.get_object(Bucket=BUCKET, Key=KEY)
data = obj["Body"].read().decode("utf-8")

df = pl.read_csv(StringIO(data))
```

---

## Data Inspection

```python
print(df.head())
print(df.schema)
print(df.describe())
print(df.shape)
```

* **Rows:** 2666
* **Columns:** 6 (Date, SPX, GLD, USO, SLV, EUR/USD)
* No missing values.

---

## Filtering & Grouping

```python
# GLD > 180
high_gold = df.filter(pl.col("GLD") > 180)

# Add Year and compute average gold price
df = df.with_columns(pl.col("Date").cast(pl.Date))
df = df.with_columns(pl.col("Date").dt.year().alias("Year"))
yearly_avg = df.group_by("Year").agg(pl.col("GLD").mean().alias("avg_GLD"))
```

**Result:** Shows rising average GLD prices, peaking in 2024.

---

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of GLD
sns.histplot(df["GLD"].to_numpy(), bins=30, kde=True)
plt.title("Distribution of Gold Prices (2015–2025)")
plt.xlabel("GLD Price")
plt.ylabel("Frequency")
plt.show()

# Scatter plot SPX vs GLD
sns.scatterplot(x=df["SPX"].to_numpy(), y=df["GLD"].to_numpy())
plt.title("SPX vs Gold (GLD)")
plt.xlabel("S&P 500 (SPX)")
plt.ylabel("Gold (GLD)")
plt.show()
```

---

## Machine Learning Exploration

### Linear Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X = df.select(["SPX", "USO", "SLV", "EUR/USD"]).to_numpy()
y = df["GLD"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
```

**Results:**

* R²: **0.921**
* RMSE: **12.82**

---

### XGBoost Regressor

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    reg_lambda=1,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
def eval_metrics(y_true, y_pred, label="Test"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{label} R²: {r2:.4f}")
    print(f"{label} RMSE: {rmse:.4f}")
    print("-" * 30)

eval_metrics(y_train, y_train_pred, "Train")
eval_metrics(y_test, y_test_pred, "Test")
```

**Results:**

* Train R²: **0.999** | RMSE: **1.61**
* Test R²: **0.993** | RMSE: **3.77**

The train/test results are very close, indicating **no major overfitting** and excellent predictive performance.

---

## Key Takeaways

* Gold prices are strongly related to SPX, SLV, USO, and EUR/USD.
* Linear Regression gives a strong baseline fit (R² ≈ 0.92).
* XGBoost significantly improves performance (R² ≈ 0.99) while avoiding overfitting.
* Polars provided efficient data handling with clean syntax.

---

## Requirements

```
boto3
polars
matplotlib
seaborn
scikit-learn
xgboost
```

---

## How to Run

```bash
git clone https://github.com/excecutors/IDS706_DE_WK2.git
cd IDS706_DE_WK2
pip install -r requirements.txt
python analysis.py  # or open analysis.ipynb in Jupyter/VS Code
```