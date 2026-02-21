import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ID_data_mass_18122012.csv")

print("Initial shape:", df.shape)

# target

targets = ["G_total", "КГФ"]
df = df.dropna(subset=targets, how="all")

print("After target cleaning:", df.shape)

# missing

df = df.replace([" ", "", "NA", "na", "-", "--"], np.nan)

# names

features = df.columns.tolist()
print("Features:", features)

# ABS

numeric_df = df.select_dtypes(include=np.number)

corr_matrix = numeric_df.corr().abs()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation heatmap (absolute)")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()

# ratio proxy

from sklearn.feature_selection import mutual_info_regression

target_for_importance = None
for t in targets:
    if t in numeric_df.columns:
        target_for_importance = t
        break

if target_for_importance is not None:
    X = numeric_df.drop(columns=[target_for_importance], errors="ignore")
    y = numeric_df[target_for_importance].fillna(0)

    X_filled = X.fillna(0)

    if X_filled.shape[1] > 0:
        mi = mutual_info_regression(X_filled, y)
        importance = pd.Series(mi, index=X.columns).sort_values(ascending=False)

        plt.figure(figsize=(8, 6))
        importance.plot(kind="barh")
        plt.title("Feature importance")
        plt.tight_layout()
        plt.savefig("importance.png")
        plt.close()

        print("Top important features:")
        print(importance.head(10))

# statistics

print(df.describe())

# Q1/Q3

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    series = df[col].dropna()
    if len(series) == 0:
        continue

    plt.figure()
    sns.histplot(series, kde=True)

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    plt.axvline(q1, linestyle="--", label="Q1")
    plt.axvline(q3, linestyle="--", label="Q3")

    plt.title(f"Distribution: {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"dist_{col}.png")
    plt.close()

print("=== LAB1 DONE ===")
