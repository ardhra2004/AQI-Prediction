import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importances(model, feature_names, title="Feature Importance"):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=pd.Series(feature_names)[indices], palette="Blues_r")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted AQI"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color="royalblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
