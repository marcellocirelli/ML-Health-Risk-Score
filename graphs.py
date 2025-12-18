import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("dataset.csv")

# Generates distribution histograms for variables
hist_variables = ["healthRiskScore", "pm2.5", "temp"]
for col in hist_variables:
    plt.figure()

    plt.hist(data[col], bins = 30)

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Generates scatter plots
scatter_variables = ["pm2.5", "temp"]
for col in scatter_variables:
    plt.figure()
    plt.scatter(data[col], data["healthRiskScore"])
    plt.title(f"{col} vs healthRiskScore")
    plt.xlabel(col)
    plt.ylabel("healthRiskScore")
    plt.tight_layout()
    plt.show()