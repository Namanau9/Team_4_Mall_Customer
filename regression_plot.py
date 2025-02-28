import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def generate_regression_plot():
    df = pd.read_csv("Mall_Customers.csv")
    X = df[['Age']]
    y = df['Spending Score (1-100)']
    model = LinearRegression()
    model.fit(X, y)
    df['Predicted Spending Score'] = model.predict(X)
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=df['Age'], y=df['Predicted Spending Score'], color='red', label="Predicted Spending Score")
    sns.histplot(df['Spending Score (1-100)'], kde=True, bins=20, color='blue', label="Distribution of Spending Scores")
    plt.xlabel("Age")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Linear Regression: Age vs. Spending Score")
    plt.legend()
    plt.savefig("static/regression_plot.png")
    plt.close()
