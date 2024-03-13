import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def density_plot():
    df = pd.read_csv('data_with_predictions_11.csv')
    # Create a new column for the combination of state_i and state_j as a tuple
    df['State_Combination'] = list(zip(df.state_i, df.state_j))
    # Convert to string for categorical coloring
    df['State_Combination'] = df['State_Combination'].astype(str)
    # Now, plot the density plots for Predictions and Actuals
    plt.figure(figsize=(12, 6))
    # Melt the DataFrame for sns.kdeplot
    df_melted = df.melt(id_vars="State_Combination",
                        value_vars=["Predictions", "Actuals"],
                        var_name="Type",
                        value_name="Value")
    # Density plot with color gradient based on State_Combination
    sns.kdeplot(data=df_melted, x="Value",
                hue="State_Combination", color="Type",
                fill=True, common_norm=False, palette="viridis", alpha=0.5)
    plt.title("Density Plot of Predictions and Actuals by State Combination")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()


#density_plot()

df = pd.read_csv('data_with_predictions_12.csv')

# Create a new column for the combination of state_i and state_j as a tuple
df['State_Combination'] = list(zip(df.state_i, df.state_j))
df['State_Combination'] = df['State_Combination'].astype(str)  # Convert to string for categorical coloring

# Plotting the scatter plot for Predictions vs. Actuals
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=df, x="Predictions", y="Actuals", hue="State_Combination", palette="viridis", alpha=0.6, ax=ax)

x = np.linspace(0, df.Predictions.max())
y = x
sns.lineplot(x=x, y=y, ax=ax, color="black")

fig.title("Scatter Plot of Actuals vs. Predictions by State Combination")
fig.xlabel("Predictions")
fig.ylabel("Actuals")
fig.tight_layout()
fig.legend(title='State Combination')
fig.show()