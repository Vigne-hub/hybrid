import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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


def linearplot_transition_colored_dynamic(csv='data_with_predictions_27.csv', outfile="transition_colored.png"):
    # Load the dataset
    df = pd.read_csv(csv)

    # Create a new column for the combination of state_i and state_j as a tuple, then convert to string
    df['State_Combination'] = list(zip(df['state_i_int'].astype(int), df['state_j_int'].astype(int)))
    df['State_Combination'] = df['State_Combination'].astype(str)

    # List of possible features to plot
    features_to_plot = ['s_bias_mean', 'outer_fpt', 'inner_fpt']
    available_features = [f for f in features_to_plot if f + '_pred' in df.columns and f + '_actual' in df.columns]

    # Determine the number of plots needed based on available features
    num_plots = len(available_features)
    if num_plots == 0:
        print("No features available for plotting.")
        return

    # Create a figure with appropriate subplots
    fig, axs = plt.subplots(1, num_plots, figsize=(10 * num_plots, 6))  # Adjust figure size dynamically

    if num_plots == 1:
        axs = [axs]  # Make axs a list if only one plot

    # Define a function to plot the scatterplot and identity line
    def plot_scatter_identity(ax, x, y, hue, data):
        sns.scatterplot(x=x, y=y, hue=hue, data=data, palette="viridis", alpha=0.6, ax=ax)
        identity_line = np.linspace(data[x].min(), data[x].max(), 100)
        sns.lineplot(x=identity_line, y=identity_line, color="black", ax=ax)
        ax.set_xlabel(f"Predictions of {x.split('_')[0]}")
        ax.set_ylabel(f"Actuals of {y.split('_')[0]}")

    # Plot each of the available features
    for ax, feature in zip(axs, available_features):
        plot_scatter_identity(ax, f"{feature}_pred", f"{feature}_actual", 'State_Combination', df)

    # Unified title and legend
    fig.suptitle("Predictions vs. Actuals", fontsize=16)
    #handles, labels = axs[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for the title

    # Show and save the figure
    plt.show()
    fig.savefig(outfile)
