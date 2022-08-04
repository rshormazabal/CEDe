import seaborn as sns
import matplotlib.pyplot as plt


def plot_string_len_distribution(data, col_name):
    """
    Plot the string length distribution using seaborn.
    :param data: pandas DataFrame
    :param col_name: column name of the string DataFrame
    :return:
    """
    sns.set(style="darkgrid")
    ax = sns.distplot(data[col_name].str.len())
    ax.set_xlabel('Length of strings')
    ax.set_ylabel('Frequency')
    plt.show()
