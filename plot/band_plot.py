"""
A module for plotting line plots with bands as confidence intervals.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def line_bar_plot(x,
                  y,
                  data,
                  ax=None,
                  dpi=100,
                  xlabel=None,
                  ylabel=None,
                  title=None):
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    grouped_data = data.groupby(x)
    line_values = grouped_data.mean()[y]
    band_values = grouped_data.std()[y]
    x_values = grouped_data.mean().index

    plt.plot(x_values,
             line_values,
             color="royalblue",
             lw=1.5)
    plt.fill_between(x_values,
                     line_values - band_values,
                     line_values + band_values,
                     color="royalblue",
                     alpha=0.25)

    ax.spines["top"].set_alpha(0.1)
    ax.spines["bottom"].set_alpha(1)
    ax.spines["right"].set_alpha(0.1)
    ax.spines["left"].set_alpha(1)

    ax.grid(alpha=0.5, linestyle='--')

    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    return fig, ax
