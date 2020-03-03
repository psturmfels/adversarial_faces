"""
A module for plotting line plots with bands as confidence intervals.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as st

def _bootstrap_confidence(x,
                          ci=0.95,
                          num_trials=1000):
    computed_means = np.zeros(num_trials)
    for i in range(num_trials):
        sampled_x = np.random.choice(x,
                                     size=len(x),
                                     replace=True)
        computed_means[i] = np.mean(sampled_x)
    sorted_means = np.sort(computed_means)

    low_index  = int((1.0 - ci) * 0.5 * num_trials)
    high_index = int((1.0 + ci) * 0.5 * num_trials)
    return sorted_means[low_index], sorted_means[high_index]

def _population_confidence(x,
                           ci=0.95):
    return st.t.interval(ci,
                         len(x) - 1,
                         loc=np.mean(x),
                         scale=st.sem(x))

def line_bar_plot(x,
                  y,
                  data,
                  ax=None,
                  dpi=100,
                  xlabel=None,
                  ylabel=None,
                  title=None,
                  use_bootstrap=False,
                  **kwargs):
    if 'color' not in kwargs:
        kwargs['color'] = 'royalblue'

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    grouped_data = data.groupby(x)
    line_values = grouped_data.mean()[y]
    x_values = grouped_data.mean().index

    lower_conf = []
    upper_conf = []
    for _, group in grouped_data:
        if use_bootstrap:
            confidence = _bootstrap_confidence(group[y].values)
        else:
            confidence = _population_confidence(group[y].values)

        lower_conf.append(confidence[0])
        upper_conf.append(confidence[1])

    plt.plot(x_values,
             line_values,
             lw=1.5,
             **kwargs)
    plt.fill_between(x_values,
                     lower_conf,
                     upper_conf,
                     alpha=0.25,
                     **kwargs)

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
