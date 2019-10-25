""" Plotting tool
"""
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def plot_roc(y_true, y_probas, label='ROC Curve',
             ax=None, figsize=None,
             title_fontsize='large', text_fontsize='large'):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title('ROC Curves', fontsize=title_fontsize)
    fpr, tpr, _ = roc_curve(y_true, y_probas)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label=f'{label} (area = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)

    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize='medium')
    return ax
