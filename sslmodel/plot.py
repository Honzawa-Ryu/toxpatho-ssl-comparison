# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

visualization 

@author: Katsuhisa Morita, tadahaya
"""
import matplotlib.pyplot as plt

# plot
def plot_progress(train_loss, test_loss, outdir):
    """ plot learning progress """
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 18
    ax.plot(list(range(1, len(train_loss) + 1, 1)), train_loss, c='purple', label='train loss')
    ax.plot(list(range(1, len(test_loss) + 1, 1)), test_loss, c='orange', label='test loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir + '/progress.tif', dpi=100, bbox_inches='tight')

def plot_progress_train(train_loss, outdir):
    """ plot learning progress """
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 18
    ax.plot(list(range(1, len(train_loss) + 1, 1)), train_loss, c='purple', label='train loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir + '/progress_train.tif', dpi=100, bbox_inches='tight')

def plot_accuracy(scores, labels, outdir):
    """ plot learning progress """
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auroc = metrics.auc(fpr, tpr)
    precision, _, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(tpr, precision)
    fig, axes = plt.subplots(1, 2, tight_layout=True)
    plt.rcParams['font.size'] = 18
    axes[0, 1].plot(fpr, tpr, c='purple')
    axes[0, 1].set_title(f'ROC curve (area: {auroc:.3})')
    axes[0, 1].set_xlabel('FPR')
    axes[0, 1].set_ylabel('TPR')
    axes[0, 2].plot(tpr, precision, c='orange')
    axes[0, 2].set_title(f'PR curve (area: {aupr:.3})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    plt.grid()
    plt.savefig(outdir + '/accuracy.tif', dpi=100, bbox_inches='tight')
    df = pd.DataFrame({'labels':labels, 'predicts':scores})
    df.to_csv(outdir + '/predicted.txt', sep='\t')
    return auroc, aupr