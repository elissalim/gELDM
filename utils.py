import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='', subject=None, fold=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_counter = 0
        self.subject = subject
        self.fold = fold

    def __call__(self, score, model, *args):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, *args)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'...EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, *args)
            self.counter = 0

    def save_checkpoint(self, score, model, *args):
        """Saves model when validation loss decrease."""
        self.save_counter += 1
        if self.verbose:
            self.trace_func(f'Validation accuracy decreased ({self.val_acc_min:.6f} --> {score:.6f}).  Saving model ...\n')
        if self.subject is not None:
            torch.save(model.state_dict(), self.path + f'best_s{self.subject}_kfold{self.fold}_geldm.pth')
            torch.save(args[0].state_dict(), self.path + f'best_s{self.subject}_kfold{self.fold}_ddpm.pth')
        self.val_acc_min = score

def end_epoch(ll_features_test, ll_labels_test, ll_features_rest_test, num_classes, epoch, subject, acc, f1, timestamp, dataset_name, model_name, interval=False):

    all_ll_features_test = torch.cat((ll_features_test, ll_features_rest_test), dim=0)
    all_ll_labels_test = torch.cat((ll_labels_test, torch.Tensor([num_classes] * ll_features_rest_test.shape[0])), dim=0)

    tsne = TSNE(n_components=2, n_iter=1000, random_state=0)
    fig, axs = plt.subplots(1, figsize=(8, 6))

    X_2d = tsne.fit_transform(all_ll_features_test)

    markers = ['^', '^', '^', '^', '.']
    cdict = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    unique_labels = torch.unique(all_ll_labels_test)

    for label in unique_labels:
        if label == unique_labels[-1]:
            continue
        else:
            idx = torch.nonzero(all_ll_labels_test == label)
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=cdict[int(label)], s=100, alpha=1,
                        marker=markers[int(label)])
    axs.title.set_text(f'Subject: {subject}, Epoch: {epoch}, Accuracy: {acc * 100:.2f}, F1-score: {f1 * 100:.2f}')
    fig.legend(loc=1, mode='expand', numpoints=1, ncol=4, fancybox=True, fontsize='small',
               labels=['LH_MI', 'RH_MI', 'F_MI', 'T_MI'])
    axs.set(xticklabels=[])
    axs.set(yticklabels=[])
    axs.grid(False)

    if interval:
        plt.savefig(f'./results/{dataset_name}_{model_name}_{timestamp}/s{subject}_e{epoch}.png', dpi=300)
    else:
        plt.savefig(f'./results/{dataset_name}_{model_name}_{timestamp}/s{subject}.png', dpi=300)
    plt.close()
