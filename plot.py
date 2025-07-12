import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def plot_losses(val_loss_curve: list[int], train_loss_curve: list[int], epochs: int, save_path: str | None = None):
    plt.plot([i for i in range(0, epochs)], train_loss_curve, label='Train Loss')
    plt.plot([i for i in range(0, epochs)], val_loss_curve, label='Value Loss')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def plot_roc_curve(model, dataloader, device="cpu", title="ROC Curve", save_path: str | None = None):
    """
    Plots ROC curve and prints AUC for a PyTorch model.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader yielding (inputs, labels).
        device: Device string, e.g., "cpu" or "cuda".
        title: Title of the plot.
    """
    model.eval()
    model.to(device)

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Handle binary and multi-class
            if len(outputs) == 1:
                probs = torch.sigmoid(outputs).squeeze()
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1 probability

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = all_labels
    y_scores = all_probs

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
def plot_multiclass_roc(model, dataloader, num_classes, device='cpu', title="ROC Curve", save_path: str | None =None):
    """
    Plots multiclass ROC curves given a model and dataloader.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with (X, y) batches
        num_classes: Number of target classes
        device: 'cpu' or 'cuda'
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    model.to(device)
    model.eval()

    all_probs = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0]
            y = batch[1]
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            # Apply softmax if not already applied in the model
            probs = torch.nn.functional.softmax(outputs, dim=1)

            all_probs.append(probs)
            all_true.append(y)

    all_probs = torch.cat(all_probs).cpu().numpy()
    all_true = torch.cat(all_true).cpu().numpy()

    # One-hot encode true labels
    y_true_bin = label_binarize(all_true, classes=np.arange(num_classes))

    # Plot ROC curves for each class
    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        auc = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend([], loc="lower right")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_metrics(metrics, save_path: str | None = None):
    legends = []
    for k, v in metrics.items():
        plt.plot(v)
        legends.append(k)
    plt.xlabel("Epoch")
    plt.legend(legends)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_pd_series_describe(series: pd.Series, save_path: str | None = None, title: str = "Pandas Series Describe Output"):
    """
    Plots a pandas Series describe output as a bar chart.
    
    Args:
        series: Pandas Series to plot.
        save_path: Path to save the plot image.
    """
    desc = series.describe()

    # The number of rows far outnumbers any metric of the dataset, so remove the count
    desc = desc.drop('count', errors='ignore')

    plt.bar(desc.index, desc.values)
    plt.xlabel("Statistics")
    plt.ylabel("Values")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=40)

    plt.show()
