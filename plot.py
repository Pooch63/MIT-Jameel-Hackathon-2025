import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

def plot_losses(val_loss_curve: list[int], train_loss_curve: list[int], EPOCHS: int):
    plt.plot([i for i in range(0, EPOCHS)], train_loss_curve, label='Train Loss')
    plt.plot([i for i in range(0, EPOCHS)], val_loss_curve, label='Value Loss')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_roc_curve(model, dataloader, device="cpu", title="ROC Curve"):
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
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Handle binary and multi-class
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs).squeeze()
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1 probability

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)

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
    plt.show()

def plot_metrics(metrics):
    legends = []
    for k, v in metrics.items():
        plt.plot(v)
        legends.append(k)
    plt.xlabel("Epoch")
    plt.legend(legends)