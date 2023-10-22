import torch
from torch import Tensor
import matplotlib.pyplot as plt


def AUC(classifier_outputs: Tensor, labels: Tensor) -> tuple[float, Tensor]:
    """
    Computes ROC and AUC of given binary classifier.

    Args:
        classifier_outputs (Tensor [N]): Tensor of classifier outputs
        labels (Tensor [N]): Labels of given data. Must be numeric, with positive class having higher value.

    Returns:
        AUC (float [1]): Area under the curve.
        ROC (Tensor [2, N]): Receiver operator characteristics curve. (TPR, FPR) 
    """

    # Send to CPU for simplicity
    classifier_outputs = classifier_outputs.cpu()
    labels = labels.cpu()

    # Check if labels are correct and get positive class
    classes = torch.unique(labels)
    if len(classes) != 2: raise RuntimeError(f"Labels contain more then 2 labels: {classes}")
    positive_class = torch.max(classes)

    # Reorder ouputs from biggest to smallest
    order = torch.argsort(classifier_outputs, descending=True)
    labels = labels[order]

    # Compute ROC curve
    TPR = torch.cumsum(labels == positive_class, dim=0)
    FPR = torch.cumsum(labels != positive_class, dim=0)
    TPR = TPR / TPR[-1]
    FPR = FPR / FPR[-1]

    # Get AUC by the trapezoid rule
    auc = torch.trapezoid(TPR, FPR).item()
    
    return auc, torch.stack([TPR, FPR])

def show_ROC(TPR: Tensor, FPR: Tensor, auc: float = None) -> None:
    if not auc:
        auc = torch.trapezoid(TPR, FPR).item()

    plt.figure('ROC')
    plt.title(f"Receive operator characteric curve (AUC = {round(auc, 2)})", weight='bold', fontsize=20)
    plt.plot(FPR.numpy(), TPR.numpy(), color='blue', linewidth=2)
    plt.fill_between(FPR.numpy(), TPR.numpy(), alpha=0.75)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('False Positive Rate', weight='bold', fontsize=16)
    plt.ylabel('True Positive Rate', weight='bold', fontsize=16)
    plt.box(False)
    plt.grid(linestyle='dashed', alpha=0.5)
    plt.show()



if __name__ == "__main__":
    N = 1000
    outputs = torch.linspace(0, 1, N) + torch.rand(N)
    labels = torch.linspace(0, 1, N) > 0.5

    auc, roc = AUC(outputs, labels)
    
    show_ROC(roc[0, :], roc[1, :], auc)