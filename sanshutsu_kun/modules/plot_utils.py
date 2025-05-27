import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

def plot_and_save_graphs(y_test, y_pred, y_prob, train_losses, train_accuracies, run_dir):
    graph_dir = os.path.dirname(__file__)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # 1. ROC曲線
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_path = os.path.join(graph_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    run_roc_path = os.path.join(run_dir, 'roc_curve.png')
    shutil.copy(roc_path, run_roc_path)
    plt.close()

    # 2. PR曲線
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    pr_path = os.path.join(graph_dir, 'pr_curve.png')
    plt.savefig(pr_path)
    run_pr_path = os.path.join(run_dir, 'pr_curve.png')
    shutil.copy(pr_path, run_pr_path)
    plt.close()

    # 3. 学習曲線
    plt.figure()
    plt.plot(train_losses, label='Loss')
    plt.plot(train_accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Learning Curve')
    plt.legend()
    lc_path = os.path.join(graph_dir, 'learning_curve.png')
    plt.savefig(lc_path)
    run_lc_path = os.path.join(run_dir, 'learning_curve.png')
    shutil.copy(lc_path, run_lc_path)
    plt.close()

    # 4. 混同行列
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(graph_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    run_cm_path = os.path.join(run_dir, 'confusion_matrix.png')
    shutil.copy(cm_path, run_cm_path)
    plt.close()

    return [run_roc_path, run_pr_path, run_lc_path, run_cm_path] 