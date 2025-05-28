import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from datetime import datetime
import numpy as np

def save_script_and_make_run_dir(base_dir, script_path=None):
    """
    base_dir: 保存先のベースディレクトリ（例: 'results' ディレクトリまで）
    script_path: コピーするスクリプトのパス（Noneならsys.argv[0]を自動取得）
    戻り値: run_dirのパス
    """
    import sys
    if script_path is None:
        script_path = os.path.abspath(sys.argv[0])
    run_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(script_path):
        shutil.copyfile(script_path, os.path.join(run_dir, os.path.basename(script_path)))
    else:
        print(f'警告: 実行ファイルが見つかりません: {script_path}')
    return run_dir

def plot_and_save_graphs(y_test, y_pred, y_prob, train_losses, train_accuracies, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    # 1. ROC曲線
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        run_roc_path = os.path.join(run_dir, 'roc_curve.png')
        plt.savefig(run_roc_path)
        plt.close()
    except Exception as e:
        print(f'ROC曲線描画エラー: {e}')
        run_roc_path = ''

    # 2. PR曲線
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        if len(precision) > 1 and len(recall) > 1 and not (np.all(np.isnan(precision)) or np.all(np.isnan(recall))):
            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            run_pr_path = os.path.join(run_dir, 'pr_curve.png')
            plt.savefig(run_pr_path)
            plt.close()
        else:
            print('PRカーブ用データが空または定数です')
            run_pr_path = ''
    except Exception as e:
        print(f'PRカーブ描画エラー: {e}')
        run_pr_path = ''

    # 3. 学習曲線
    try:
        if (train_losses and train_accuracies) and (len(train_losses) == 1 or len(train_accuracies) == 1):
            plt.figure()
            if train_losses:
                plt.plot([0], [train_losses[0]], 'o', label='Loss')
            if train_accuracies:
                plt.plot([0], [train_accuracies[0]], 'o', label='Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Learning Curve (single point)')
            plt.legend()
            run_lc_path = os.path.join(run_dir, 'learning_curve.png')
            plt.savefig(run_lc_path)
            plt.close()
        elif train_losses and train_accuracies and (len(set(train_losses)) > 1 or len(set(train_accuracies)) > 1):
            plt.figure()
            plt.plot(train_losses, label='Loss')
            plt.plot(train_accuracies, label='Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Learning Curve')
            plt.legend()
            run_lc_path = os.path.join(run_dir, 'learning_curve.png')
            plt.savefig(run_lc_path)
            plt.close()
        else:
            print('学習曲線用データが空です')
            run_lc_path = ''
    except Exception as e:
        print(f'学習曲線描画エラー: {e}')
        run_lc_path = ''

    # 4. 混同行列
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        run_cm_path = os.path.join(run_dir, 'confusion_matrix.png')
        plt.savefig(run_cm_path)
        plt.close()
    except Exception as e:
        print(f'混同行列描画エラー: {e}')
        run_cm_path = ''

    return [run_roc_path, run_pr_path, run_lc_path, run_cm_path]

# 進捗バー関数はprogress_utils.pyに移動します 