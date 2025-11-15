import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    balanced_accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Evaluation focused on what matters for imbalanced data.
    
    Regular accuracy is useless here. A model that always predicts
    "nothing" would get ~50% accuracy but be completely worthless.
    """
    
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print(f"{'='*60}\n")
    
    # Balanced accuracy: average of per-class recalls
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Weighted F1: accounts for class imbalance
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    
    # Macro F1: treats all classes equally (shows if rare classes are learned)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"F1 Score (macro): {f1_macro:.4f}")
    
    print("\nPer-class breakdown:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False):
    """
    Visual confusion matrix. Normalize to see percentage errors,
    which is more useful than raw counts with imbalanced data.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        # Normalize by true class (row-wise)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()


def compare_models(results_dict):
    """
    Side-by-side comparison of multiple models.
    
    Args:
        results_dict: {'Model Name': {'metric': value, ...}, ...}
    """
    df = pd.DataFrame(results_dict).T
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(df.to_string())
    print()
    
    # Visual comparison
    df.plot(kind='bar', figsize=(10, 6), rot=0)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.tight_layout()
    return plt.gcf()