# Baseline 1: Use only Perplexity + Threshold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from src.utils import load_jsonl, read_existed_scores

def evaluate_perplexity_baseline(scores, labels, thresholds=None):
    """use perplexity score as a classifier with given thresholds."""
    scores = np.array(scores)
    labels = np.array(labels)

    if thresholds is None:
        # Find the best threshold on the training set
        best_threshold = None
        best_f1 = 0

        # Try different thresholds
        for percentile in range(10, 100, 5):
            threshold = np.percentile(scores, percentile)
            preds = (scores > threshold).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds = [best_threshold]
        print(f"  Best threshold from training: {best_threshold:.4f} (F1={best_f1:.4f})")

    results = []
    for threshold in thresholds:
        preds = (scores > threshold).astype(int)
        results.append({
            'threshold': threshold,
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
        })

    return results

# Load preprocessed data and existing scores
train_data = load_jsonl('/dataset/sampled_train_data.jsonl')
val_data = load_jsonl('/dataset/sampled_val_data.jsonl')
test_data = load_jsonl('/dataset/sampled_test_data.jsonl')

train_scores = read_existed_scores('/dataset/train_dna_scores.json')
val_scores = read_existed_scores('/dataset/val_dna_scores.json')
test_scores = read_existed_scores('/dataset/test_dna_scores.json')

print("=== Baseline 1: Perplexity Only ===")
print("\nFinding optimal threshold on train set...")


train_labels = [d['label'] for d in train_data]
train_results = evaluate_perplexity_baseline(train_scores, train_labels)
optimal_threshold = train_results[0]['threshold']

print(f"\nEvaluating on validation set with threshold={optimal_threshold:.4f}...")
val_labels = [d['label'] for d in val_data]
val_results = evaluate_perplexity_baseline(val_scores, val_labels, thresholds=[optimal_threshold])

print(f"\nValidation Results:")
print(f"  Accuracy:  {val_results[0]['accuracy']:.4f}")
print(f"  Precision: {val_results[0]['precision']:.4f}")
print(f"  Recall:    {val_results[0]['recall']:.4f}")
print(f"  F1-Score:  {val_results[0]['f1']:.4f}")

print(f"\nEvaluating on test set with threshold={optimal_threshold:.4f}...")
test_labels = [d['label'] for d in test_data]
test_results = evaluate_perplexity_baseline(test_scores, test_labels, thresholds=[optimal_threshold])
print(f"\nTest Results:")
print(f"  Accuracy:  {test_results[0]['accuracy']:.4f}")
print(f"  Precision: {test_results[0]['precision']:.4f}")
print(f"  Recall:    {test_results[0]['recall']:.4f}")
print(f"  F1-Score:  {test_results[0]['f1']:.4f}")