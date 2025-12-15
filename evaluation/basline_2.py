# Baseline 2: Pretrained DeBERTa (No Perplexity)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import tqdm
from src.model import load_pretrained_components
from src.utils import load_jsonl

def evaluate_deberta_baseline(model, data_list, tokenizer, batch_size=32, threshold=0.3):
    """evaluate DeBERTa baseline model on given data."""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(data_list), batch_size), desc="DeBERTa Baseline"):
            batch = data_list[i:i+batch_size]
            texts = [item['text'] for item in batch]
            labels = [item['label'] for item in batch]

            # Tokenize
            inputs = tokenizer(
                texts,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)

            # Forward
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(AI)

            # Threshold-based prediction
            preds = (probs > threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Calculate metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }

    return results, all_preds, all_probs

print("=== Baseline 2: Pretrained DeBERTa (No Perplexity) ===")
print("Evaluating on test set...")
tokenizer, config, pretrained_model = load_pretrained_components(model_name="OU-Advacheck/deberta-v3-base-daigenc-mgt1a")
test_data = load_jsonl('/dataset/test_set_en_with_label.jsonl')

deberta_results, deberta_preds, deberta_probs = evaluate_deberta_baseline(
    pretrained_model,
    test_data,
    tokenizer,
    threshold=0.3 # set threshold which can reach the best f1-score
)

print(f"\nTest Results (threshold=0.3):")
print(f"  Accuracy:  {deberta_results['accuracy']:.4f}")
print(f"  Precision: {deberta_results['precision']:.4f}")
print(f"  Recall:    {deberta_results['recall']:.4f}")
print(f"  F1-Score:  {deberta_results['f1']:.4f}")
print(f"  AUC:       {deberta_results['auc']:.4f}")

print("Evaluating on val set...")
val_data = load_jsonl('/dataset/val_set_en_with_label.jsonl')

deberta_results, deberta_preds, deberta_probs = evaluate_deberta_baseline(
    pretrained_model,
    val_data,
    tokenizer,
    threshold=0.3 # set threshold which can reach the best f1-score
)

print(f"\nTest Results (threshold=0.3):")
print(f"  Accuracy:  {deberta_results['accuracy']:.4f}")
print(f"  Precision: {deberta_results['precision']:.4f}")
print(f"  Recall:    {deberta_results['recall']:.4f}")
print(f"  F1-Score:  {deberta_results['f1']:.4f}")
print(f"  AUC:       {deberta_results['auc']:.4f}")