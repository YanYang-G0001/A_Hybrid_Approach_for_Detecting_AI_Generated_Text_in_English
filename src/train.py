import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from tqdm.auto import tqdm

from model import MGTDatasetWithPerplexity, create_model_with_perplexity
from utils import load_jsonl, read_existed_scores


def load_and_prepare_data(train_data_path, val_data_path, train_scores_path, val_scores_path):
    """Load and normalize data with perplexity scores."""
    # Load data
    train_data = load_jsonl(train_data_path)
    val_data = load_jsonl(val_data_path)
    
    train_scores = read_existed_scores(train_scores_path)
    val_scores = read_existed_scores(val_scores_path)
    
    # Normalize scores
    scaler = StandardScaler()
    train_scores_normalized = scaler.fit_transform(np.array(train_scores).reshape(-1, 1)).flatten()
    val_scores_normalized = scaler.transform(np.array(val_scores).reshape(-1, 1)).flatten()
    
    return train_data, val_data, train_scores_normalized, val_scores_normalized


def create_dataloaders(train_data, val_data, train_scores, val_scores, tokenizer, batch_size=32):
    """Create training and validation DataLoaders."""
    train_dataset = MGTDatasetWithPerplexity(train_data, train_scores, tokenizer)
    val_dataset = MGTDatasetWithPerplexity(val_data, val_scores, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def setup_model_and_optimizer(train_loader, num_epochs=1, lr=3e-4, warmup_steps=50):
    """Setup model, freeze encoder, and create optimizer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = create_model_with_perplexity()
    model.to(device)
    
    print("=== Training Classifier Head Only ===")
    
    # Freeze DeBERTa layers
    for param in model.deberta.parameters():
        param.requires_grad = False
    for param in model.pooler.parameters():
        param.requires_grad = False
    
    # Setup optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training for {num_epochs} epoch(s), {total_steps} steps")
    
    return model, optimizer, scheduler, device


def train_one_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        perplexity_feature = batch['perplexity_feature'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            perplexity_feature=perplexity_feature,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Average Loss: {avg_loss:.4f}")
    
    return avg_loss


def validate(model, val_loader, device):
    """Validate model and return metrics."""
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            perplexity_feature = batch['perplexity_feature'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                perplexity_feature=perplexity_feature
            )
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
    
    val_f1 = f1_score(val_labels, val_preds)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    return val_acc, val_f1


def train_model(train_loader, val_loader, num_epochs=1, lr=3e-4, warmup_steps=50, save_path='best_model_stage1.pt'):
    """Main training function."""
    model, optimizer, scheduler, device = setup_model_and_optimizer(train_loader, num_epochs, lr, warmup_steps)
    
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc, val_f1 = validate(model, val_loader, device)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved (F1={val_f1:.4f})")
    
    print(f"\nTrain Complete. Best Val F1: {best_val_f1:.4f}")
    return model, best_val_f1


if __name__ == "__main__":
    # Data paths
    train_data_path = '/dataset/sampled_train_data.jsonl'
    val_data_path = '/dataset/sampled_val_data.jsonl'
    train_scores_path = '/dataset/train_dna_scores.json'
    val_scores_path = '/dataset/val_dna_scores.json'
    
    # Load and prepare data
    train_data, val_data, train_scores, val_scores = load_and_prepare_data(
        train_data_path, val_data_path, train_scores_path, val_scores_path
    )
    
    # Create DataLoaders
    tokenizer = AutoTokenizer.from_pretrained("OU-Advacheck/deberta-v3-base-daigenc-mgt1a")
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, train_scores, val_scores, tokenizer, batch_size=32
    )
    
    # Train model
    model, best_f1 = train_model(
        train_loader, val_loader, 
        num_epochs=1, 
        lr=3e-4, 
        warmup_steps=50,
        save_path='best_model_stage1.pt'
    )