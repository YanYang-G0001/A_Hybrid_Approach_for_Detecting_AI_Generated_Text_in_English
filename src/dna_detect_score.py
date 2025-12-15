# Part of the function here is adapted from https://github.com/Xiaoweizhu57/DNA-DetectLLM
# For academic use only. 

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse
from utils import load_jsonl

class SimpleDNADetectLLM:
    # DNA-DetectLLM implementation

    def __init__(self, observer_model_name="gpt2-medium", performer_model_name="gpt2-xl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"load Observer: {observer_model_name}...")
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_model_name).to(self.device)
        self.observer_model.eval()

        print(f"load Performer: {performer_model_name}...")
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_model_name).to(self.device)
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def compute_score(self, text):
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        ).to(self.device)


        with torch.no_grad():
            observer_logits = self.observer_model(**encoding).logits
            performer_logits = self.performer_model(**encoding).logits


        ppl = self._sum_perplexity(encoding, performer_logits)
        x_ppl = self._entropy(observer_logits, performer_logits, encoding, self.tokenizer.pad_token_id)


        score = ppl / (2 * x_ppl)

        return score[0] if isinstance(score, np.ndarray) else score

    @staticmethod
    def _sum_perplexity(encoding, logits):
        shifted_logits = logits[..., :-1, :]
        attention = encoding.attention_mask[..., 1:]
        labels_std = encoding.input_ids[..., 1:]
        labels_max = torch.argmax(shifted_logits, dim=-1)

        logits_T = shifted_logits.transpose(1, 2)

        ce_std = F.cross_entropy(logits_T, labels_std, reduction='none')
        ce_max = F.cross_entropy(logits_T, labels_max, reduction='none')

        attn_sum = attention.sum(dim=1).clamp(min=1)
        ppl_std = (ce_std * attention).sum(dim=1) / attn_sum
        ppl_max = (ce_max * attention).sum(dim=1) / attn_sum

        return (ppl_std + ppl_max).cpu().numpy()

    @staticmethod
    def _entropy(p_logits, q_logits, encoding, pad_token_id):
        vocab_size = p_logits.shape[-1]
        total_tokens = q_logits.shape[-2]

        p_proba = F.softmax(p_logits, dim=-1).view(-1, vocab_size)
        q_scores = q_logits.view(-1, vocab_size)

        ce = F.cross_entropy(q_scores, p_proba, reduction='none').view(-1, total_tokens)
        padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

        agg_ce = ((ce * padding_mask).sum(1) / padding_mask.sum(1)).cpu().float().numpy()
        return agg_ce

def compute_and_save_scores(data_list, output_file, detector):
    """
    Compute DNA-DetectLLM scores for a dataset and save to a file.

    Args:
        data_list (list): List of data samples, each containing a 'text' field.
        output_file (str): Path to save the computed scores.
        detector (SimpleDNADetectLLM): Initialized DNA-DetectLLM detector.

    Returns:
        list: List of computed scores.
    """
    if os.path.exists(output_file):
        print(f"Loading existing scores from {output_file}")
        with open(output_file, 'r') as f:
            return json.load(f)

    print(f"Computing scores for {len(data_list)} samples...")
    scores = []
    for item in tqdm(data_list, desc="DNA-DETECT"):
        try:
            score = detector.compute_score(item['text'])
            scores.append(float(score))
        except Exception as e:
            print(f"Error: {e}")
            scores.append(0.5)  # Default value for errors

    with open(output_file, 'w') as f:
        json.dump(scores, f)
    print(f"Scores saved to {output_file}")

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer_model", type=str, default="gpt2-medium", help="Observer_model used for calculate dna-detect-score.")
    parser.add_argument("--performer_model", type=str, default="gpt2-large", help="Performer_model used for calculate dna-detect-score.")
    parser.add_argument("--train_data_path", type=str, default="/dataset/sampled_train_data.jsonl", help="Path to load the train data scores.")
    parser.add_argument("--val_data_path", type=str, default="/dataset/sampled_val_data.jsonl", help="Path to load the validation data scores.")
    parser.add_argument("--test_data_path", type=str, default="/dataset/sampled_test_data.jsonl", help="Path to load the test data scores.")
    
    args = parser.parse_args()
    # Initialize detector
    detector = SimpleDNADetectLLM(
        observer_model_name=args.observer_model, # or "EleutherAI/pythia-160m"
        performer_model_name=args.performer_model # or "EleutherAI/pythia-410m" 
    )

    # Load preprossessed data
    train_data = load_jsonl(args.train_data_path)
    val_data = load_jsonl(args.val_data_path)
    test_data = load_jsonl(args.test_data_path)

    # Compute scores
    train_scores = compute_and_save_scores(train_data, '/dataset/train_dna_scores.json', detector)
    val_scores = compute_and_save_scores(val_data, '/dataset/val_dna_scores.json', detector)
    test_scores = compute_and_save_scores(test_data, '/dataset/test_dna_scores.json', detector)

    # Normalize scores
    # scaler = StandardScaler()
    # train_scores_normalized = scaler.fit_transform(np.array(train_scores).reshape(-1, 1)).flatten()
    # val_scores_normalized = scaler.transform(np.array(val_scores).reshape(-1, 1)).flatten()
    # test_scores_normalized = scaler.transform(np.array(test_scores).reshape(-1, 1)).flatten()
    
    # print("\n=== Normalized Perplexity ===")
    # print(f"Train - Mean: {train_scores_normalized.mean():.4f}, Std: {train_scores_normalized.std():.4f}")
    # print(f"Val - Mean: {val_scores_normalized.mean():.4f}, Std: {val_scores_normalized.std():.4f}")
    # print(f"Test - Mean: {test_scores_normalized.mean():.4f}, Std: {test_scores_normalized.std():.4f}")